import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
from transformers import WavLMModel
from vocos import Vocos

def apply_mel_frequency_shift(mel_t, shift_bins):
    T = mel_t.shape[0]
    device = mel_t.device
    x = torch.arange(100, device=device).float().unsqueeze(0).repeat(T, 1)
    taper = torch.clamp((25.0 - torch.arange(100, device=device).float()) / 25.0, 0.0, 1.0).unsqueeze(0)
    effective_shift = shift_bins * taper
    
    x_shifted = torch.clamp(x - effective_shift, 0.0, 99.0)
    x_low = x_shifted.floor().long()
    x_high = torch.clamp(x_low + 1, 0, 99)
    weight = (x_shifted - x_low.float())
    
    m_low = torch.gather(mel_t, 1, x_low)
    m_high = torch.gather(mel_t, 1, x_high)
    mel_out = m_low * (1.0 - weight) + m_high * weight
    return mel_out

def main():
    device = "cpu"
    torch.set_num_threads(4)
    
    cb_path = "/home/antigravity/sharmt1411/voice-enhancement/checkpoints/yunyang_studio_codebook.pt"
    input_wav_path = "/home/antigravity/upload/1041.mp3"
    output_wav_path = "/home/antigravity/upload/1041_yunyang_warm.wav"
    
    print("[1] Loading WavLM and Vocos...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
    wavlm.eval()
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()
    
    print("[2] Loading Studio Codebook from", cb_path)
    cb = torch.load(cb_path, map_location=device)
    target_feats = cb["feats"].to(device)
    target_feats = F.normalize(target_feats, p=2, dim=-1)
    target_mels = cb["mels"].to(device)
    N = target_feats.shape[0]
    
    print("[3] Reading input whisper audio...")
    y_in, sr = sf.read(input_wav_path)
    if y_in.ndim > 1:
        y_in = y_in.mean(axis=1)
    if sr != 24000:
        y_24k = librosa.resample(y_in, orig_sr=sr, target_sr=24000)
    else:
        y_24k = y_in.copy()
        
    y_16k = librosa.resample(y_24k, orig_sr=24000, target_sr=16000)
    
    # Calculate Voicing Ratio from whisper audio
    hop = 256
    win = 1024
    S = np.abs(librosa.stft(y_24k, n_fft=win, hop_length=hop, win_length=win))
    freqs = librosa.fft_frequencies(sr=24000, n_fft=win)
    low_mask = (freqs >= 100) & (freqs < 1800)
    high_mask = (freqs >= 2800) & (freqs < 8000)
    e_low = np.sum(S[low_mask, :], axis=0)
    e_high = np.sum(S[high_mask, :], axis=0)
    v_ratio = np.log10((e_low + 1e-4) / (e_high + 1e-4))
    
    print("[4] Extracting WavLM Layer 11 features...")
    t_16k = torch.from_numpy(y_16k).float().unsqueeze(0)
    with torch.no_grad():
        out = wavlm(t_16k, output_hidden_states=True)
        feat_l11 = out.hidden_states[11].squeeze(0)
        
        t_24k = torch.from_numpy(y_24k).float().unsqueeze(0)
        mel_ref = vocos.feature_extractor(t_24k).squeeze(0).transpose(0, 1)
        
        feat_interp = F.interpolate(
            feat_l11.unsqueeze(0).transpose(1, 2),
            size=mel_ref.shape[0],
            mode="linear",
            align_corners=False
        ).squeeze(0).transpose(0, 1)
        feat_interp = F.normalize(feat_interp, p=2, dim=-1)
        
        print("[5] Target matching with Viterbi regularization...")
        sim = torch.matmul(feat_interp, target_feats.t())
        T_m = sim.shape[0]
        
        mels_matched = []
        prev_idx = None
        continuity_beta = 0.08
        tau = 0.005
        k = 2
        
        for t in range(T_m):
            curr_sim = sim[t].clone()
            if prev_idx is not None and continuity_beta > 0:
                target_center = prev_idx + 1
                if 0 <= target_center < N:
                    w_r = 6
                    s_w = max(0, target_center - w_r)
                    e_w = min(N, target_center + w_r + 1)
                    delta = torch.arange(s_w, e_w, device=device) - target_center
                    bonus = continuity_beta * torch.exp(- (delta**2) / (2.0 * (1.8**2)))
                    curr_sim[s_w:e_w] += bonus
                    
            topk_sim, topk_idx = torch.topk(curr_sim, k=k, dim=-1)
            weights = F.softmax(topk_sim / tau, dim=-1)
            mel_t = torch.sum(target_mels[topk_idx] * weights.unsqueeze(-1), dim=0)
            mels_matched.append(mel_t)
            prev_idx = topk_idx[0].item()
            
        full_mel = torch.stack(mels_matched, dim=0) # [T_m, 100]
        
        # [6] Frequency-selective temporal smoothing:
        # Channels 0-18: 3-frame sharp smoothing [0.25, 0.50, 0.25]
        # Channels 19-99: 5-frame crisp smoothing [0.10, 0.20, 0.40, 0.20, 0.10]
        full_mel_3d = full_mel.transpose(0, 1).unsqueeze(0)
        k_low = torch.tensor([0.25, 0.50, 0.25], device=device).view(1, 1, 3).repeat(19, 1, 1)
        k_high = torch.tensor([0.10, 0.20, 0.40, 0.20, 0.10], device=device).view(1, 1, 5).repeat(81, 1, 1)
        
        mel_low = F.conv1d(F.pad(full_mel_3d[:, :19, :], (1, 1), mode="replicate"), k_low, groups=19)
        mel_high = F.conv1d(F.pad(full_mel_3d[:, 19:, :], (2, 2), mode="replicate"), k_high, groups=81)
        mel_combined = torch.cat([mel_low, mel_high], dim=1).squeeze(0).transpose(0, 1) # [T_m, 100]
        
        # [7] Direct Spectral Equalization in Mel domain:
        # Boost chest resonance & warmth in channels 6-16 (+1.6 dB)
        mel_combined[:, 6:17] = mel_combined[:, 6:17] + 1.6
        # Suppress harsh metallic buzz in channels 40-65 (-2.2 dB)
        mel_combined[:, 40:66] = mel_combined[:, 40:66] - 2.2
        
        # [8] Physiological micro-prosody (vibrato + smoothed jitter + shimmer + question rise)
        t_sec = torch.arange(T_m, device=device).float() * (hop / 24000.0)
        vibrato = 0.22 * torch.sin(2.0 * np.pi * 4.8 * t_sec)
        
        torch.manual_seed(42)
        raw_noise = torch.randn(T_m, device=device)
        noise_k = torch.ones(5, device=device) / 5.0
        jitter = F.conv1d(raw_noise.view(1, 1, -1), noise_k.view(1, 1, 5), padding=2).squeeze() * 0.12
        
        # Question pitch inflection: Chinese question particle "话呀？" at sentence end (last 75 frames ~ 800ms)
        question_rise = torch.zeros(T_m, device=device)
        if T_m > 75:
            question_rise[-75:] = torch.linspace(0.0, 0.65, 75, device=device)
            
        total_shift = (vibrato + jitter + question_rise).unsqueeze(1)
        mel_prosodic = apply_mel_frequency_shift(mel_combined, total_shift)
        
        # Natural Shimmer (~0.25dB on channels 0-20)
        shimmer_raw = torch.randn(T_m, device=device)
        shimmer = F.conv1d(shimmer_raw.view(1, 1, -1), noise_k.view(1, 1, 5), padding=2).squeeze() * 0.25
        mel_prosodic[:, :20] = mel_prosodic[:, :20] + shimmer.unsqueeze(1)
        
        # [9] Strict Voiced/Unvoiced gating: clamp unvoiced frames down to silence floor
        if len(v_ratio) < T_m:
            v_ratio_full = np.pad(v_ratio, (0, T_m - len(v_ratio)), mode="edge")
        else:
            v_ratio_full = v_ratio[:T_m]
        v_smooth = np.convolve(v_ratio_full, np.ones(5)/5.0, mode="same")
        
        unvoiced_prob = 1.0 / (1.0 + np.exp((v_smooth - (-0.15)) * 8.0))
        uv_t = torch.from_numpy(unvoiced_prob).float().unsqueeze(1).to(device)
        
        floor_tensor = torch.full_like(mel_prosodic[:, :16], -12.0)
        mel_prosodic[:, :16] = mel_prosodic[:, :16] * (1.0 - uv_t) + floor_tensor * uv_t
        
        # [10] Neural Vocos Decoding
        print("[10] Decoding with Vocos...")
        out_audio = vocos.decode(mel_prosodic.transpose(0, 1).unsqueeze(0))[0].squeeze(0).cpu().numpy()
        
        # [11] Transparent mastering: Sub-bass 45Hz highpass + Silence gating + Peak norm
        sos_hp = butter(2, 45, btype="highpass", fs=24000, output="sos")
        out_clean = sosfilt(sos_hp, out_audio)
        
        # Dynamic energy silence gating to ensure silent pauses
        win_g = 512
        rms_env = [np.sqrt(np.mean(y_24k[i:i+win_g]**2)) if len(y_24k[i:i+win_g]) > 0 else 0.0 for i in range(0, len(y_24k), hop)]
        gate_f = np.clip((np.array(rms_env) - 0.002) / (0.006 - 0.002), 0.0, 1.0)
        gate_s = np.repeat(gate_f, hop)[:len(out_clean)]
        if len(gate_s) < len(out_clean):
            gate_s = np.pad(gate_s, (0, len(out_clean) - len(gate_s)), mode="edge")
        k_sz = int(0.02 * 24000)
        gate_smooth = np.convolve(gate_s, np.ones(k_sz)/k_sz, mode="same")
        out_clean = out_clean * (gate_smooth * 0.95 + 0.05)
        
        peak = np.max(np.abs(out_clean))
        if peak > 0:
            out_clean = out_clean * (0.88 / peak)
            
        sf.write(output_wav_path, out_clean, 24000)
        print(f"Generated {output_wav_path} successfully!")

if __name__ == "__main__":
    main()
