import os
import time
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
from transformers import WavLMModel
from vocos import Vocos

DEFAULT_CODEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "target_codebook_full_l11.pt"
)

# Standard Audio EQ Biquad cookbook functions for mastering
def peaking_eq_sos(f0, Q, gain_db, fs=24000):
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def low_shelf_sos(f0, gain_db, fs=24000):
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((A + 1/A)*(1/0.707 - 1) + 2)
    b0 =    A*( (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha )
    b1 =  2*A*( (A-1) - (A+1)*cos_w0                       )
    b2 =    A*( (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha )
    a0 =        (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 =   -2*( (A-1) + (A+1)*cos_w0                       )
    a2 =        (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

def high_shelf_sos(f0, gain_db, fs=24000):
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((A + 1/A)*(1/0.707 - 1) + 2)
    b0 =    A*( (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha )
    b1 = -2*A*( (A-1) + (A+1)*cos_w0                       )
    b2 =    A*( (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha )
    a0 =        (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 =    2*( (A-1) - (A+1)*cos_w0                       )
    a2 =        (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

class StreamingArtificialLarynx:
    """
    Software Artificial Larynx (软件版人工喉)
    Real-time streaming Audio -> Audio converter without text/ASR/TTS bottleneck.
    Strictly causal, supports arbitrary human vocalizations, dialects, accents,
    non-speech sounds, sighing, and tone transitions.
    """
    def __init__(self, codebook_path=None, device=None, k=4, tau=0.015, layer=11, pitch_shift_steps=0.0, num_threads=4):
        torch.set_num_threads(num_threads)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if codebook_path is None:
            codebook_path = DEFAULT_CODEBOOK_PATH
            if not os.path.exists(codebook_path):
                alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "target_codebook.pt")
                if os.path.exists(alt_path):
                    codebook_path = alt_path

        print(f"[ArtificialLarynx] Initializing on {self.device} (CPU threads: {torch.get_num_threads()})...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(self.device)
        self.wavlm.eval()
        
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
        self.vocos.eval()
        
        print(f"[ArtificialLarynx] Loading codebook from {codebook_path}...")
        cb = torch.load(codebook_path, map_location=self.device)
        self.target_feats = cb["feats"].to(self.device) # [N, 768]
        self.target_mels = cb["mels"].to(self.device)   # [N, 100]
        self.k = k
        self.tau = tau
        self.layer = layer
        self.pitch_shift_steps = pitch_shift_steps
        
        # 3-frame Mel temporal smoothing kernel
        self.smooth_kernel = torch.tensor([0.22, 0.56, 0.22], device=self.device).view(1, 1, 3).repeat(100, 1, 1)
        
        # Acoustic mastering filter coefficients
        self.sos_sub = butter(2, 45, btype="highpass", fs=24000, output="sos")
        self.sos_warmth = low_shelf_sos(180, 2.0, fs=24000)
        self.sos_debuzz = peaking_eq_sos(2500, 1.4, -3.0, fs=24000)
        self.sos_air = high_shelf_sos(6000, 2.0, fs=24000)
        
        print(f"[ArtificialLarynx] Ready! Codebook contains {self.target_feats.shape[0]} natural speech frames.")
        
        # Pre-cache common resamplers
        self.resamplers_16k = {
            16000: None,
            24000: torchaudio.transforms.Resample(24000, 16000).to(self.device),
            44100: torchaudio.transforms.Resample(44100, 16000).to(self.device),
            48000: torchaudio.transforms.Resample(48000, 16000).to(self.device),
        }
        self.resamplers_24k = {
            24000: None,
            16000: torchaudio.transforms.Resample(16000, 24000).to(self.device),
            44100: torchaudio.transforms.Resample(44100, 24000).to(self.device),
            48000: torchaudio.transforms.Resample(48000, 24000).to(self.device),
        }

    def process_mel_stream(self, chunk_raw_np, sr=24000, left_ctx_samples=0):
        """
        Extract causal WavLM features and retrieve kNN target Mel frames for a chunk.
        """
        chunk_t = torch.from_numpy(chunk_raw_np).float().unsqueeze(0).to(self.device)
        
        r16 = self.resamplers_16k.get(sr)
        if r16 is None and sr != 16000:
            r16 = torchaudio.transforms.Resample(sr, 16000).to(self.device)
            self.resamplers_16k[sr] = r16
        chunk_16k = r16(chunk_t) if r16 is not None else chunk_t
        
        r24 = self.resamplers_24k.get(sr)
        if r24 is None and sr != 24000:
            r24 = torchaudio.transforms.Resample(sr, 24000).to(self.device)
            self.resamplers_24k[sr] = r24
        chunk_24k = r24(chunk_t) if r24 is not None else chunk_t
        
        with torch.no_grad():
            out = self.wavlm(chunk_16k, output_hidden_states=True)
            feat = out.hidden_states[self.layer].squeeze(0) # [T_w, 768]
            
            mel_ref = self.vocos.feature_extractor(chunk_24k).squeeze(0).transpose(0, 1) # [T_m, 100]
            
            feat_interp = F.interpolate(
                feat.unsqueeze(0).transpose(1, 2),
                size=mel_ref.shape[0],
                mode="linear",
                align_corners=False
            ).squeeze(0).transpose(0, 1)
            
            # If left context is provided, slice target portion only
            if left_ctx_samples > 0:
                left_frames = int(left_ctx_samples * 24000 / sr / 256)
                feat_tgt = feat_interp[left_frames:]
            else:
                feat_tgt = feat_interp
                
            feat_norm = F.normalize(feat_tgt, p=2, dim=-1)
            sim = torch.matmul(feat_norm, self.target_feats.t())
            topk_sim, topk_idx = torch.topk(sim, k=self.k, dim=-1)
            weights = F.softmax(topk_sim / self.tau, dim=-1)
            out_mel = torch.sum(self.target_mels[topk_idx] * weights.unsqueeze(-1), dim=1) # [T_tgt, 100]
            
        return out_mel

    def apply_mastering_dsp(self, audio_np):
        """
        Apply biquad mastering chain and pitch enhancement.
        """
        y = sosfilt(self.sos_sub, audio_np)
        y = sosfilt(self.sos_warmth, y)
        y = sosfilt(self.sos_debuzz, y)
        y = sosfilt(self.sos_air, y)
        
        if self.pitch_shift_steps != 0.0:
            y = librosa.effects.pitch_shift(y, sr=24000, n_steps=self.pitch_shift_steps)
            
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (0.88 / peak)
        return y

    def process_file_streaming(self, input_wav_path, output_wav_path, chunk_len_ms=400, left_ctx_ms=400):
        """
        Process an entire audio file using causal chunk-by-chunk streaming with left lookback context.
        Returns: (output_wav_path, end_to_end_rtf, avg_chunk_latency_ms)
        """
        y_np, sr = sf.read(input_wav_path)
        if y_np.ndim > 1:
            y_np = y_np.mean(axis=1)
            
        hop_samples = int(chunk_len_ms * sr / 1000)
        left_ctx_samples = int(left_ctx_ms * sr / 1000)
        total_samples = len(y_np)
        dur_total_s = total_samples / sr
        
        all_mels = []
        latencies = []
        
        t_start = time.time()
        for idx in range(0, total_samples, hop_samples):
            t0 = time.time()
            start_ctx = max(0, idx - left_ctx_samples)
            actual_left = idx - start_ctx
            chunk_raw = y_np[start_ctx : idx + hop_samples]
            target_raw = chunk_raw[actual_left:]
            
            if len(target_raw) < int(sr * 0.03):
                continue
                
            out_mel = self.process_mel_stream(chunk_raw, sr=sr, left_ctx_samples=actual_left)
            all_mels.append(out_mel)
            latencies.append(time.time() - t0)
            
        # Continuous Mel temporal smoothing
        full_mel = torch.cat(all_mels, dim=0) # [Total_T, 100]
        full_mel_3d = full_mel.transpose(0, 1).unsqueeze(0)
        mel_smooth = F.conv1d(F.pad(full_mel_3d, (1, 1), mode="replicate"), self.smooth_kernel, groups=100)
        
        with torch.no_grad():
            raw_audio = self.vocos.decode(mel_smooth)[0].squeeze(0).cpu().numpy()
            
        full_out = self.apply_mastering_dsp(raw_audio)
        
        total_time = time.time() - t_start
        overall_rtf = total_time / max(dur_total_s, 1e-4)
        avg_chunk_lat = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0
        
        sf.write(output_wav_path, full_out, 24000)
        print(f"[ArtificialLarynx] Streamed {dur_total_s:.2f}s in {total_time:.2f}s | End-to-End RTF: {overall_rtf:.3f} | Avg Chunk Latency: {avg_chunk_lat:.1f}ms")
        return output_wav_path, overall_rtf, avg_chunk_lat

    def convert_audio(self, input_wav_path, output_wav_path):
        """
        Convert full audio file in streaming chunks.
        """
        return self.process_file_streaming(input_wav_path, output_wav_path, chunk_len_ms=400)[:2]
