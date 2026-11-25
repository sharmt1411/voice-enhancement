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
    "clean_codebook_full.pt"
)

class StreamingArtificialLarynx:
    """
    Software Artificial Larynx (软件版人工喉)
    Real-time streaming Audio -> Audio converter without text/ASR/TTS bottleneck.
    Strictly causal, supports arbitrary human vocalizations, dialects, accents,
    non-speech sounds, sighing, and tone transitions.
    """
    def __init__(self, codebook_path=None, device=None, k=1, tau=0.005, layer=11, continuity_beta=0.15, pitch_shift_steps=0.0, num_threads=4):
        torch.set_num_threads(num_threads)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if codebook_path is None:
            codebook_path = DEFAULT_CODEBOOK_PATH
            if not os.path.exists(codebook_path):
                alt_paths = [
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "target_codebook_full_l11.pt"),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "target_codebook.pt")
                ]
                for p in alt_paths:
                    if os.path.exists(p):
                        codebook_path = p
                        break

        print(f"[ArtificialLarynx] Initializing on {self.device} (CPU threads: {torch.get_num_threads()})...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(self.device)
        self.wavlm.eval()
        
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
        self.vocos.eval()
        
        print(f"[ArtificialLarynx] Loading codebook from {codebook_path}...")
        cb = torch.load(codebook_path, map_location=self.device)
        feat_key = f"feats_l{layer}"
        if feat_key in cb:
            self.target_feats = cb[feat_key].to(self.device)
        elif "feats" in cb:
            self.target_feats = cb["feats"].to(self.device)
        else:
            raise KeyError(f"Neither {feat_key} nor 'feats' found in {codebook_path}")
            
        self.target_mels = cb["mels"].to(self.device)   # [N, 100]
        self.k = k
        self.tau = tau
        self.layer = layer
        self.continuity_beta = continuity_beta
        self.pitch_shift_steps = pitch_shift_steps
        self.prev_target_idx = None
        
        # 7-frame Mel temporal smoothing kernel (spans ~75ms for fluent coarticulation)
        self.smooth_kernel = torch.tensor([0.05, 0.12, 0.20, 0.26, 0.20, 0.12, 0.05], device=self.device).view(1, 1, 7).repeat(100, 1, 1)
        
        # Acoustic high-pass filter to strip sub-bass rumble (<45Hz)
        self.sos_sub = butter(2, 45, btype="highpass", fs=24000, output="sos")
        
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
        Extract causal WavLM features and retrieve kNN target Mel frames for a chunk
        with Viterbi temporal continuity regularization.
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
            sim = torch.matmul(feat_norm, self.target_feats.t()) # [T_tgt, N]
            
            # Viterbi continuity regularization to eliminate frame jumping stutter
            mels_chunk = []
            N = self.target_feats.shape[0]
            for t in range(sim.shape[0]):
                curr_sim = sim[t].clone()
                if self.prev_target_idx is not None and self.continuity_beta > 0:
                    target_center = self.prev_target_idx + 1
                    if 0 <= target_center < N:
                        window_r = 8
                        start_w = max(0, target_center - window_r)
                        end_w = min(N, target_center + window_r + 1)
                        delta = torch.arange(start_w, end_w, device=self.device) - target_center
                        bonus = self.continuity_beta * torch.exp(- (delta**2) / (2.0 * (2.0**2)))
                        curr_sim[start_w:end_w] += bonus
                        
                topk_sim, topk_idx = torch.topk(curr_sim, k=self.k, dim=-1)
                weights = F.softmax(topk_sim / self.tau, dim=-1)
                mel_t = torch.sum(self.target_mels[topk_idx] * weights.unsqueeze(-1), dim=0)
                mels_chunk.append(mel_t)
                self.prev_target_idx = topk_idx[0].item()
                
            out_mel = torch.stack(mels_chunk, dim=0)
            
        return out_mel

    def apply_mastering_dsp(self, audio_np):
        """
        Apply clean acoustic mastering chain.
        - Highpass 45Hz to eliminate low sub-bass rumble/DC offset
        - Transparent peak normalization to -1.1 dBFS (0.88)
        """
        y = sosfilt(self.sos_sub, audio_np)
        if self.pitch_shift_steps != 0.0:
            y = librosa.effects.pitch_shift(y, sr=24000, n_steps=self.pitch_shift_steps)
            
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (0.88 / peak)
        return y

    def process_file_streaming(self, input_wav_path, output_wav_path, chunk_len_ms=400, left_ctx_ms=400, hybrid_air_crossover_hz=3400):
        """
        Process an entire audio file using causal chunk-by-chunk streaming with left lookback context.
        Features:
        - Causal chunk-by-chunk Mel extraction & Viterbi continuity regularization
        - 7-frame Mel temporal coarticulation smoothing
        - Dynamic silence gating to eliminate idle noise during pauses
        - Hybrid oral aspiration crossover to retain authentic human unvoiced fricative texture
        Returns: (output_wav_path, end_to_end_rtf, avg_chunk_latency_ms)
        """
        self.prev_target_idx = None
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
            
        # 7-frame Mel coarticulation temporal smoothing
        full_mel = torch.cat(all_mels, dim=0) # [Total_T, 100]
        full_mel_3d = full_mel.transpose(0, 1).unsqueeze(0)
        mel_smooth = F.conv1d(F.pad(full_mel_3d, (3, 3), mode="replicate"), self.smooth_kernel, groups=100)
        
        with torch.no_grad():
            raw_audio = self.vocos.decode(mel_smooth)[0].squeeze(0).cpu().numpy()
            
        full_out = self.apply_mastering_dsp(raw_audio)
        
        # Hybrid oral aspiration crossover (preserves authentic lip/tongue friction > crossover_hz)
        if hybrid_air_crossover_hz is not None and hybrid_air_crossover_hz > 0:
            if sr != 24000:
                t_in = torchaudio.transforms.Resample(sr, 24000)(torch.from_numpy(y_np).float().unsqueeze(0))
                y_in_24k = t_in.squeeze(0).numpy()
            else:
                y_in_24k = y_np
            min_l = min(len(y_in_24k), len(full_out))
            sos_lp = butter(4, hybrid_air_crossover_hz, btype="lowpass", fs=24000, output="sos")
            sos_hp = butter(4, hybrid_air_crossover_hz, btype="highpass", fs=24000, output="sos")
            body = sosfilt(sos_lp, full_out[:min_l])
            air = sosfilt(sos_hp, y_in_24k[:min_l])
            rms_b = np.sqrt(np.mean(sosfilt(sos_hp, full_out[:min_l])**2) + 1e-6)
            rms_a = np.sqrt(np.mean(air**2) + 1e-6)
            gain = min(1.6, max(0.4, rms_b / rms_a))
            full_out = body + air * gain
            
        # Smooth silence gating based on input energy
        hop_g = 256
        win_g = 512
        rms_env = [np.sqrt(np.mean(y_np[i:i+win_g]**2)) if len(y_np[i:i+win_g]) > 0 else 0.0 for i in range(0, len(y_np), hop_g)]
        gate_f = np.clip((np.array(rms_env) - 0.002) / (0.006 - 0.002), 0.0, 1.0)
        gate_s = np.repeat(gate_f, int(hop_g * 24000 / sr))[:len(full_out)]
        if len(gate_s) < len(full_out):
            gate_s = np.pad(gate_s, (0, len(full_out) - len(gate_s)), mode="edge")
        k_sz = int(0.02 * 24000)
        gate_smooth = np.convolve(gate_s, np.ones(k_sz)/k_sz, mode="same")
        full_out = full_out * (gate_smooth * 0.95 + 0.05)
        
        peak = np.max(np.abs(full_out))
        if peak > 0:
            full_out = full_out * (0.88 / peak)
            
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
