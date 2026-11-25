import os
import time
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import butter, sosfilt
from transformers import WavLMModel
from vocos import Vocos

DEFAULT_CODEBOOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "target_codebook.pt")

class StreamingArtificialLarynx:
    """
    Software Artificial Larynx (软件版人工喉)
    Real-time streaming Audio -> Audio converter without text/ASR/TTS bottleneck.
    Strictly causal, supports arbitrary human vocalizations, dialects, accents,
    non-speech sounds, sighing, and tone transitions.
    """
    def __init__(self, codebook_path=None, device=None, k=4, tau=0.01, layer=6, num_threads=4):
        torch.set_num_threads(num_threads)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if codebook_path is None:
            codebook_path = DEFAULT_CODEBOOK_PATH
            if not os.path.exists(codebook_path):
                alt_path = "/home/antigravity/target_codebook.pt"
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
        
        # Highpass filter for removing sub-acoustic boxiness (<80Hz)
        self.hp_sos = butter(2, 80, btype="highpass", fs=24000, output="sos")

    def process_audio_chunk(self, chunk_waveform_np, sr=24000):
        """
        Process a single streaming chunk of input audio with adaptive energy gating.
        Returns: (output_audio_np, latency_seconds, rtf)
        """
        t0 = time.time()
        dur_s = len(chunk_waveform_np) / sr
        
        # Adaptive Energy Gate / VAD to keep inter-sentence pauses dead quiet
        rms = float(np.sqrt(np.mean(chunk_waveform_np**2)))
        gate = float(np.clip((rms - 0.0035) / 0.007, 0.0, 1.0))
        
        if gate < 0.01:
            expected_samples = int(len(chunk_waveform_np) * 24000 / sr)
            return np.zeros(expected_samples, dtype=np.float32), time.time() - t0, 0.0
            
        chunk_t = torch.from_numpy(chunk_waveform_np).float().unsqueeze(0).to(self.device)
        
        # Resample to 16k for WavLM and 24k for Vocos
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
            
            mel = self.vocos.feature_extractor(chunk_24k).squeeze(0).transpose(0, 1) # [T_m, 100]
            
            feat_interp = F.interpolate(
                feat.unsqueeze(0).transpose(1, 2),
                size=mel.shape[0],
                mode="linear",
                align_corners=False
            ).squeeze(0).transpose(0, 1)
            
            feat_norm = F.normalize(feat_interp, p=2, dim=-1)
            
            # kNN retrieval against natural target codebook
            sim = torch.matmul(feat_norm, self.target_feats.t())
            topk_sim, topk_idx = torch.topk(sim, k=self.k, dim=-1)
            weights = F.softmax(topk_sim / self.tau, dim=-1)
            
            out_mel = torch.sum(self.target_mels[topk_idx] * weights.unsqueeze(-1), dim=1)
            out_audio = self.vocos.decode(out_mel.transpose(0, 1).unsqueeze(0))[0].squeeze(0)
            
        out_np = out_audio.cpu().numpy()
        if gate < 1.0:
            out_np = out_np * gate
            
        latency_s = time.time() - t0
        rtf = latency_s / max(dur_s, 1e-4)
        return out_np, latency_s, rtf

    def process_file_streaming(self, input_wav_path, output_wav_path, chunk_len_ms=400, overlap_ms=50):
        """
        Simulate real-time chunk streaming conversion over a file with Hanning Overlap-Add.
        """
        y_np, sr = sf.read(input_wav_path)
        if y_np.ndim > 1:
            y_np = y_np.mean(axis=1)
            
        hop_samples_in = int(chunk_len_ms * sr / 1000)
        win_samples_in = int((chunk_len_ms + overlap_ms) * sr / 1000)
        overlap_samples_24k = int(overlap_ms * 24000 / 1000)
        
        fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, overlap_samples_24k)) ** 2
        fade_out = np.cos(0.5 * np.pi * np.linspace(0, 1, overlap_samples_24k)) ** 2
        
        total_samples = len(y_np)
        dur_total_s = total_samples / sr
        
        out_stream = []
        latencies = []
        prev_tail = None
        
        t_start = time.time()
        for idx in range(0, total_samples, hop_samples_in):
            chunk_in = y_np[idx : idx + win_samples_in]
            if len(chunk_in) < int(sr * 0.05): # skip tail < 50ms
                continue
            chunk_out, lat_s, rtf = self.process_audio_chunk(chunk_in, sr=sr)
            latencies.append(lat_s)
            
            if prev_tail is None:
                if len(chunk_out) > overlap_samples_24k:
                    out_stream.append(chunk_out[:-overlap_samples_24k])
                    prev_tail = chunk_out[-overlap_samples_24k:]
                else:
                    out_stream.append(chunk_out)
                    prev_tail = None
            else:
                cross = prev_tail * fade_out + chunk_out[:overlap_samples_24k] * fade_in
                out_stream.append(cross)
                if len(chunk_out) > 2 * overlap_samples_24k:
                    out_stream.append(chunk_out[overlap_samples_24k:-overlap_samples_24k])
                    prev_tail = chunk_out[-overlap_samples_24k:]
                else:
                    prev_tail = None
                    
        if prev_tail is not None:
            out_stream.append(prev_tail)
            
        total_time = time.time() - t_start
        overall_rtf = total_time / max(dur_total_s, 1e-4)
        avg_chunk_lat = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0
        
        full_out = np.concatenate(out_stream, axis=0) if out_stream else np.zeros(0)
        # Apply gentle highpass to eliminate sub-80Hz rumble
        if len(full_out) > 0:
            full_out = sosfilt(self.hp_sos, full_out)
            peak = np.max(np.abs(full_out))
            if peak > 0:
                full_out = full_out * (0.88 / peak)
            
        sf.write(output_wav_path, full_out, 24000)
        print(f"[ArtificialLarynx] Streamed {dur_total_s:.2f}s in {total_time:.2f}s | End-to-End RTF: {overall_rtf:.3f} | Avg Chunk Latency: {avg_chunk_lat:.1f}ms (Chunk: {chunk_len_ms}ms, Overlap: {overlap_ms}ms)")
        return output_wav_path, overall_rtf, avg_chunk_lat

    def convert_audio(self, input_wav_path, output_wav_path):
        """
        Convert full audio file in streaming chunks.
        """
        return self.process_file_streaming(input_wav_path, output_wav_path, chunk_len_ms=400)[:2]
