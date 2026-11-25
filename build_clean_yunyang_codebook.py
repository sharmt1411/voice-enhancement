import os
import json
import asyncio
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
import edge_tts
from transformers import WavLMModel
from vocos import Vocos
from tqdm import tqdm

device = "cpu"
torch.set_num_threads(4)

OUT_DIR = "/home/antigravity/yunyang_cache"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Collect all texts from transcriptions.json
with open("/home/antigravity/transcriptions.json", "r") as f:
    trans = json.load(f)

texts = []
# Collect texts from all normal segments
for k in ["audio_normal_1.wav", "audio_normal_2.wav", "audio_normal_3.wav", "audio_normal_9.wav"]:
    segs = trans.get(k, {}).get("segments", [])
    for s in segs:
        t = s.get("text", "").strip()
        if len(t) >= 2 and t not in texts:
            texts.append(t)

# Also add common conversational phonemes and conversational phrases to guarantee full phonetic coverage
extra_phrases = [
    "你好，你是谁？你现在能听到我说话吗？",
    "喂，你好，请问你是哪一位？",
    "今天天气真不错，我们一起去公园散步吧。",
    "非常感谢你的帮助，祝你今天过得愉快。",
    "能听见我说的话吗？声音清不清楚？",
    "我们现在正在测试智能人工喉系统的发音效果。",
    "一二三四五，上山打老虎，老虎没打到，打到小松鼠。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "请问现在是几点钟了？今天星期几？",
    "能不能麻烦你再说一遍，我刚才没有听清楚。"
]
for ep in extra_phrases:
    if ep not in texts:
        texts.append(ep)

print(f"Total unique texts to synthesize: {len(texts)}")

async def gen_tts_batch(batch):
    tasks = []
    for idx, (t, out_p) in enumerate(batch):
        if not os.path.exists(out_p):
            comm = edge_tts.Communicate(t, "zh-CN-YunyangNeural")
            tasks.append(comm.save(out_p))
    if tasks:
        await asyncio.gather(*tasks)

async def main_synth():
    batch = []
    for idx, t in enumerate(texts):
        out_p = os.path.join(OUT_DIR, f"yy_{idx:03d}.mp3")
        batch.append((t, out_p))
        if len(batch) >= 15:
            await gen_tts_batch(batch)
            batch = []
            await asyncio.sleep(0.3)
    if batch:
        await gen_tts_batch(batch)

asyncio.run(main_synth())
print("All Yunyang speech synthesized successfully!")

# 2. Extract WavLM Layer 11 features and Vocos Mel
print("Loading WavLM and Vocos...")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
wavlm.eval()
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
vocos.eval()

all_feats = []
all_mels = []

resampler_16k = torchaudio.transforms.Resample(24000, 16000)

audio_files = sorted([os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.endswith(".mp3")])
print(f"Processing {len(audio_files)} audio files for codebook...")

for p in tqdm(audio_files):
    try:
        y, sr = sf.read(p)
    except:
        y, sr = librosa.load(p, sr=24000)
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != 24000:
        y_24k = librosa.resample(y, orig_sr=sr, target_sr=24000)
    else:
        y_24k = y.copy()
        
    t_24k = torch.from_numpy(y_24k).float().unsqueeze(0)
    t_16k = resampler_16k(t_24k)
    
    with torch.no_grad():
        out = wavlm(t_16k, output_hidden_states=True)
        feat_l11 = out.hidden_states[11].squeeze(0) # [Tw, 768]
        mel = vocos.feature_extractor(t_24k).squeeze(0).transpose(0, 1) # [Tm, 100]
        
        feat_interp = torch.nn.functional.interpolate(
            feat_l11.unsqueeze(0).transpose(1, 2),
            size=mel.shape[0],
            mode="linear",
            align_corners=False
        ).squeeze(0).transpose(0, 1) # [Tm, 768]
        
        all_feats.append(feat_interp)
        all_mels.append(mel)

full_feats = torch.cat(all_feats, dim=0)
full_feats = torch.nn.functional.normalize(full_feats, p=2, dim=-1)
full_mels = torch.cat(all_mels, dim=0)

print(f"Final Yunyang Codebook: {full_feats.shape[0]} frames (~{full_feats.shape[0]*256/24000:.1f}s)")
cb_save_path = "/home/antigravity/sharmt1411/voice-enhancement/checkpoints/yunyang_studio_codebook.pt"
torch.save({
    "feats": full_feats,
    "mels": full_mels,
    "spk": "zh-CN-YunyangNeural"
}, cb_save_path)
print(f"Saved studio codebook to {cb_save_path} successfully!")
