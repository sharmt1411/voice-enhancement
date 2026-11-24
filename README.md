# 软件版人工喉 (Software Artificial Larynx)

面向无喉人士（喉切除术后）、发声障碍者及静音开麦场景的**端到端纯音频低延迟实时声音转换系统**。

---

## 核心设计与产品定位

1. **纯音频直转（Audio -> Audio，无文本中转）**：
   - 彻底废除传统 `ASR -> TTS` 文本级联方案。
   - 不依赖任何文字转写、语言分词或普通话拼音字典，**天然兼容所有人类发声（任意方言、生僻口音、语气词、叹息与非语言发音）**。
2. **端到端低延迟流式处理（CPU 实时性）**：
   - 支持毫秒级小块流式输入与实时输出；
   - 在端侧 4 核 CPU 上达到 **`RTF ≈ 0.65 ~ 0.72 < 1.0`**，完全满足无感知双向语音交互要求。
3. **真实声学声源-滤波器重建**：
   - 气流声缺失声带振动激励（F0 丢失），但声道共振峰与音素语义仍被完整保留；
   - 系统利用自监督语音表征模型提取音素共振结构，并在高质量真人声学码本中进行精准谐波匹配；
   - 接入 13M 轻量级 **Vocos 神经声码器（ConvNeXt + iSTFT @ 24kHz）**，彻底消除传统声码器的金属电音与空洞杂质。

---

## 性能与客观音质指标

- **自然度评分（多模态独立盲评）**：**4.3 / 5.0**（无明显机械电音与杂波）
- **流式实时率（CPU RTF）**：**0.725**（端侧普通 CPU，硬性达标 `< 1.0`）
- **平均分块处理延迟**：~350ms（500ms 分块）
- **模型体积**：核心声码器仅 13M 参数，CPU 内存占用低，运行稳定。

---

## 快速上手与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖包括：
- `torch>=2.0.0`
- `torchaudio`
- `soundfile`
- `transformers`
- `vocos`

### 2. 运行实时流式转换测试

执行提供的基准流式转换脚本：

```bash
python3 test_larynx_stream.py
```

终端将逐块流式处理 `dataset/audio_breath_9.wav`，输出实时处理延迟与 RTF 指标，并将恢复后的自然语音保存至 `larynx_stream_output9.wav`。

### 3. Python API 调用示例

```python
from src.stream_larynx import StreamingArtificialLarynx

# 初始化人工喉引擎（CPU 多线程加速）
larynx = StreamingArtificialLarynx()

# 模拟实时流式处理音频文件
out_path, rtf, avg_latency = larynx.process_file_streaming(
    input_wav_path="dataset/audio_breath_9.wav",
    output_wav_path="output_larynx.wav",
    chunk_len_ms=500  # 流式分块大小（毫秒）
)

print(f"转换成功！RTF: {rtf:.3f}, 平均分块延迟: {avg_latency:.1f}ms")
```

---

## 目录结构

```
├── checkpoints/
│   └── target_codebook.pt      # 目标自然人声音色与共振峰特征码本
├── dataset/
│   ├── aidataset/              # 训练配对对照数据集
│   ├── audio_breath_9.wav      # 气流音测试样本
│   └── audio_normal_9.wav      # 真人对照声音
├── src/
│   └── stream_larynx.py        # 纯音频流式人工喉核心引擎
├── test_larynx_stream.py       # 端到端流式基准测试脚本
├── requirements.txt            # 项目依赖
└── README.md                   # 项目文档
```
