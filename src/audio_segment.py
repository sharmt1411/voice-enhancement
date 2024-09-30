from pydub import AudioSegment
from pydub.silence import split_on_silence

file = "data_breath_1.wav"
# file = "data_voice_1.wav"
audio_type = file.split("_")[1]
filename = f"../src_data/{file}"
# 加载音频文件
audio = AudioSegment.from_file(filename, format="wav")

# 设置静音检测的参数
# min_silence_len：静音的最短持续时间（毫秒）
# silence_thresh：静音的阈值（低于该音量认为是静音），单位为dBFS
# keep_silence：保持静音的时间长度（用于音频分段的过渡）
min_silence_len = 500  # 静音最短持续500毫秒
silence_thresh = audio.dBFS - 16  # 小于平均音量16dB的部分认为是静音
keep_silence = 10050  # 保持250毫秒的静音作为分段

# 根据静音拆分音频
audio_chunks = split_on_silence(audio,
                                min_silence_len=min_silence_len,
                                silence_thresh=silence_thresh,
                                keep_silence=keep_silence)

# 保存拆分的音频片段
for i, chunk in enumerate(audio_chunks):
    if i == 0:
        continue # 跳过第一个片段,会有空白不对齐
    chunk.export(f"../src_data/output_chunks/output_chunk_{i}_{audio_type}.wav", format="wav")

print("音频处理完成，已保存所有非静音片段。")
