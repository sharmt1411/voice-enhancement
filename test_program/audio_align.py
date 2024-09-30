import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# 加载两个音频样本
audio1_path = "../test_data/audio_fast.wav"  # 使用1秒的音频
audio2_path = "../test_data/audio_slow.wav"  # 使用2秒的音频

audio1, sr1 = librosa.load(audio1_path)
audio2, sr2 = librosa.load(audio2_path)
print("音频1的采样率：", sr1)
print("音频2的采样率：", sr2)
# 确保采样率相同
if sr1 != sr2:
    raise ValueError("两个音频文件的采样率必须相同。")

# 计算两个音频的Mel谱图
n_mels = 128
n_fft = 2048
hop_length = 512

mel1 = librosa.feature.melspectrogram(y=audio1, sr=sr1, n_mels=n_mels)
mel2 = librosa.feature.melspectrogram(y=audio2, sr=sr2, n_mels=n_mels)

# 计算两个谱图之间的DTW
D, wp = librosa.sequence.dtw(X=librosa.power_to_db(mel1, ref=np.max),
                             Y=librosa.power_to_db(mel2, ref=np.max))

# 可视化对齐路径
plt.figure()
plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.power_to_db(mel1, ref=np.max), sr=sr1, x_axis='time', y_axis='mel')
plt.title('快速音频 Mel-Spectrogram')

plt.subplot(1, 2, 2)
librosa.display.specshow(librosa.power_to_db(mel2, ref=np.max), sr=sr2, x_axis='time', y_axis='mel')
plt.title('慢速音频 Mel-Spectrogram')

plt.figure()
librosa.display.specshow(D, x_axis='time', y_axis='time', cmap='gray_r')
plt.plot(wp[:, 1], wp[:, 0], marker='o', color='r')
plt.title('DTW 代价矩阵和最优路径')


# 根据wp对音频进行对齐
aligned_mel1 = librosa.util.sync(mel1, wp[:, 0])
aligned_mel2 = librosa.util.sync(mel2, wp[:, 1])

# 恢复原始音频
audio1_aligned = librosa.feature.inverse.mel_to_audio(aligned_mel1, n_fft=n_fft, hop_length=hop_length, sr=sr1)
audio2_aligned = librosa.feature.inverse.mel_to_audio(aligned_mel2, n_fft=n_fft, hop_length=hop_length, sr=sr2)

# 保存对齐后的音频
sf.write(f'../test_data/audio1_aligned.wav', audio1_aligned, sr1)
sf.write(f'../test_data/audio2_aligned.wav', audio2_aligned, sr2)

# 可视化对齐路径
plt.figure()
plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.power_to_db(aligned_mel1, ref=np.max), sr=sr1, x_axis='time', y_axis='mel')
plt.title('快速音频 Mel-Spectrogram')

plt.subplot(1, 2, 2)
librosa.display.specshow(librosa.power_to_db(aligned_mel2, ref=np.max), sr=sr2, x_axis='time', y_axis='mel')
plt.title('慢速音频 Mel-Spectrogram')
plt.show()
