import wave
from datetime import datetime

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, sosfilt, correlate

n_mel = 128  # 频谱图的通道数


def load_audio(file_path, sr=16000):
    """加载音频文件，并将其重采样到指定的采样率"""
    audio, sr = librosa.load(file_path, sr=sr)
    # print("load_audio_shape",audio.shape)
    # print("load_audio_sr",sr)
    # print("load_audio_content", audio[5001:5005])
    return audio, sr


def normalize_audio(audio):
    """音频归一化处理：将音频幅度归一化到[-1, 1]之间"""
    max_val = np.max(np.abs(audio))
    return audio / max_val


def lms_denoise(noisy_signal, reference_noise=None, mu=0.01, filter_order=10):
    """
    使用LMS自适应算法来进行噪声消除。
    """
    if reference_noise == None:
        audio, sr = load_audio("../test_data/noise-10s.wav")
        reference_noise = audio

    # 初始化变量
    n = len(noisy_signal)
    w = np.zeros(filter_order)  # 滤波器权重
    output = np.zeros(n)

    # 自适应噪声消除过程
    for i in range(filter_order, n):
        x = reference_noise[i-filter_order:i][::-1]  # 参考噪声信号
        # print("x_shape",x.shape)
        y = np.dot(w, x)  # 预测噪声
        # print("y_shape",y.shape)
        e = noisy_signal[i] - y  # 误差（噪声消除后的信号）
        w += 2 * mu * e * x  # 更新滤波器权重
        output[i] = e

    return output


def noise_reduction(audio, noise_reduction_factor=0.95):
    """简单的噪声去除算法：通过减小低能量部分来减少噪声"""
    # 获取音频的均方根能量
    rms_energy = np.sqrt(np.mean(audio ** 2))
    # 阈值设定为音频均方根能量的指定比例
    threshold = rms_energy * noise_reduction_factor
    # 对低于阈值的部分进行去噪处理
    processed_audio = np.where(np.abs(audio) < threshold, 0, audio)
    return processed_audio


def realtime_highpass_filter(audio, filter_order=10, sr=16000, cutoff=100):
    """
    实时高通滤波器，去除低频噪声。
    """
    # 设计高通滤波器
    sos = butter(filter_order, cutoff, 'hp', fs=sr, output='sos')
    # 应用滤波器
    filtered_audio = sosfilt(sos, audio)
    return filtered_audio


def fft_denoise(audio, sr, low_freq=80, high_freq=8000):
    """
    简单的FFT降噪方法。
    """
    # FFT变换
    fft_audio = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1 / sr)

    # 设置频域的阈值，保留特定频段
    fft_audio[(freqs < low_freq) | (freqs > high_freq)] = 0

    # IFFT反变换
    denoised_audio = np.fft.ifft(fft_audio).real
    return denoised_audio


def audio_denoise(audio, method=None, reference_noise=None, mu=0.01, filter_order=10, noise_reduction_factor=0.95,
                  sr=16000, cutoff=100, low_freq=80, high_freq=8000):
    """
    根据选择的降噪方式对音频进行处理。

    参数:
    audio: 输入的音频数组
    method: 降噪方法 ('lms', 'noise_reduction', 'highpass', 'fft')
    reference_noise: 参考噪声，用于LMS算法
    mu: LMS算法的步长
    filter_order: 滤波器的阶数
    noise_reduction_factor: 噪声降低系数
    sr: 采样率
    cutoff: 高通滤波器的截止频率
    low_freq: FFT降噪的低频阈值
    high_freq: FFT降噪的高频阈值
    返回处理后的音频数组
    """

    if method is None:
        return audio
    for m in method:
        if m == 'lms':
            audio = lms_denoise(audio, reference_noise, mu, filter_order)
        elif m == 'noise_reduction':
            audio = noise_reduction(audio, noise_reduction_factor)
        elif m == 'highpass':
            audio = realtime_highpass_filter(audio, filter_order, sr, cutoff)
        elif m == 'fft':
            audio = fft_denoise(audio, sr, low_freq, high_freq)
        else:
            print("不支持的降噪方法: {}".format(method))
    # 保存重建的音频文件
    sf.write(f'../test_data/denoise_audio_{method}.wav', audio, sr)
    return audio


def perform_stft(audio, n_fft=2048, hop_length=512):
    """短时傅里叶变换(STFT)"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)  # 获取幅度和相位信息
    return magnitude, phase


def preprocess_audio(file, is_file_path=True):
    """音频预处理主函数"""
    # 加载音频
    if is_file_path:
        audio, sr = load_audio(file)
    else:
        audio, sr = file, 16000

    # 去噪
    audio_denoised = noise_reduction(audio)

    # 归一化
    audio_normalized = normalize_audio(audio_denoised)
    print("normalize_audio_shape",audio_normalized.shape,audio_normalized[0:10])
    print("stft_start_time",datetime.now())
    # 短时傅里叶变换
    magnitude, phase = perform_stft(audio_normalized)
    print("stft_end_time", datetime.now())
    # 返回预处理后的结果
    return magnitude, phase, sr


def mel_to_audio(mel_spectrogram, sr=16000, n_fft=2048, hop_length=512, n_iter=32):
    """
    使用Griffin-Lim算法将Mel谱图转换回音频信号

    参数:
    - mel_spectrogram: 输入的Mel频谱图（二维数组）
    - sr: 采样率（默认16000 Hz）
    - n_fft: FFT的窗口大小（默认2048）
    - hop_length: 每一帧之间的跳步大小（默认512）
    - n_iter: Griffin-Lim算法的迭代次数（默认32）

    返回:
    - 重建的音频信号
    """
    # 将Mel谱图转换为stft
    mel_inverted_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=n_fft)

    # 使用Griffin-Lim算法估计相位并重建音频
    audio_reconstructed = librosa.griffinlim(mel_inverted_spectrogram, n_iter=n_iter, hop_length=hop_length)

    return audio_reconstructed


def audio_to_logmel_mel(audio, sr=16000, n_mels=64, fmax=8000):
    """
    将音频信号转换为Mel频谱图

    参数:
    - audio: 输入的音频信号
    - sr: 采样率（默认16000 Hz）
    - n_fft: FFT的窗口大小（默认2048）
    - hop_length: 每一帧之间的跳步大小（默认512）
    - n_mels: Mel滤波器的数量（默认32）
    - fmax: Mel滤波器的最高频率（默认8000 Hz）

    返回:
    - log-Mel频谱图（二维数组）, Mel频谱图（二维数组）
    """
    # 计算Mel频谱
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,  n_mels=n_mels, fmax=fmax)

    # 将幅度转换为dB
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram, mel_spectrogram


def audio_to_mel(audio, sr=16000, n_mels=64, fmax=8000):
    # 计算Mel频谱
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax)
    return mel_spectrogram


def mel_align(mel1, mel2):
    """
    对两个Mel频谱图进行对齐，使得它们的中心时间位置相同。
    1.用长序列去对齐短序列。
    2.根据对齐情况，截断尾部，保证尾部对齐。
    3.从对齐的尾部开始，截断头部，保证头部对齐。
    :param mel1: 第一个Mel频谱图, (n_mel, 时间维度)
    :param mel2: 第二个Mel频谱图, (n_mel, 时间维度)
    :return: 对齐后的两个Mel频谱图
    """
    # print("准备对齐，mel1", mel1, "mel2", mel2)
    len_mel1 = mel1.shape[1]  # (n_mel, 时间维度)
    len_mel2 = mel2.shape[1]  # (n_mel, 时间维度)

    # print("mel1数值范围", np.min(mel1), np.max(mel1),mel1[10000:10005])
    if len_mel1 > len_mel2:
        # 选择最长的Mel图，对齐较短的Mel图
        aligned_mel2, aligned_mel1 = mel_align(mel2, mel1)
        return aligned_mel1, aligned_mel2
    # print("mel1.mean(axis=0)", mel1.mean(axis=0), "mel2.mean(axis=0)", mel2.mean(axis=0))

    align_shape = True  # 是否对齐形状
    if align_shape:
        max1 = np.max(np.abs(mel1))
        max2 = np.max(np.abs(mel2))
        # 使用mel形状的进行对齐，即将mel图替换为1，0图
        mel1_mask = (mel1 > max1/1000).astype(np.float32)  # 阈值设置
        mel2_mask = (mel2 > max2/1000).astype(np.float32)  # 阈值设置
        corr = correlate(mel1_mask.sum(axis=0), mel2_mask.sum(axis=0))  # 长度为len(A) + len(B)-1,将b在a上滑动，并逐步计算两者的点积，结果是每个偏移位置的相关性值。
    else:
        # 计算两个 Mel 图之间的交叉相关
        # 由于mel图频率分布差异，所以不能平均后测试相关，待测试？而是直接求和后测试相关
        corr = correlate(mel1[64:128].sum(axis=0), mel2[64:128].sum(axis=0))  # 长度为len(A) + len(B)-1,将b在a上滑动，并逐步计算两者的点积，结果是每个偏移位置的相关性值。
    # print("corr", corr)
    # 找到最大相关值的位置（对应的时间偏移量）
    align = np.argmax(corr)+1  # 最大相关值的位置（对应B尾部在A头部上的的时间偏移量,从1开始）
    shift = align - len_mel1  # 尾端对齐的偏移量，>0表示B在A右侧，<0表示B在A左侧
    if abs(shift) > 6:
        # 防止出现偏移过大导致的对齐错误，一般为（128，50）形状，且音频收集预对齐过，所以偏移量不会超出太多
        print("mel_align: 无法对齐，shift偏移量过大，请检查输入")
        shift = 0  # 置零，相当于右对齐
    if shift > 0:
        mel1_aligned = mel1
        mel2_aligned = mel2[:, :-shift]
    elif shift < 0:
        mel1_aligned = mel1[:, :align]
        mel2_aligned = mel2
    else:
        mel1_aligned = mel1
        mel2_aligned = mel2

    # 截断到相同长度
    min_length = min(mel1_aligned.shape[1], mel2_aligned.shape[1])
    mel1_aligned = mel1_aligned[:, -min_length:]
    mel2_aligned = mel2_aligned[:, -min_length:]

    # print("对齐后mel1", mel1_aligned, "mel2", mel2_aligned, "shift", shift)

    show = False
    if show:
        mel_list = [(mel1_aligned, 'breath_before_align'), (mel2_aligned, 'normal_before_align'), (mel1, 'breath'),
                    (mel2, 'normal')]
        plot_mel_spectrogram_list(mel_list, is_log=False)
        print("Mel频谱图绘制完成")
    return mel1_aligned, mel2_aligned


def save_audio(audio, sr, file_path):
    """
    保存音频文件
    """
    sf.write(file_path, audio, sr)

    print(f"成功保存音频文件: {file_path}")


def plot_mel_spectrogram_list(mel_list, sr=16000, n_mels=128, fmax=8000, is_log=True):
    """
    绘制Mel频谱图

    参数:
    - mel_list: 输入的Mel频谱图列表（log-mel图，message），如(log-mel，”lms_denoise")
    - sr: 采样率（默认16000 Hz）
    - n_mels: Mel滤波器的数量（默认32）
    - fmax: Mel滤波器的最高频率（默认8000 Hz）
    """
    if not is_log:
        log_mel_list = [(librosa.amplitude_to_db(mel_spectrogram, ref=np.max), method) for mel_spectrogram, method in mel_list]
    else:
        log_mel_list = mel_list
    # 绘制Mel频谱图
    plt.figure(figsize=(10, 4*len(log_mel_list)))
    for i, (log_mel_spectrogram, method) in enumerate(log_mel_list):
        plt.subplot(len(log_mel_list), 1, i + 1)
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
        plt.title(f'Mel spectrogram-{method}')
        plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


# mel图数据分布统计直方图
def plot_mel_hist(mel_list):
    """
    绘制Mel频谱图的直方图
    参数:
    - mel_list: 输入的Mel频谱图列表（log-mel图，message），如(log-mel，”lms_denoise")
    """
    plt.figure(figsize=(10, 4*len(mel_list)))
    for i, (mel_spectrogram, message) in enumerate(mel_list):
        plt.subplot(len(mel_list), 1, i + 1)
        plt.hist(mel_spectrogram.flatten(), bins=2000, density=True, alpha=0.5, label=message)
        plt.xlim(-1, 1)  # 设置X轴的最小值和最大值
        plt.grid()  # 显示网格
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.legend(loc='upper right')
        plt.title(f'Mel spectrogram-{message} Histogram')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    def test_audio_denoise():
        # 请替换为你自己的音频文件路径
        audio_path = "../test_data/example-breath.wav"
        y, sr = librosa.load(audio_path)
        print("y", y.shape, sr, "\n", y[8000:8004])

        # 音频处理主函数
        # print("load_start_time",datetime.now())
        # # 调用预处理函数
        # magnitude, phase, sr = preprocess_audio(audio_file_path)
        # print("Magnitude shape:", magnitude.shape)
        # print("Phase shape:", phase.shape)
        #
        # # 使用plt绘制图形
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(magnitude, sr=sr, x_axis='time', y_axis='log')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Log-frequency power spectrogram')
        # plt.tight_layout()
        # plt.show()

        print("测试各个降噪方法用时--------------------------------------------")
        method = ['lms', 'noise_reduction', 'highpass', 'fft']
        y, sr = librosa.load(audio_path)
        import time
        for m in method:
            print("测试方法：", m, "开始-")
            start_time = time.time()
            audio_denoise_result = audio_denoise(y, method=[m])
            print("测试方法：", m, "结束，用时：", time.time() - start_time)

        print("测试各个降噪方法音频质量及恢复状态--------------------------------------------")
        """
        测试各个降噪方法音频质量及恢复状态
        先降噪，保存降噪音频为denoise_audio_(method).wav，
        再转换为mel图，并使用Mel谱图恢复，保存重建音频为reconstructed_audio.wav
        每种方法mel图包括原始图，降噪后，重建图3种
        """
        method = ['lms', 'noise_reduction', 'highpass', 'fft']
        # 测试所有组合
        import itertools
        n_mel = 32
        log_mel_original, _ = audio_to_logmel_mel(y, sr=sr, n_mels=n_mel, fmax=8000)
        mel_list = [(log_mel_original, "_original")]
        for r in range(1, len(method) + 1) :
            combinations = list(itertools.combinations(method, r))
            for combo in combinations :
                print("测试方法：", combo, "开始--------------------------------------------------")
                y, sr = librosa.load(audio_path)
                audio_denoise_result = audio_denoise(y, method=combo)
                log_mel_spectrogram, mel_spectrogram = audio_to_logmel_mel(y, sr=sr, n_mels=n_mel, fmax=8000)
                mel_list.append((log_mel_spectrogram, str(combo)+"_denoise"))
                print("mel_spectrogram_shape", log_mel_spectrogram.shape)
                audio_reconstructed = mel_to_audio(mel_spectrogram, sr=sr)
                sf.write(f'../test_data/reconstructed_audio{str(combo)}.wav', audio_reconstructed, sr)
                log_mel_spectrogram_reconstructed, _ = audio_to_logmel_mel(audio_reconstructed, sr=sr, n_mels=32, fmax=8000)
                mel_list.append((log_mel_spectrogram_reconstructed, str(combo)+"_reconstructed"))

        plot_mel_spectrogram_list(mel_list, sr=sr, n_mels=32, fmax=8000)
        print("测试结束-")

    def test_align_mel():

        mel_align(np.array([[1, 2, 0],]), np.array([[0, 2, 3, 0], ]))
        mel_align(np.array([[1, 2, 0], ]), np.array([[0, 0, 2, 3, 0], ]))
        mel_align(np.array([[1, 2, 0], ]), np.array([[0, 2, 3], ]))
        mel_align(np.array([[0, 1, 2, 0], ]), np.array([[0, 2, 3, 0], ]))

    test_align_mel()




    # # 可视化原始与重建的Mel谱图
    # plt.figure(figsize=(10, 4))
    # plt.subplot(2, 1, 1)
    # librosa.display.specshow(librosa.amplitude_to_db(mel_spectrogram, ref=np.max), sr=sr, hop_length=512, y_axis='mel',
    #                          x_axis='time')
    # plt.title('Original Mel-spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    # plt.subplot(2, 1, 2)
    # librosa.display.specshow(librosa.amplitude_to_db(librosa.feature.melspectrogram(y=audio_reconstructed, sr=sr, n_mels=32, fmax=8000), ref=np.max), sr=sr, hop_length=512,
    #                          y_axis='mel', x_axis='time')
    # plt.title('Reconstructed Spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()
