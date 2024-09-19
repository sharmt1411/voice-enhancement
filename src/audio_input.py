import os
import time

import librosa
import pyaudio
import wave
import threading
import queue
import numpy as np
import scipy.signal as signal
import torch

from audio_utils import plot_mel_spectrogram_list, audio_to_mel, load_audio, mel_align, mel_to_audio, save_audio, plot_mel_hist
from  audio_model_transformer import TransformerMelModel
from  audio_model_LSTM import SimpleLSTMModel
from  audio_model_conv import BreathToSpeechModel


# 定义音频流参数
CHUNK = 1024  # 缓冲区的块大小，单位为帧
FORMAT = pyaudio.paInt16  # 采样深度为16位
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率为16kHz

# 初始化队列作为麦克风输入缓冲区
buffer_queue = queue.Queue()
# 处理好的音频数据将会被放入此队列
processed_queue = queue.Queue()


def audio_callback(in_data, frame_count, time_info, status):
    """
    pyaudio.PyAudio()音频流stream回调函数
    存储音频数据到缓冲区进行处理
    """
    # 将音频数据放入缓冲区
    buffer_queue.put(in_data)
    return in_data, pyaudio.paContinue


def audio_consumer():
    """
    消费者线程，用于处理缓冲区音频数据，
    处理逻辑在此函数中实现，
    并将处理后的数据放入处理队列
    需要配合consumer_thread线程一起使用
    """
    i = 0
    while True:
        # 从缓冲区获取音频数据进行处理
        data = buffer_queue.get()
        if data is None:
            break
        # 这里可以添加音频处理逻辑
        # processed_data = audio_process(data)
        # processed_queue.put(processed_data)
        print("Processing audio data..."+str(i))
        i += 1


def start_audio_stream(audio=None):
    """
    启动麦克风音频流输入，并启动消费者线程
    """
    if audio is None:
        # 初始化PyAudio
        audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=audio_callback)

    # 创建并启动消费者线程，用于处理缓冲区中的音频数据
    consumer_thread = threading.Thread(target=audio_consumer)
    consumer_thread.start()

    try:
        # 开始音频流
        print("Recording...")
        stream.start_stream()

        # 运行10s后停止

        # time.sleep(10)
        # stream.stop_stream()

        # 持续运行直到用户停止
        while stream.is_active():
            time.sleep(0.5)

    except KeyboardInterrupt:
        # 用户停止录音
        stream.stop_stream()

    finally:
        # 关闭音频流和PyAudio
        print("Stopping...")
        buffer_queue.put(None)  # 发送停止信号到消费者线程
        consumer_thread.join()
        stream.close()
        audio.terminate()


def test_audio_model_process(save_path,model_name,model_type,element_size=64, std_out=False):
    """
    测试音频模型处理函数
    """
    # 导入模型
    if model_type == 'transformer' :
        model = TransformerMelModel(seq_length=element_size, d_model=64, n_head=2)
    elif model_type == 'lstm' :
        model = SimpleLSTMModel()
    elif model_type == 'conv' :
        model = BreathToSpeechModel(seq_len=element_size)
    else :
        raise ValueError("Unsupported model type")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model_config = model.config
    print(f'Model config: {model_config}')

    # 打印模型参数数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {params}')

    if device == 'cuda' :
        model.load_state_dict(torch.load(os.path.join(save_path, model_name), map_location=device))
        print(f'load model successfully-{device}')
    else :
        model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        print(f'load model successfully-cpu')

    model.eval()

    # 读取音频数据
    test_data_path = "../dataset/audio_breath_9.wav"
    breath_mean = 0.14538872241973877
    breath_std = 0.9532793164253235
    normal_mean = 0.979012966156006
    normal_std = 18.868282318115234
    # 读取音频数据
    mel_tensor, sr = load_audio(test_data_path, sr=16000)
    print("audio_test.shape:", mel_tensor.shape, "sr:", sr)  # (16000,)
    # 计算mel频谱
    num_mel = 128
    f_max = 8000

    mel_spectrogram = audio_to_mel(mel_tensor, sr, num_mel, f_max)
    print("原始mel_spectrogram.shape:", mel_spectrogram.shape)  # (128, 108)

    # 标准化mel频谱的均值和标准差
    mel_spectrogram_std = (mel_spectrogram - breath_mean) / breath_std

    with torch.no_grad():
        # 计算模型输出
        if model_type == "lstm":
            model_config["seq_len"] = mel_spectrogram.shape[1]+1  # LSTM模型无序列长度属性，多长都可以输入
        mel_tensor = torch.from_numpy(mel_spectrogram_std).permute(1, 0).unsqueeze(0).float()
        print("mel_tensor.shape:", mel_tensor.shape)  # torch.Size([1, 108, 128])
        output = None
        for i in range((mel_tensor.shape[1] // model_config["seq_len"]) + 1):
            input_seq = mel_tensor[:, i * model_config["seq_len"] : min(mel_tensor.shape[1], (i + 1) * model_config["seq_len"]), :]

            if model_type == 'transformer' and input_seq.shape[1] < model_config["seq_len"]:
                input_seq = torch.cat((input_seq, torch.zeros((1, model_config["seq_len"] - input_seq.shape[1], input_seq.shape[2]))), dim=1)
                print("末尾填充input_seq.shape:", input_seq.shape)
            input_seq = input_seq.to(device)
            print("input_seq.shape:", input_seq.shape)
            predict = model(input_seq).permute(0, 2, 1)
            print("转换后的predict.shape:", predict.shape)
            predict = predict.squeeze(0)
            if output is None:
                output = predict
            else:
                output = torch.cat((output, predict), dim=1)
        print("output.shape:", output.shape)

        # 反标准化
        if std_out:
            output = output * normal_std + normal_mean
        output_std = output.cpu().numpy()
        print("output_std.shape:", output.shape)
        # 计算音频
        audio_output = mel_to_audio(output_std,  sr=sr, n_fft=2048, hop_length=512, n_iter=32)
        # 保存音频
        save_path = os.path.join(save_path, 'audio_output_test.wav')
        save_audio(audio_output, sr, save_path)

        txt_file = "mel_spectrogram.txt"
        with open(txt_file, 'w') as f:
            f.write(str(list(mel_spectrogram[:, 21:22].flatten())))
            f.write('\n')
            f.write(str(list(mel_spectrogram_std[:, 21:22].flatten())))
            f.write('\n')
            f.write(str(list(output_std[:, 21:22].flatten())))
        # 统计模型输出的均值和标准差
        print("mel_spectrogram.mean:", mel_spectrogram.mean(), "mel_spectrogram.std:", mel_spectrogram.std(),"max:", mel_spectrogram.max(), "min:", mel_spectrogram.min())
        print("mel_spectrogram_std.mean:", mel_spectrogram_std.mean(), "mel_spectrogram_std.std:", mel_spectrogram_std.std(),"max:", mel_spectrogram_std.max(), "min:", mel_spectrogram_std.min())
        print("output_std.mean:", output_std.mean(), "output_std.std:", output_std.std(),"max:", output_std.max(), "min:", output_std.min())

        # 分析mel频谱的数据分布并用matplotlib直方图绘制
        plot_mel_hist([(output_std, "output_std"), (mel_spectrogram_std, "input_std"), (mel_spectrogram, "mel_spectrogram")])

        # 绘制mel频谱
        plot_mel_spectrogram_list([(mel_spectrogram, "input"), (output_std,"output")] ,is_log= False)







if __name__ == '__main__':
    # 初始化PyAudio
    # audio = pyaudio.PyAudio()
    # start_audio_stream(audio)
    test_audio_model_process(save_path='../model_save',
                             model_name='model_lstm_692352__mel_128_hsize_128_layers_2_drop_0.1.pth',
                             model_type='lstm', std_out=True, element_size=24)

    # test_data_path = "../dataset/audio_breath_9.wav"
    # save_path = '../model_save'
    # # 读取音频数据
    # audio_test, sr = load_audio(test_data_path, sr=16000)
    # print("audio_test.shape:", audio_test.shape, "sr:", sr)  # (16000,)
    # # 计算mel频谱
    # num_mel = 128
    # f_max = 8000
    #
    # mel_spectrogram = audio_to_mel(audio_test, sr, num_mel, f_max)
    # print("原始mel_spectrogram.shape:", mel_spectrogram.shape)
    #
    # audio_output = mel_to_audio(mel_spectrogram, sr=sr, n_fft=2048, hop_length=512, n_iter=32)
    # # mel_inverted_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=2048)
    # # # 使用Griffin-Lim算法估计相位并重建音频
    # # audio_output = librosa.griffinlim(mel_inverted_spectrogram, n_iter=32, hop_length=512)
    # print("audio_output.shape:", audio_output.shape)
    # # 保存音频
    # save_path = os.path.join(save_path, 'audio_output_test.wav')
    # save_audio(audio_output, sr, save_path)
    # # 绘制mel频谱
    # plot_mel_spectrogram_list([(mel_spectrogram, "input"), (mel_spectrogram, "output")], is_log=False)
