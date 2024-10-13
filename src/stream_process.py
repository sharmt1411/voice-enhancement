"""
实时流式处理测试
输入麦克风音频，直接生成模型输出的语音
"""
import os
import time

import librosa
import pyaudio
import threading
import queue
import numpy as np
import torch
from numpy import floating

from audio_utils import audio_to_mel, mel_to_audio
from audio_model_conv import BreathToSpeechModel


def audio_callback(in_data, frame_count, time_info, status):
    # 将音频数据放入缓冲区
    buffer_queue.put(in_data)
    return in_data, pyaudio.paContinue


def process_data(data, silent_threshold, index):
    """处理音频数据，使用模型推理"""
    sr = RATE  # 采样率
    # 增益倍数
    gain = 50    # 30目前最好，正常声音不影响,测试关闭音频增益，录制的声音差大概50倍
    audio_input = np.frombuffer(data, dtype=np.float32)*gain  # float，并归一化到[-1, 1],并增强10倍

    # 静音检测
    max_value = np.max(audio_input)
    # if max_value < silent_threshold*gain :
    #     print("阈值静音音频Silent audio detected, skip. max_value:", max_value)
    #     return audio_input/gain, False


    # print("audio_input.shape:", audio_input.shape, f", audio输入{index}最大值:", np.max(np.abs(audio_input)), ", 绝对值均值:", np.mean(np.abs(audio_input)))  # 一个chunk的大小
    mel_input = audio_to_mel(audio_input, n_fft, hop_length, sr, num_mel, f_max, )
    # 低频最大值

    mel_input_log = librosa.amplitude_to_db(mel_input, ref=130)  # 训练为ref130
    mel_input_log_clip = np.clip(mel_input_log, min_val, 0)

    mel_low_freq_max = np.max(mel_input_log_clip[5:18]) # 对应80-300Hz 在fmax=8000，nmel=128
    print("mel低频最大值：", np.max(mel_input_log_clip[5 :18]), ", 低频绝对值均值：", np.mean(np.abs(mel_input_log_clip[5 :18])))
    if mel_low_freq_max > -10:
        print("正常音频Skip")
        return audio_input, False
    # if mel_low_freq_max < -60:
    #     print("mel静音音频Silent audio detected, skip. max_value:", max_value)
    #     return audio_input / gain, False

    print("呼吸音频》》》》》》》》")

    mel_input_log_std = (mel_input_log_clip - min_val) / (max_val - min_val)

    with torch.no_grad():
        audio_input_tensor = torch.from_numpy(mel_input_log_std).permute(1, 0).unsqueeze(0).to(device)
        predict = model(audio_input_tensor).permute(0, 2, 1).squeeze(0).cpu().numpy()
        output_log = predict * (max_val - min_val) + min_val  # 线性恢复(0,1)对应(-100,0)
        output = librosa.db_to_amplitude(output_log, ref=130) # 线性归一化恢复
        output = output.astype(np.float32)  # 转为float32
        output[output < mel_threshold] = 0  # 降噪

        # 计算音频

        return output, True


def audio_consumer():
    """
    消费者线程，用于处理缓冲区音频数据，
    处理逻辑在此函数中实现，
    并将处理后的数据放入处理队列
    需要配合consumer_thread线程一起使用
    """
    i = 1
    silent_threshold = 0.003  # 静默阈值

    while True:
        # 从缓冲区获取音频数据进行处理
        data = buffer_queue.get()
        if data is not None:
            # print("Processing audio data..." + str(i))
            if PROCESS_BOOL:
                audio_processed, is_breath = process_data(data, silent_threshold, i)
                if audio_processed is not None:
                    processed_queue.put((audio_processed, is_breath))

            else:
                audio_processed = np.frombuffer(data, dtype=np.float32)*10
                # 嗓音*10后，0.2-0.5 呼吸声同 静音 0.003
                # print("audio_processed.shape:", audio_processed.shape, f", 最大值:{np.max(np.abs(audio_processed))}, 绝对值均值:{np.mean(np.abs(audio_processed))}")  # 一个chunk的大小
                if audio_processed is not None:
                    processed_queue.put(audio_processed)

        else:
            # 停止信号
            print("Consumer thread received Stop signal.")
            processed_queue.put(None)
            break
        i += 1


def audio_player(output_stream):
    """
    播放线程，用于播放处理后的数据
    """
    i = 1
    while True:
        # 从处理队列获取处理后的数据进行播放
        data = processed_queue.get()

        if data is None:
            print("Player thread received Stop signal.")
            break

        if PROCESS_BOOL:
            data, is_breath = data
            if is_breath:
                data = mel_to_audio(data, sr=RATE, n_fft=n_fft, hop_length=hop_length, n_iter=32)
            # print(f"audio_output.shape:{data.shape}, 输出最大值:{np.max(data)}, 最小值:{np.min(data)}")

        if i % 1 == 0:
            output_stream.write(data.tobytes())
            # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Playing audio data---------------{i}")
        i += 1


def start_audio_stream(audio=None):
    """
    启动麦克风音频流输入，并启动消费者线程
    """
    consumer_thread = None
    player_thread = None
    input_stream = None
    output_stream = None
    if audio is None:
        # 初始化PyAudio
        audio = pyaudio.PyAudio()
    try:
        # 打开音频流

        input_stream = audio.open(format=FORMAT_INPUT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK,
                                  stream_callback=audio_callback)

        # 打开输出流 (播放)
        output_stream = audio.open(format=FORMAT_OUTPUT,
                                   channels=CHANNELS,
                                   rate=RATE,
                                   output=True,
                                   frames_per_buffer=CHUNK)

        # 创建并启动消费者线程，用于处理缓冲区中的音频数据
        consumer_thread = threading.Thread(target=audio_consumer)
        consumer_thread.start()

        # 创建并启动播放线程，用于播放处理后的数据
        player_thread = threading.Thread(target=audio_player, args=(output_stream,))
        player_thread.start()

        # 开始音频流
        print("Starting Recording...")
        input_stream.start_stream()

        # 持续运行直到用户停止
        while input_stream.is_active():
            time.sleep(0.5)

    except KeyboardInterrupt:
        # 用户停止录音
        print("KeyboardInterrupt, Stopping...")
        # input_stream.stop_stream()
        # output_stream.stop_stream()
        # buffer_queue.put(None)  # 发送停止信号到消费者线程

    finally:
        # 关闭音频流和PyAudio
        print("Stopping...")
        buffer_queue.put(None)  # 发送停止信号到消费者线程
        if consumer_thread is not None:
            consumer_thread.join()
        if player_thread is not None:
            player_thread.join()
        print("Closing PyAudio...")
        # 确保流被正确停止和关闭
        try :
            if input_stream.is_active() :
                input_stream.stop_stream()  # 停止输入流
            if output_stream.is_active() :
                output_stream.stop_stream()  # 停止输出流
        except Exception as e :
            print(f"Error stopping streams: {e}")
        print("Closing streams...")
        # 关闭流
        try :
            input_stream.close()  # 关闭输入流
            output_stream.close()  # 关闭输出流
        except Exception as e :
            print(f"Error closing streams: {e}")
        print("Terminated...")
        audio.terminate()
        print("Terminated PyAudio...")


def load_model():
    model_path = '../model_save'
    global element_size

    model_name = 'model_1011noise-conv5_conv_1311873__mel_128_seq_len_64_hidden_s_128_layers_2_dropout_0.2.pth'
    # 读取参数
    element_size = int(model_name.split('seq_len_')[1].split('_')[0])  # 需要对应训练模型的参数seq_len
    lstm_hidden_size = int(model_name.split('hidden_s_')[1].split('_')[0])
    lstm_layers = int(model_name.split('layers_')[1].split('_')[0])
    dropout_rate = float(model_name.split('dropout_')[1].split('.pth')[0])
    print("lstm_layers:", lstm_layers)
    print("lstm_hidden_size:", lstm_hidden_size)
    print("element_size:", element_size)
    model = BreathToSpeechModel(seq_len=element_size, lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, dropout_rate=dropout_rate)

    model = model.to(device)
    model_config = model.config
    print(f'Model config: {model_config}')

    # 打印模型参数数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {params}')

    if device == 'cuda' :
        model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location=device))
        print(f'load model successfully-{device}')
    else :
        model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
        print(f'load model successfully-cpu')
    model.eval()

    return model


# 定义音频流参数
CHUNK = 8064  # 128*63 8064 0.5s
FORMAT_INPUT = pyaudio.paFloat32
FORMAT_OUTPUT = pyaudio.paFloat32  # 输出格式为16位整型
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率为16kHz
PROCESS_BOOL = True  # 是否进行模型推理

min_val = -70
# min_val = -60
max_val = 0
ref_normal = 130
mel_threshold = ref_normal * 10 ** (0.5 * (min_val+15) / 10) * 1.025   # +15对正常影响较小，gain30，ref130
num_mel = 128
f_max = 8000
n_fft = 512
hop_length = 128
element_size = 0  # 由模型决定

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化队列作为麦克风输入缓冲区
buffer_queue = queue.Queue()
# 处理好的音频数据将会被放入此队列
processed_queue = queue.Queue()

model = load_model()
model_config = model.config

start_audio_stream()




