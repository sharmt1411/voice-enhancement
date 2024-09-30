import copy
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
from torch.utils.data import DataLoader

from audio_utils import plot_mel_spectrogram_list, audio_to_mel, load_audio, mel_align, mel_to_audio, save_audio, plot_mel_hist
from  audio_model_transformer import TransformerMelModel
from  audio_model_LSTM import SimpleLSTMModel
from  audio_model_conv import BreathToSpeechModel
from  dataset import BreathToSpeechDataset


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


def test_audio_model_process(model, model_path, save_path, model_name, transform, is_norm, model_type, n_fft, hop_length, element_size=64, std_out=False):
    """
    测试音频模型处理函数
    """
    # 导入模型

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # 读取音频数据
    test_data_path = "../dataset/audio_breath_9.wav"
    target_data_path = "../dataset/audio_normal_9.wav"
    # test_data_path = "../dataset/aidataset/audio_breath_1.wav"
    # target_data_path = "../dataset/aidataset/audio_normal_1.wav"
    # breath_mean = 0.21608571708202362
    # breath_std = 1.563319444656372
    # normal_mean = 0.46226173639297485
    # normal_std = 3.082866907119751
    # 读取音频数据
    audio_input, sr = load_audio(test_data_path, sr=16000)
    if audio_input.shape[0] > 55000:
        print("超过3s，截取前3s，audio_input.shape:", audio_input.shape)
        audio_input = audio_input[20000:68000]
    print("audio_test.shape:", audio_input.shape, "sr:", sr)  # (16000,)
    # 计算mel频谱
    num_mel = 128
    f_max = 8000

    mel_input = audio_to_mel(audio_input, n_fft, hop_length, sr, num_mel, f_max, )
    print(f"原始mel_spectrogram.shape:{mel_input.shape}, 最大值:{np.max(mel_input)}, 最小值:{np.min(mel_input)}")  # (128, 108)

    # 读取目标音频数据,用于对比mel图分析
    audio_target, _ = load_audio(target_data_path, sr=16000)
    if audio_target.shape[0] > 55000:
        audio_target = audio_target[20000:68000]
    target_mel = audio_to_mel(audio_target, n_fft, hop_length, sr, num_mel, f_max)
    target_mel_log = librosa.amplitude_to_db(target_mel, ref= 130)  # dataset统计最大值呼吸50 ，正常130
    target_mel_log_clip = np.clip(target_mel_log, min_val, 0)
    target_mel_log_min = np.min(target_mel_log)
    target_mel_log_max = np.max(target_mel_log)
    target_mel_log_std = (target_mel_log_clip-min_val)/(max_val-min_val)    # 线性归一化

    #  输入标准化
    if transform and is_norm:
        mel_input_log_std = mel_input/50  # 线性归一化
    elif transform and not is_norm:
        mel_input_log = librosa.amplitude_to_db(mel_input, ref=50)
        mel_input_log_clip = np.clip(mel_input_log, min_val, 0)
        print(f"mel_input_log.shape:{mel_input_log.shape}, 最大值:{np.max(mel_input_log)}, 最小值:{np.min(mel_input_log)}")
        mel_input_log_std = (mel_input_log_clip - min_val) / (max_val - min_val)
    else:
        mel_input_log_std = mel_input  # 不处理
    print(f"mel_input_log_std.shape:{mel_input_log_std.shape}, 最大值:{np.max(mel_input_log_std)}, 最小值:{np.min(mel_input_log_std)}")

    print(f"targetmel.shape:{target_mel.shape}, 最大值:{np.max(target_mel)}, 最小值:{np.min(target_mel)}")
    print("target_mel_log_min:", target_mel_log_min, "target_mel_log_max:", target_mel_log_max,
          "target_mel_log_std.min(), target_mel_log_std.max():", target_mel_log_std.min(), target_mel_log_std.max())

    with torch.no_grad():
        # 计算模型输出
        if model_type == "lstm":
            model_config["seq_len"] = mel_input.shape[1]+1   # LSTM模型无序列长度属性，多长都可以输入,+1是防止循环超出
        audio_input_tensor = torch.from_numpy(mel_input_log_std).permute(1, 0).unsqueeze(0).float()
        print("mel_tensor.shape:", audio_input_tensor.shape)  # torch.Size([1, 108, 128])
        output_log_std = None
        for i in range((audio_input_tensor.shape[1] // model_config["seq_len"]) + 1):
            input_seq = audio_input_tensor[:, i * model_config["seq_len"] : min(audio_input_tensor.shape[1], (i + 1) * model_config["seq_len"]), :]

            if model_type == 'transformer' and input_seq.shape[1] < model_config["seq_len"]:
                input_seq = torch.cat((input_seq, torch.zeros((1, model_config["seq_len"] - input_seq.shape[1], input_seq.shape[2]))), dim=1)
                print("末尾填充input_seq.shape:", input_seq.shape)
            input_seq = input_seq.to(device)
            print("input_seq.shape:", input_seq.shape)
            predict = model(input_seq).permute(0, 2, 1)
            # print("转换后的predict.shape:", predict.shape)
            predict = predict.squeeze(0)
            if output_log_std is None:
                output_log_std = predict
            else:
                output_log_std = torch.cat((output_log_std, predict), dim=1)
        print("output.shape:", output_log_std.shape)

        output_log_std = output_log_std.cpu().numpy()

        # 反标准化
        if transform and is_norm:
            output = output_log_std * 130  # 线性恢复(0,1)对应(0,1)
        elif transform and not is_norm:
            output_log = output_log_std * (max_val-min_val) + min_val    # outputmax0.8恢复到1.2倍，线性恢复(0,1)对应(-100,0)
            output = librosa.db_to_amplitude(output_log, ref= 130)
            print(f"test_audio_model_output.shape:{output.shape}, 最大值:{np.max(output)}, 最小值:{np.min(output)}")
            output_original = copy.deepcopy(output)
            output[output < threshold] = 0  # 降噪
        else:
            output = output_log_std

        # 计算音频
        audio_output = mel_to_audio(output,  sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32)
        output_original = mel_to_audio(output_original,  sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32)
        # 保存音频
        save_path1 = os.path.join(save_path, 'audio_output_test_clip.wav')
        save_audio(audio_output, sr, save_path1)

        save_path2 = os.path.join(save_path, 'audio_output_original.wav')
        save_audio(output_original, sr, save_path2)


        # 统计模型输出的均值和标准差
        print("输入mel_input.mean:", mel_input.mean(), "mel_spectrogram.std:", mel_input.std(), "max:", mel_input.max(), "min:", mel_input.min())
        print("输入标准化的mel_input_log_std.mean:", mel_input_log_std.mean(), "mel_spectrogram_std.std:", mel_input_log_std.std(), "max:", mel_input_log_std.max(), "min:", mel_input_log_std.min())
        print("输出标准化的output-log-std.mean:", output_log_std.mean(), "output-log-std.std:", output_log_std.std(), "max:", output_log_std.max(), "min:", output_log_std.min())
        if transform and not is_norm:
            print("[-100-0]output-log.mean:", output_log.mean(), "output-log.std:", output_log.std(), "max:", output_log.max(), "min:", output_log.min())
            print("目标的标准化target_mel-db.mean:", target_mel.mean(), "target_mel-db.std:", target_mel.std(), "max:", target_mel.max(), "min:", target_mel.min())
        print("对标的output.mean:", output.mean(), "output.std:", output.std(), "max:", output.max(), "min:", output.min())
        print("对标的target_mel.mean:", target_mel.mean(), "target_mel.std:", target_mel.std(), "max:", target_mel.max(), "min:", target_mel.min())

        # 分析mel频谱的数据分布并用matplotlib直方图绘制
        plot_mel_hist([(output.flatten(), "output"), (target_mel.flatten(), "target"), (mel_input.flatten(), "mel_input"), (mel_input_log_std.flatten(), "input_log_std"), ])

        # 绘制mel频谱
        plot_mel_spectrogram_list([(mel_input, "input"), (output, "output"), (target_mel, "target")], is_log= False)


def test_overfit(model, model_path, save_path, model_name, transform, is_norm, model_type, dataset_path, n_fft, hop_length, element_size=64, train_ratio=0.9, batch_size= 8, ):
    """
    使用训练数据测试模型过拟合
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # 读取音频数据
    dataset = BreathToSpeechDataset(dataset_path=dataset_path, element_size=element_size,
                                    transform=transform, is_norm=is_norm, n_fft=n_fft, hop_length=hop_length)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    print(f'Total dataset size: {total_size}, train size: {train_size}, val size: {val_size}')

    audio_input = []
    audio_target = []
    for i in range(10):
        audio_input.append(dataset[i][0])  #(64,128)
        audio_target.append(dataset[i][1])
    audio_input_mel_log_std = np.array(audio_input).reshape(-1,128)
    audio_target_mel_log_std = np.array(audio_target).reshape(-1,128).transpose(1,0)

    # 恢复原始输出mel用于生成音频对比
    if transform and not is_norm:
        audio_target_mel_log = audio_target_mel_log_std * (max_val-min_val) + min_val
        audio_target_mel = librosa.db_to_amplitude(audio_target_mel_log, ref=dataset.stats['max-normal'].item())
        audio_target_mel[audio_target_mel < threshold] = 0  # 降噪
    elif transform and is_norm:
        audio_target_mel = audio_target_mel_log_std*dataset.stats['max-normal'].item()
    else:
        audio_target_mel = audio_target_mel_log_std

    with torch.no_grad() :
        # 计算模型输出
        if model_type == "lstm" :
            model_config["seq_len"] = audio_input_mel_log_std.shape[0] + 1  # LSTM模型无序列长度属性，多长都可以输入,+1是防止循环超出
        audio_input_tensor = torch.from_numpy(audio_input_mel_log_std).unsqueeze(0).float()
        print("mel_tensor.shape:", audio_input_tensor.shape)  # torch.Size([1, 640, 128])
        output_log_std = None
        for i in range((audio_input_tensor.shape[1] // model_config["seq_len"])) :
            input_seq = audio_input_tensor[:, i * model_config["seq_len"] : min(audio_input_tensor.shape[1],
                                                                                (i + 1) * model_config["seq_len"]), :]

            if model_type == 'transformer' and input_seq.shape[1] < model_config["seq_len"] :
                input_seq = torch.cat(
                    (input_seq, torch.zeros((1, model_config["seq_len"] - input_seq.shape[1], input_seq.shape[2]))),
                    dim=1)
                print("末尾填充input_seq.shape:", input_seq.shape)
            input_seq = input_seq.to(device)
            print("input_seq.shape:", input_seq.shape)
            predict = model(input_seq).permute(0, 2, 1)   # 模型输入输出都是btc
            # print("转换后的predict.shape:", predict.shape)
            predict = predict.squeeze(0)
            if output_log_std is None :
                output_log_std = predict
            else :
                output_log_std = torch.cat((output_log_std, predict), dim=1)
        print("output.shape:", output_log_std.shape)

        output_log_std = output_log_std.cpu().numpy()

        if transform and not is_norm:
            output_log = output_log_std * (max_val-min_val)  # outputmax0.8恢复到1.2倍，线性恢复(0,1)对应(-80,0)
            output_log = output_log + min_val  # 对齐最大值为0
            output_mel = librosa.db_to_amplitude(output_log, ref=dataset.stats['max-normal'].item())
            output_mel[output_mel < threshold] = 0  # 降噪
        elif transform and is_norm:
            output_mel = output_log_std * 130
        else:
            output_mel = output_log_std



        # 计算音频
        audio_output = mel_to_audio(output_mel, sr=16000, n_fft=n_fft, hop_length=hop_length, n_iter=32)
        target_audio = mel_to_audio(audio_target_mel, sr=16000, n_fft=n_fft, hop_length=hop_length, n_iter=32)
        # 保存音频
        save_path1 = os.path.join(save_path, 'audio_output_test_overfit.wav')
        save_audio(audio_output, 16000, save_path1)
        save_path = os.path.join(save_path, 'audio_target_test_overfit.wav')
        save_audio(target_audio, 16000, save_path)

    # 统计模型输出的均值和标准差
    print(
        f"audio_input_mel_log_std.shape:{audio_input_mel_log_std.shape}, 最大值:{np.max(audio_input_mel_log_std)}, 最小值:{np.min(audio_input_mel_log_std)}")
    print(
        f"audio_target_mel_log_std.shape:{audio_target_mel_log_std.shape}, 最大值:{np.max(audio_target_mel_log_std)}, 最小值:{np.min(audio_target_mel_log_std)}")

    print("模型输出output-log-std.mean:", output_log_std.mean(), "output-log-std.std:", output_log_std.std(),
          "max:", output_log_std.max(), "min:", output_log_std.min())
    if transform and not is_norm:
        print("scaler+归一化恢复后output-log.mean:", output_log.mean(), "output-log.std:", output_log.std(), "max:",
              output_log.max(), "min:", output_log.min())
    print("最终线性output-mel.mean:", output_mel.mean(), "output-db.std:", output_mel.std(), "max:",
          output_mel.max(), "min:", output_mel.min())
    print("线性target_mel.mean:", audio_target_mel.mean(), "target_mel-db.std:", audio_target_mel.std(), "max:",
          audio_target_mel.max(), "min:", audio_target_mel.min())

    # 分析mel频谱的数据分布并用matplotlib直方图绘制
    plot_mel_hist([(output_log_std.flatten()[0:128*160], "output_std"), (audio_target_mel_log_std.flatten()[0:128*160], "target_std"),
                   (audio_input_mel_log_std.flatten()[0:128*160], "input_log_std"), ])

    # 绘制mel频谱
    if transform and not is_norm:
        plot_mel_spectrogram_list(
            [(audio_input_mel_log_std.transpose()[:, 0:160], "input-log_std"), (output_log[:, 0:160], "model-output-std+scaler"), (audio_target_mel_log[:, 0:160], "target-log"),
             (audio_target_mel_log_std[:, 0:160], "target-log-std")], is_log=True)
    elif transform and is_norm:
        plot_mel_spectrogram_list(
            [(audio_input_mel_log_std.transpose()[:, 0 :160], "input_std"),
             (output_mel[:, 0 :160], "model-output-amp"),
             (audio_target_mel_log_std[:, 0 :160], "target-norm")], is_log=False)


def test_normalize_effect_on_audio(dataset_path, save_path, element_size=64, transform=True, is_norm=False,
                                    n_fft=1024, hop_length=256, n_mels=128, f_max=8000, target_sr=16000, is_plot=True):
    """
    测试归一化对音频的影响,有可能目标音频已经是包含很多噪声了
    """

    # 读取音频数据
    dataset = BreathToSpeechDataset(dataset_path=dataset_path, element_size=element_size, transform=transform, is_norm=is_norm,
                                    n_fft=n_fft, hop_length=hop_length, num_mel=n_mels, f_max=f_max, target_sr=target_sr)
    total_size = len(dataset)
    print(f'Total dataset size: {total_size}')

    audio_target = []
    for i in range(10) :
        audio_target.append(dataset[i][1])
    audio_target_mel_log_std = np.array(audio_target).reshape(-1, n_mels).transpose(1, 0)

    # 判断是否是归一化输入，否则是log——mel归一化输入
    if transform and is_norm:
        audio_target_mel = audio_target_mel_log_std * dataset.stats['max-normal'].item()   # 根据dataset载入的最大值归一化
    elif transform and not is_norm:
        audio_target_mel_log = audio_target_mel_log_std * (max_val-min_val) + min_val       # 线性恢复(0,1)对应(-100,0)
        ref = dataset.stats['max-normal'].item()
        print("dataset.ref:", ref)
        audio_target_mel = librosa.db_to_amplitude(audio_target_mel_log, ref=ref)
        audio_target_mel[audio_target_mel < threshold] = 0  # 降噪
    else:
        # 不做transform
        audio_target_mel = audio_target_mel_log_std

        # 计算音频
    target_audio = mel_to_audio(audio_target_mel, sr=16000, n_fft=n_fft, hop_length=hop_length, n_iter=32)
    # 保存音频
    save_path = (
        os.path.join(save_path,
                     f'audio_target_test_normalize_{n_fft}_{hop_length}_{n_mels}_trans{transform}.wav'))
    save_audio(target_audio, 16000, save_path)
    # print("audio saved to:", save_path)

    # 统计模型输出的均值和标准差
    print(
        f"test-normalize-audio_target_mel_log_std.shape:{audio_target_mel_log_std.shape}, 最大值:{np.max(audio_target_mel_log_std)}, 最小值:{np.min(audio_target_mel_log_std)}")
    if transform:
        print("最终恢复的线性audio-target_mel.mean:", audio_target_mel.mean(), "target_mel-db.std:", audio_target_mel.std(), "max:",
               audio_target_mel.max(), "min:", audio_target_mel.min())

    # 分析mel频谱的数据分布并用matplotlib直方图绘制
    if is_plot:
        plot_mel_hist([(audio_target_mel_log_std.flatten()[0 :128 * 160], "target_std"),])

        # 绘制mel频谱
        plot_mel_spectrogram_list(
            [(audio_target_mel_log[:, 0 :160], "target-log"),
             (audio_target_mel_log_std[:, 0 :160], "target-log-std")], is_log=True)


if __name__ == '__main__':
    # 初始化PyAudio
    # audio = pyaudio.PyAudio()
    # start_audio_stream(audio)

    model_type = 'conv'

    if model_type == 'transformer' :
        model_name = 'model_transformer_1754368__mel_128_d_128_seq_len_24_n_h_2_layers_8_drop_0.4.pth'
        element_size = 24
    elif model_type == 'lstm' :
        model_name = 'model_lstm_4677760__mel_128_hsize_128_layers_16_drop_0.4.pth'
        element_size = 24
        lstm_hidden_size = int(model_name.split('hsize_')[1].split('_')[0])
        print("lstm_hidden_size:", lstm_hidden_size)
        lstm_layers = int(model_name.split('layers_')[1].split('_')[0])
        print("lstm_layers:", lstm_layers)
    elif model_type == 'conv':
        model_name = 'model_conv_886529__mel_128_seq_len_64_hidden_s_128_layers_2_dropout_0.2.pth'
        # model_name = 'model_conv_886529__mel_128_seq_len_64_hidden_s_128_layers_2_dropout_0.3.pth'  # 正常 is_norm=False模型，用于测试
        # 读取参数
        element_size = int(model_name.split('seq_len_')[1].split('_')[0])  # 需要对应训练模型的参数seq_len
        lstm_hidden_size = int(model_name.split('hidden_s_')[1].split('_')[0])
        lstm_layers = int(model_name.split('layers_')[1].split('_')[0])
        dropout_rate = float(model_name.split('dropout_')[1].split('.pth')[0])
        print("lstm_layers:", lstm_layers)
        print("lstm_hidden_size:", lstm_hidden_size)
        print("element_size:", element_size)

    if model_type == 'transformer' :
        model = TransformerMelModel(seq_length=element_size, d_model=128, n_head=2, num_decoder_layers=8, dropout=0.4)
    elif model_type == 'lstm' :
        model = SimpleLSTMModel(mel_bins=128, lstm_hidden_size= lstm_hidden_size, lstm_layers=lstm_layers, dropout_rate=0.4)
    elif model_type == 'conv' :
        model = BreathToSpeechModel(seq_len=element_size, lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, dropout_rate=dropout_rate)
    else :
        raise ValueError("Unsupported model type")

    test_list = ['test_testset', 'test_overfit', 'test_normalize']
    to_test = [
               # "test_testset",
               "test_overfit",
               # "test_normalize"
    ]

    min_val = -70
    max_val = 0
    ref_normal =130
    threshold = ref_normal * 10 ** (0.5 * min_val / 10) * 1.025
    print("threshold:", threshold)
    if "test_testset" in to_test:
        print("测试测试集--------------------------------------------------")
        test_audio_model_process(model, model_path='../model_save', save_path='../test_data/test_testset',
                                 model_name=model_name, transform=True, is_norm=False,
                                 model_type=model_type, n_fft=512,hop_length=128,
                                 element_size=element_size, std_out=True)

    if "test_overfit" in to_test:
        print("测试过拟合----------------------------------------------------")
        test_overfit(model, model_path='../model_save', save_path='../test_data/test_overfit',
                     model_name=model_name, transform=True, is_norm=False,
                     model_type=model_type, dataset_path='../dataset/aidataset/', n_fft=512, hop_length=128,
                     element_size=element_size, train_ratio=0.9, batch_size=8)

    if "test_normalize" in to_test:
        """测试不同的数据集加载方式对音频的影响"""
        print("测试归一化对音频的影响----------------------------------------------")
        test_normalize_effect_on_audio(dataset_path='../dataset/aidataset/', save_path='../test_data/test_normalize',
                                       element_size=element_size
                                       , transform=True, is_norm=False, n_fft=512, hop_length=128, n_mels=128, f_max=8000,
                                       target_sr=16000, is_plot=False)





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
