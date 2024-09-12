import time

import pyaudio
import wave
import threading
import queue
import numpy as np
import scipy.signal as signal

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


def start_audio_stream():
    """
    启动麦克风音频流输入，并启动消费者线程
    """
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


if __name__ == '__main__':
    start_audio_stream()