import torch
print(torch.__version__)

# pytorch 版本
print(torch.version.cuda)

print(torch.cuda.is_available())


import os
import re

from audio_utils import *

audio,_ = load_audio("../dataset/aidataset/audio_breath_1.wav")
mel = audio_to_mel(audio,512,128)
print(mel.shape)
print(mel.dtype)
print(mel[0][0].dtype)

noise_stream = generate_white_noise_stream(100)
print(noise_stream.shape)
print(noise_stream.dtype)
def rename_files_in_directory(directory) :
    # 获取文件夹中的所有文件
    files = os.listdir(directory)

    # 定义正则表达式匹配
    wav_pattern = re.compile(r'^(\d+)\.WAV$')  # 匹配 25.WAV 这种格式
    breath_pattern = re.compile(r'^录音 \((\d+)\)\.wav$')  # 匹配 录音 (25).wav 这种格式

    for file_name in files :
        # 处理 audio_normal 文件
        wav_match = wav_pattern.match(file_name)
        if wav_match :
            index = int(wav_match.group(1))  # 提取文件中的数字部分
            new_name = f"audio_normal_{index - 20}.wav"  # 按照要求重命名
            os.rename(os.path.join(directory, file_name), os.path.join(directory, new_name))
            print(f"Renamed '{file_name}' to '{new_name}'")

        # 处理 audio_breath 文件
        breath_match = breath_pattern.match(file_name)
        if breath_match :
            index = int(breath_match.group(1))  # 提取文件中的数字部分
            new_name = f"audio_breath_{index - 20}.wav"  # 按照要求重命名
            os.rename(os.path.join(directory, file_name), os.path.join(directory, new_name))
            print(f"Renamed '{file_name}' to '{new_name}'")


# 使用示例：将 'path' 替换为你的文件夹路径
# directory_path = r"C:\Users\14116\Desktop\dataset"
# rename_files_in_directory(directory_path)