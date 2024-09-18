"""
数据集模块
处理数据集相关的功能
转换为将来模型输入的格式，预设pyaudio采样，采样率16kHz，深度int16，块大小1024~64ms
考虑实时性要求，输入时长0.5s，nfft=512, hop_length=256,num_mel=64
"""

import os

import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

from audio_utils import plot_mel_spectrogram_list, audio_to_mel, load_audio, mel_align


target_sr = 16000
n_fft = 512
hop_length = 256
num_mel = 128
f_max = 8000
# element_size = 128  # 输入块的大小


class BreathToSpeechDataset(Dataset):
    def __init__(self, dataset_path, transform=None, element_size=64):
        self.element_size = element_size
        self.dataset_path = dataset_path
        self.transform = transform
        self.file_pairs = self._load_file_pairs()
        self.breath_mel_data = np.array([])
        self.normal_mel_data = np.array([])
        self._load_audio_data()

    def _load_file_pairs(self):
        # 匹配相同index的呼吸音频和正常音频文件
        print(f"开始加载{self.dataset_path}数据集")
        normal_files = sorted([f for f in os.listdir(self.dataset_path) if 'audio_normal' in f])
        breath_files = sorted([f for f in os.listdir(self.dataset_path) if 'audio_breath' in f])
        print(f"共有{len(normal_files)}个正常音频文件，{len(breath_files)}个呼吸音频文件")
        file_pairs = []
        for normal_file, breath_file in zip(normal_files, breath_files):
            normal_index = normal_file.split('_')[-1]
            breath_index = breath_file.split('_')[-1]
            if normal_index == breath_index:
                file_pairs.append((normal_file, breath_file))
        print(f"共有{len(file_pairs)}个匹配的音频文件")
        return file_pairs

    def _load_audio(self, filename):
        """
        加载音频文件，转换为mel频谱图，并合并所有mel图
        注意统一处理数据类型，使用audio_utils中的方法
        如果使用torchaudio.load，则需转换采样率并且处理数据类型，注意数据形状的差异
        """
        audio_path = os.path.join(self.dataset_path, filename)
        audio, sr = load_audio(audio_path, sr=target_sr)
        return audio, sr

    def _load_audio_data(self):
        """加载所有数据集数据，并合并拆分"""
        length = len(self.file_pairs)

        for i in range(len(self.file_pairs)) :
            # print(f"正在加载处理第{i+1}个音频---------------")
            normal_file, breath_file = self.file_pairs[i]
            breath_waveform, breath_sr = self._load_audio(breath_file)
            normal_waveform, normal_sr = self._load_audio(normal_file)
            # 确保采样率一致
            assert breath_sr == normal_sr, "采样率不一致"
            # 转换为Mel频谱图
            breath_mel_ = audio_to_mel(breath_waveform, breath_sr, num_mel, f_max)
            normal_mel_ = audio_to_mel(normal_waveform, normal_sr, num_mel, f_max)

            # print("对齐前mei形状", breath_mel_.shape, normal_mel_.shape)
            breath_mel, normal_mel = mel_align(breath_mel_, normal_mel_)

            # print(f"呼吸音频{breath_file}，正常音频{normal_file}，音频形状{breath_waveform.shape}, "
            #       f"{normal_waveform.shape}Mel频谱图形状{breath_mel.shape}, {normal_mel.shape}")

            # 绘制对数Mel频谱图,比较差异
            plot = False
            if plot :
                mel_list = [(breath_mel_, 'breath_before_align'), (normal_mel_, 'normal_before_align'),
                            (breath_mel, 'breath'), (normal_mel, 'normal')]
                plot_mel_spectrogram_list(mel_list, target_sr, num_mel, fmax=f_max, is_log=False)
                print("Mel频谱图绘制完成")

            if self.transform :
                breath_mel = self.transform(breath_mel)
                normal_mel = self.transform(normal_mel)

            self.breath_mel_data = np.hstack((self.breath_mel_data, breath_mel)) if self.breath_mel_data.size else breath_mel
            self.normal_mel_data = np.hstack((self.normal_mel_data, normal_mel)) if self.normal_mel_data.size else normal_mel
            # print(f"第{i+1}个音频处理完成,目前数据形状，呼吸{self.breath_mel_data.shape}, 正常{self.normal_mel_data.shape}")
        assert self.breath_mel_data.shape[1] == self.normal_mel_data.shape[1], "呼吸和正常音频时长不一致"
        self.breath_mel_data = self.breath_mel_data.transpose()  # 转换为(time, dim)
        self.normal_mel_data = self.normal_mel_data.transpose()  # 转换为(time, dim)
        self.breath_mel_data = torch.from_numpy(self.breath_mel_data).float()
        self.normal_mel_data = torch.from_numpy(self.normal_mel_data).float()
        print(f"------------------------------------------------\n数据集加载完成，呼吸{self.breath_mel_data.shape}, 正常{self.normal_mel_data.shape}")

    def __len__(self):
        return self.breath_mel_data.shape[0]//self.element_size

    def __getitem__(self, idx):
        # print(f"加载第{idx}个音频片段")
        start = idx * self.element_size
        end = start + self.element_size
        breath_mel = self.breath_mel_data[start:end]
        normal_mel = self.normal_mel_data[start:end]
        return breath_mel, normal_mel
        # 加载呼吸音频和正常音频



if __name__ == '__main__':
    dataset = BreathToSpeechDataset(dataset_path='../dataset')
    print(f"数据集大小{len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    print("-------------------------------------------------------\n数据集加载完成")
    num = 0
    for brea_mel, norm_mel in dataloader:
        print(brea_mel.shape, norm_mel.shape)
        num += 1
        # print(num)
