"""
数据集模块
处理数据集相关的功能
转换为将来模型输入的格式，预设pyaudio采样，采样率16kHz，深度int16，块大小1024~64ms
考虑实时性要求，输入时长0.5s，nfft=512, hop_length=256,num_mel=64
"""

import os

import librosa
import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

from audio_utils import plot_mel_spectrogram_list, audio_to_mel, load_audio, mel_align





class BreathToSpeechDataset(Dataset):
    def __init__(self, dataset_path, transform=False, is_norm=False, element_size=64, num_mel=128, n_fft=512, hop_length=256, target_sr=16000,f_max=8000):
        self.element_size = element_size
        self.num_mel = num_mel
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.f_max = f_max

        self.dataset_path = dataset_path
        self.transform = transform
        self.is_norm = is_norm
        self.file_pairs = self._load_file_pairs()
        self.stats = {}
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
        print(f"数据集加载完成,采样率", self.target_sr)
        return file_pairs

    def _load_audio(self, filename):
        """
        加载音频文件，转换为mel频谱图，并合并所有mel图
        注意统一处理数据类型，使用audio_utils中的方法
        如果使用torchaudio.load，则需转换采样率并且处理数据类型，注意数据形状的差异
        """
        audio_path = os.path.join(self.dataset_path, filename)
        audio, sr = load_audio(audio_path, sr=self.target_sr)
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
            breath_mel_ = audio_to_mel(breath_waveform, self.n_fft, self.hop_length, breath_sr, self.num_mel, self.f_max)
            normal_mel_ = audio_to_mel(normal_waveform, self.n_fft, self.hop_length, normal_sr, self.num_mel, self.f_max)

            # print("对齐前mei形状", breath_mel_.shape, normal_mel_.shape)
            breath_mel, normal_mel = mel_align(breath_mel_, normal_mel_)

            # print(f"呼吸音频{breath_file}，正常音频{normal_file}，音频形状{breath_waveform.shape}, "
            #       f"{normal_waveform.shape}Mel频谱图形状{breath_mel.shape}, {normal_mel.shape}")

            # 绘制对数Mel频谱图,比较差异
            plot = False
            if plot :
                mel_list = [(breath_mel_, 'breath_before_align'), (normal_mel_, 'normal_before_align'),
                            (breath_mel, 'breath'), (normal_mel, 'normal')]
                plot_mel_spectrogram_list(mel_list, self.target_sr, self.num_mel, fmax=self.f_max, is_log=False)
                print("Mel频谱图绘制完成")

            self.breath_mel_data = np.hstack((self.breath_mel_data, breath_mel)) if self.breath_mel_data.size else breath_mel
            self.normal_mel_data = np.hstack((self.normal_mel_data, normal_mel)) if self.normal_mel_data.size else normal_mel
            # print(f"第{i+1}个音频处理完成,目前数据形状，呼吸{self.breath_mel_data.shape}, 正常{self.normal_mel_data.shape}")
        assert self.breath_mel_data.shape[1] == self.normal_mel_data.shape[1], "呼吸和正常音频时长不一致"
        self.breath_mel_data = self.breath_mel_data.transpose()  # 转换为(time, dim)
        self.normal_mel_data = self.normal_mel_data.transpose()  # 转换为(time, dim)
        self.breath_mel_data_torch = torch.from_numpy(self.breath_mel_data).float()
        self.normal_mel_data_torch = torch.from_numpy(self.normal_mel_data).float()

        # 去掉0值
        threshold = 1e-5
        max_breath_val, _ = self.breath_mel_data_torch.max(dim=1)
        max_normal_val, _ = self.normal_mel_data_torch.max(dim=1)
        valid_indices = (max_breath_val > threshold) & (max_normal_val > threshold)

        print("m-dataset-max shape", max_breath_val.shape, max_normal_val.shape)
        self.breath_mel_data_torch = self.breath_mel_data_torch[valid_indices]
        self.normal_mel_data_torch = self.normal_mel_data_torch[valid_indices]

        # 统计均值和方差
        self.stats = {'mean_breath': self.breath_mel_data_torch.mean(), 'std_breath': self.breath_mel_data_torch.std(),
                      'max-breath': self.breath_mel_data_torch.max(), 'min-breath': self.breath_mel_data_torch.min(),
                      'mean_normal': self.normal_mel_data_torch.mean(), 'std_normal': self.normal_mel_data_torch.std(),
                      'max-normal': self.normal_mel_data_torch.max(), 'min-normal': self.normal_mel_data_torch.min()}
        print(f"------------------------------------------------\n数据集加载完成，呼吸{self.breath_mel_data_torch.shape}, "
              f"正常{self.normal_mel_data_torch.shape}")
        print(f"呼吸音频均值{self.stats['mean_breath']}, 方差{self.stats['std_breath']}，最大{self.stats['max-breath']}")
        print(f"正常音频均值{self.stats['mean_normal']}, 方差{self.stats['std_normal']}, 最大{self.stats['max-normal']}")

    def __len__(self):
        return self.breath_mel_data_torch.shape[0]//self.element_size

    def __getitem__(self, idx):
        # print(f"加载第{idx}个音频片段")
        start = idx * self.element_size
        end = start + self.element_size
        breath_mel = self.breath_mel_data_torch[start:end]
        normal_mel = self.normal_mel_data_torch[start:end]
        if self.transform :
            breath_mel = self._normalize(breath_mel, is_breath=True)
            normal_mel = self._normalize(normal_mel, is_breath=False) # 输入必须标准化，输出可以测试不标准化
        return breath_mel, normal_mel  # （time，num_mel）
        # 加载呼吸音频和正常音频

    def _normalize(self, mel, is_breath=True):
        """
        输入数据的处理，由于输入>0，标准化效果不好，所以采用归一化，
        存在两种方法
        1.直接归一化，除以最大值
        2.先对数转换为-100，0db的log范围，再归一化，生成后进行指数恢复，但测试噪声较大
        """
        if self.is_norm:

            if is_breath:
                return (mel) / self.stats['max-breath']
            else:
                return (mel) / self.stats['max-normal']

        else:
            # 取logmel作为输入
            print("-----------------------------------------------------\n"
                  "m-dataset-input_mel.min", mel.min().item(), "mel.max()", mel.max().item())
            if is_breath:
                print("is_breath")
                mel = librosa.amplitude_to_db(mel, ref= self.stats['max-breath'].item())
            else:
                print("not is_breath")
                mel = librosa.amplitude_to_db(mel, ref= self.stats['max-normal'].item(), amin=1e-5)
            print("m-dataset-mel_log_db,mel.min", mel.min(), "mel.max()", mel.max())
            min_val = -70
            max_val = 0
            mel_clip = np.clip(mel, min_val, max_val)
            mel_norm = (mel_clip - min_val) / (max_val - min_val)

            print("m-dataset-mel_log_clip_norm.min", mel_norm.min(), "mel_norm.max()", mel_norm.max())
            mel_restore = librosa.db_to_amplitude(mel_norm * (max_val - min_val) + min_val, ref= self.stats['max-breath'].item())
            print("m-dataset-mel_log_clip_norm_restore.min", mel_restore.min(),"mel_restore.max()", mel_restore.max())
            return torch.from_numpy(mel_norm)


            # eps = -70   # 注意此处为分贝，db，对应数据加载时已经删除一遍
            # if mel_max < eps:
            #     print("m-dataset-Warning: empty mel spectrogram")
            #     mel = mel.clip(eps, max_val)
            # else:
            #     mel_norm = (mel - mel_min) / (mel_max - mel_min)
            # return torch.from_numpy(mel_norm)
            # 归一化、

    def _analyze_dataset(self):
        """
        分析数据集，查看数据分布，是否有异常值
        """




if __name__ == '__main__':
    # target_sr = 16000
    # n_fft = 512
    # hop_length = 256
    # num_mel = 128
    # f_max = 8000
    # element_size = 128  # 输入块的大小
    dataset = BreathToSpeechDataset(dataset_path='../dataset/aidataset', transform=True)
    print(f"数据集大小{len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    print("-------------------------------------------------------\n数据集加载完成")
    num = 0
    for brea_mel, norm_mel in dataloader:
        print(f"第{num+1}个batch")
        print(brea_mel.shape, norm_mel.shape)
        num += 1
        print(num)

# dataset统计 4056x128
# 呼吸音频均值0.14538872241973877, 方差0.9532793164253235
# 正常音频均值2.979012966156006, 方差18.868282318115234
# datasetai统计 18583x128
# 呼吸音频均值0.21608571708202362, 方差1.563319444656372
# 正常音频均值0.46226173639297485, 方差3.082866907119751