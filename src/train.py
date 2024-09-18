"""
训练3种模型
配置训练超参数，模型超参数在模型文件中单独设置
配置损失函数，采用mel图损失和STFT损失，并进行加权求和
训练模型
"""

import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.transforms import InverseMelScale, GriffinLim  # 计算可微

from  audio_model_transformer import TransformerMelModel
from  audio_model_LSTM import SimpleLSTMModel
from  audio_model_conv import BreathToSpeechModel
from src.dataset import BreathToSpeechDataset

print(torch.__version__)        # 查看PyTorch版本
print(torch.version.cuda)

# 损失函数设置
def mel_spectrogram_loss(predicted, target, loss_type='L1'):
    if loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'L2':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Unsupported loss type")

    return loss_fn(predicted, target)


# 定义计算STFT损失的函数
def stft_loss(predicted_mel, target_audio, n_fft=512, n_mels=128, sample_rate=16000) :
    """
    未测试通过
    :param predicted_mel:
    :param target_audio:
    :param n_fft:
    :param n_mels:
    :param sample_rate:
    :return:
    """
    # 输入形状：(b,  n_frames，n_mels)，需要转换mel谱图
    predicted_mel = predicted_mel.permute(0, 2, 1)
    target_audio = target_audio.permute(0, 2, 1)

    # 反向Mel滤波器组：从Mel谱图恢复到STFT幅度谱
    inv_mel_transform = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate)

    # 使用 Griffin-Lim 算法从恢复的 STFT 幅度谱重建时间域波形
    griffin_lim = GriffinLim(n_fft=n_fft)

    # 恢复时间域音频波形
    predicted_waveform = griffin_lim(inv_mel_transform(predicted_mel)) # shape: (1, n_samples)
    target_waveform = target_audio.squeeze(0).cpu().numpy()  # shape: (1, n_samples,)

    # 计算STFT
    stft_predicted = torch.stft(predicted_waveform, n_fft=n_fft, return_complex=True)
    stft_target = torch.stft(target_waveform, n_fft=n_fft, return_complex=True)

    # 计算STFT的L1损失
    stft_loss = F.l1_loss(stft_predicted.abs(), stft_target.abs())
    return stft_loss

# 定义总损失函数
def total_loss(predicted, target, alpha=0.8, beta=0.2) :
    # L1/L2 损失
    # print("计算损失", predicted.shape, target.shape)
    mel_loss = mel_spectrogram_loss(predicted, target, loss_type='L2')
    # STFT 损失
    # stft_loss_val = stft_loss(predicted, target)
    # 总损失
    # total = alpha * l1_loss + beta * stft_loss_val
    total = mel_loss
    return total


# 定义评估损失
def evaluate(model, dataset, eva_num=10):
    """
    训练验证模式损失
    :param model:
    :param dataset:dataset字典{“train”:train_dataset, “val”:val_dataset}
    :param eva_num:每次评估的样本数
    :return:
    """
    model.eval()
    return_loss = {}
    with torch.no_grad():
        num_samples = min(eva_num, len(dataset['val']))
        for split in ['train', 'val']:
            loss = 0
            # 从dataset中随机取eva_num个样本进行评估
            random_indices = random.sample(range(len(dataset[split])), num_samples)  # 随机选择10个索引
            random_samples = [dataset[split][idx] for idx in random_indices]  # 根据索引获取数据
            for input_data, target in random_samples:
                input_data = input_data.unsqueeze(0).to(device)
                target= target.unsqueeze(0).to(device)
                # print(f'evaluate:input_data shape: {input_data.shape}, target shape: {target.shape}')  # （12，128）
                predicted = model(input_data)
                loss += total_loss(predicted, target).item()
            return_loss[split] = loss / (len(dataset))
    model.train()
    return return_loss


def train(model_path, dataset, load_model=False, model_type='transformer'):
    if model_type == 'transformer':
        model = TransformerMelModel(seq_length=element_size, d_model=64, n_head=2)
    elif model_type == 'LSTM':
        model = SimpleLSTMModel()
    elif model_type == 'conv':
        model = BreathToSpeechModel(seq_len=element_size)
    else:
        raise ValueError("Unsupported model type")

    model = model.to(device)
    model_config = model.config
    print(f'Model config: {model_config}')
    # 打印模型参数数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {params}')

    if load_model:
        if device == 'cuda':
            model.load_state_dict(torch.load(os.path.join(save_path, model_name), map_location=device))
            print(f'load model successfully-{device}')
        else:
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
            print(f'load model successfully-cpu')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 加载训练数据集,并划分训练集和验证集
    dataset = BreathToSpeechDataset(dataset_path='../dataset', element_size=element_size)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    print(f'Total dataset size: {total_size}, train size: {train_size}, val size: {val_size}')

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataset_dict = {'train': dataset_train, 'val': dataset_val}
    print(f'Dataset loaded successfully, train size: {len(dataset_train)}, val size: {len(dataset_val)}')

    dataloader_train = DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=True)
    loss_train = []
    loss_val = []
    for epoch in range(max_epoch):
        print(f'Epoch {epoch+1}/{max_epoch}')
        for iter, batch in enumerate(dataloader_train):
            # print(batch[0][0].shape)  # 第一批，输入的形状， batch形状 batch_size, 2，seq_length, n_mels

            data, target = batch
            data = data.to(device)
            target = target.to(device)
            # print(f'Iter {iter + 1}/{len(dataloader_train)}, data shape: {data.shape}, target shape: {target.shape}')  #  ([8, 12, 128])
            optimizer.zero_grad()
            predicted = model(data)
            loss = total_loss(predicted, target)
            loss_train.append(loss.item())

            # print(f'Iter {iter+1}/{len(dataloader_train)}, loss: {loss.item():.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if save_interval is not None:
                if (iter+1) % save_interval == 0:
                    torch.save(model.state_dict(), os.path.join(
                    save_path, f'model_{model_type}_ep{epoch}_it{iter}_conf{model_config}.pth'))
        print(f'Epoch {epoch+1}/{max_epoch}, train loss: {loss.item():.4f}')
        print("Evaluating model...")
        eva_loss = evaluate(model, dataset_dict, eva_num=10)
        loss_val.append(eva_loss['val'])
        print(
            f'Iter {iter + 1}/{len(dataloader_train)}, train loss: {eva_loss["train"]:.4f}, valid loss: {eva_loss["val"]:.4f}')

    str_config = ''
    for key, value in model_config.items():
        str_config += f'_{key}_{value}'

    # 保存模型
    torch.save(model.state_dict(), os.path.join(
        save_path, f'model_{model_type}_{params}_{str_config}.pth'))

    # 绘制损失曲线
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.plot(loss_train)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.subplot(2, 1, 2)
    plt.plot(loss_val)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')

    plt.show()




if __name__ == '__main__':
    # 训练超参数设置
    max_epoch = 1000
    batch_size = 8
    learning_rate = 0.001
    eva_interval = 30

    save_path = '../model_save/'
    save_interval = None

    # 设置划分比例
    train_ratio = 0.9  # 90% 用作训练集
    val_ratio = 0.1  # 10% 用作验证集

    element_size = 64  # 输入数据的长度

    load_model = True  # 是否加载已训练模型
    model_name = "model_conv_701921__mel_128_seq_len_64_hidden_s_128_layers_2_dropout_0.1.pth"

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train(model_path=None, dataset=None, load_model=load_model, model_type='conv')

