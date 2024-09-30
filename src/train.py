"""
训练3种模型
配置训练超参数，模型超参数在模型文件中单独设置
配置损失函数，采用mel图损失和STFT损失，并进行加权求和
训练模型
"""
import os
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.transforms import InverseMelScale, GriffinLim  # 计算可微

from audio_model_transformer import TransformerMelModel
from audio_model_LSTM import SimpleLSTMModel
from audio_model_conv import BreathToSpeechModel
from src.dataset import BreathToSpeechDataset

print(torch.__version__)        # 查看PyTorch版本
print(torch.version.cuda)


# 损失函数设置
def mel_spectrogram_loss(predicted, target, loss_type='L2', is_log=True, is_mask=False):
    """ mel频谱图损失函数,表示单个数值损失，由于归一化，损失最大为1"""
    if loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'L2':
        if is_mask:
            loss_fn = nn.MSELoss(reduction="none")
        else:
            loss_fn = nn.MSELoss()
    else:
        raise ValueError("Unsupported loss type")
    # loss_fn_test = nn.MSELoss()
    # loss_fn_test1 = loss_fn_test(predicted, target)
    # print("loss_fn_test1:", loss_fn_test1.shape)  # []标量
    if is_mask:
        if is_log:
            threshold = 0.3  # 阈值-70db归一化 -70+100 /100 + 1 =
        else:
            threshold = 0.00001  # 如果不是log，阈值设置ref*1e-7 =300*1e-7
        mask = (target > threshold).float()  # 计算mask
        amount = mask.sum()
        # print("mask sum:", amount.item())
        # print("mask shape:", mask.shape, "amount shape:", amount.shape)
        loss = loss_fn(predicted, target)
        # print("未加权的loss-sum ", loss.sum().item())  # torch.Size([16, 64, 128])
        # print("mel_spectrogram_loss shape:", loss.shape, "mask shape:", mask.shape, "amount:", amount.shape)
        # torch.Size([16, 64, 128])
        mask[:, :, :64] *= 2  # 前64mel频率的权重为2   #注意输入是批次的 torch.Size([16, 64, 128])
        # print("mask64", mask[1, 1,:])
        weighted_mask_loss = ((loss * mask).sum()/amount)
        # print("weighted_mask_loss shape:", weighted_mask_loss.shape)  # torch.Size([16])
        # loss = weighted_mask_loss.mean()
        # print("加权的loss ", weighted_mask_loss.item())
        # print("loss shape:", loss.shape)  # scalar
        return weighted_mask_loss
    else:
        loss = loss_fn(predicted, target)
        # loss = F.smooth_l1_loss(predicted, target)
        return loss


def db_to_amplitude(mel_db_std, ref_amplitude=150):
    # 逆dB转换为线性谱
    mel_db = mel_db_std * 80-80  # 转换为标准db
    print("mel_db min:", mel_db.min().item(), "max:", mel_db.max().item())
    mel_linear = ref_amplitude * torch.pow(10.0, mel_db / 10.0)
    return mel_linear


def log_mel_spectrogram_loss(predicted, target,):
    """输入为线性mel，注意输入>0,注意log后超出范围"""
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>计算log_mel_spectrogram_loss")
    epsilon = 1e-7  # -70db, 1e-70 是-700db
    predicted = torch.clamp(predicted, min=1e-7, max=1)  # 限制值域
    log_predicted = torch.log10(predicted+ epsilon)  # [-7,0]反归一化，根据数据集dataset载入获取的最大值和最小值，呼吸与正常不同
    # log_predicted_clamp = torch.clamp(log_predicted, min=-100)    # 归一化到0-1
    # log_predicted_norm = (log_predicted+100)/100    # 归一化到0-1


    log_target = torch.log10(target + epsilon)
    # log_target_clamp = torch.clamp(log_target, min=-100)    # 归一化到0-1
    # log_target_norm = (log_target+100)/100    # 归一化到0-1
    # print("target min:", target.min().item(), "max:", target.max().item())
    print("log_target min:", log_target.min().item(), "max:", log_target.max().item())
    # print("log_target_clamp min:", log_target_clamp.min().item(), "max:", log_target_clamp.max().item())
    # print("log_target_norm min:", log_target_norm.min().item(), "max:", log_target_norm.max().item())

    # print("predicted min:", predicted.min().item(), "max:", predicted.max().item())
    print("log_predicted min:", log_predicted.min().item(), "max:", log_predicted.max().item())
    # print("log_predicted_clamp min:", log_predicted_clamp.min().item(), "max:", log_predicted_clamp.max().item())
    # print("log_predicted_norm min:", log_predicted_norm.min().item(), "max:", log_predicted_norm.max().item())
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    # loss = F.smooth_l1_loss(log_predicted, log_target)
    # loss = loss_fn(log_predicted_norm, log_target_norm)
    loss = loss_fn(log_predicted, log_target)
    return loss


# 定义计算STFT损失的函数
def stft_loss(predicted_mel, target_mel, n_fft=512, n_mels=128, sample_rate=16000):
    """未启用，报错"""
    # 输入形状：(b,  n_frames，n_mels)，需要转换mel谱图
    epsilons = 1e-5
    predicted_mel = predicted_mel.permute(0, 2, 1)+epsilons
    target_mel = target_mel.permute(0, 2, 1) + epsilons

    print("predicted_mel min:", predicted_mel.min(), "max:", predicted_mel.max())
    print("target_mel min:", target_mel.min(), "max:", target_mel.max())

    predicted_mel = torch.clamp(predicted_mel, min=1e-7, max=100)  # 限制值域
    target_mel = torch.clamp(target_mel, min=1e-7, max=100)  # 限制值域
    # 打印预处理后的统计信息
    print("After preprocessing:")
    print("predicted_mel min:", predicted_mel.min(), "max:", predicted_mel.max())
    print("target_audio min:", target_mel.min(), "max:", target_mel.max())

    inv_mel_transform = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate).to(device)

    # 使用 try-except 来捕获可能的错误
    try:
        predicted_stft = inv_mel_transform(predicted_mel)
        target_stft = inv_mel_transform(target_mel)

        # 检查 NaN 值
        if torch.isnan(predicted_stft).any():
            print("NaN values detected in predicted_stft")
        if torch.isnan(target_stft).any():
            print("NaN values detected in target_stft")

        print("predicted_stft min:", predicted_stft.min(), "max:", predicted_stft.max())
        print("target_stft min:", target_stft.min(), "max:", target_stft.max())

        # 计算STFT的L1损失
        stft_loss = F.l1_loss(predicted_stft.abs(), target_stft.abs())
        print("stft_loss", stft_loss)
        return stft_loss
    except Exception as e:
        print(f"Error occurred during inverse mel transform: {e}")
        return None

    # 使用 Griffin-Lim 算法从恢复的 STFT 幅度谱重建时间域波形
    # griffin_lim = GriffinLim(n_fft=n_fft)

    # 恢复时间域音频波形
    # predicted_waveform = griffin_lim(inv_mel_transform(predicted_mel)) # shape: (1, n_samples)
    # target_waveform = griffin_lim(inv_mel_transform(target_audio)) # shape: (1, n_samples)

    # 计算STFT的L1损失
    # stft_loss = F.l1_loss(predicted_waveform.abs(), target_waveform.abs())
    # return stft_loss


def manual_inverse_mel(mel_spec, n_fft, n_mels, sample_rate):
    # 创建 Mel 滤波器组
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_inv = np.linalg.pinv(mel_basis)
    mel_inv = torch.from_numpy(mel_inv).float().to(mel_spec.device)

    # 调整 mel_spec 的形状
    if len(mel_spec.shape) == 3:
        # 如果输入是批次数据 (batch, time, mel)
        batch_size, time_steps, _ = mel_spec.shape
        mel_spec_reshaped = mel_spec.reshape(-1, n_mels)  # (batch * time, mel)
    else:
        # 如果输入是单个频谱图 (time, mel)
        time_steps, _ = mel_spec.shape
        batch_size = 1
        mel_spec_reshaped = mel_spec

    # 应用逆变换
    stft = torch.matmul(mel_spec_reshaped, mel_inv.T)  # 注意这里使用了 mel_inv.T

    # 恢复原始形状
    stft = stft.reshape(batch_size, time_steps, n_fft // 2 + 1)

    return stft


def manual_stft_loss(predicted_mel, target_mel, n_fft=512, n_mels=128, sample_rate=16000, loss_type='L2'):
    predicted_stft = manual_inverse_mel(predicted_mel, n_fft, n_mels, sample_rate)
    target_stft = manual_inverse_mel(target_mel, n_fft, n_mels, sample_rate)
    if loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'L2':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Unsupported loss type")

    return loss_fn(predicted_stft, target_stft)


# 定义总损失函数
def total_loss(predicted, target, alpha=1.0, beta=0.0, gamma=0.0, is_eval=False):
    """各种定义loss，注意输入是否是log"""
    # L1/L2 损失
    # print("计算损失", predicted.shape, target.shape)
    total = 0
    if alpha != 0:
        mel_loss = mel_spectrogram_loss(predicted, target, loss_type='L2', is_log=True)
        total = alpha * mel_loss

    if beta != 0:
        # STFT 损失
        stft_loss_val = manual_stft_loss(predicted, target)
        total += beta * stft_loss_val
    if gamma != 0:
        # 计算mel谱图的log损失，适用于非log输入
        log_mel_loss = log_mel_spectrogram_loss(predicted, target)
        print("log_mel_loss", log_mel_loss.item())
        total += gamma * log_mel_loss

    if not is_eval:
        # print("predicted shape:", predicted.shape, "target shape:", target.shape)
        print("predicted min:", predicted.min().item(), "|max:", predicted.max().item(), "|predicted.sum ", predicted.sum().item())
        print("target min:", target.min().item(), "|max:", target.max().item(), "|target.sum ", target.sum().item())
        print("loss", total.item())
    return total


# 定义评估损失
def evaluate(model, dataset, eva_num=10, alpha=1.0, beta=0.0, gamma=0.0):
    """
    训练验证模式损失
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
                target = target.unsqueeze(0).to(device)
                # print(f'evaluate:input_data shape: {input_data.shape}, target shape: {target.shape}')  # （12，128）
                predicted = model(input_data)
                loss += total_loss(predicted, target, alpha=alpha, beta=beta, gamma=gamma, is_eval=True).item()
            return_loss[split] = loss / num_samples
    model.train()
    return return_loss


def train(model, load_model=False, model_type='transformer'):

    print("开始训练，device:", device)
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
    else:
        def init_weights(m) :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight)  # 或者使用 nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None :
                    nn.init.constant_(m.bias, 0)
        model.apply(init_weights)
        print(f'Model Xavier initialized successfully')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 加载训练数据集,并划分训练集和验证集
    dataset = BreathToSpeechDataset(n_fft=512, hop_length=128, dataset_path=dataset_path, element_size=element_size,
                                    transform=is_transform, is_norm=is_norm)
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

    str_config = ''
    for key, value in model_config.items():
        str_config += f'_{key}_{value}'

    for epoch in range(max_epoch):

        for iter, batch in enumerate(dataloader_train):
            # print(batch[0][0].shape)  # 第一批，输入的形状， batch形状 batch_size, 2，seq_length, n_mels

            data, target = batch
            data = data.to(device)
            print(f'Input data min:{data.min().item()}|max:{data.max().item()}|mean:{data.mean().item()}|std:{data.std().item()}')
            target = target.to(device)
            print(f'Target data min:{target.min().item()}|max:{target.max().item()}|mean:{target.mean().item()}|std:{target.std().item()}')
            # optimizer.zero_grad()
            predicted = model(data)
            loss = total_loss(predicted, target, alpha=1.0, beta=0.0, gamma=0.0)
            loss_train.append(loss.item())
            loss.backward()
            if (iter+1) % accumulation_steps == 0 or (iter+1) == len(dataloader_train):
                print(f'-------------------------------------------------------------------------\n'
                      f'{epoch+1}/{max_epoch} Iter {iter+1}/{len(dataloader_train)}，准备更新参数，loss: {loss.item()}')
                for param_group in optimizer.param_groups:
                    print(f"Current learning rate: {param_group['lr']}")

                for i, (name, param) in enumerate(model.named_parameters()):
                    if i == 0 or i == 5 or i == 10:  # 打印参数
                        if param.grad is not None:
                            print(
                                f"Layer: {name} | Gradient Max: {param.grad.abs().max().item()} | Gradient Min: {param.grad.abs().min().item()} | Gradient Mean: {param.grad.abs().mean().item()} | Weight Mean: {param.data.abs().mean().item()}")
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

            if (iter+1) % eva_interval == 0:
                eva_loss = evaluate(model, dataset_dict, eva_num=10, alpha=1.0, beta=0, gamma=0.0)
                print(
                    f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                    f'Iter {iter + 1}/{len(dataloader_train)}, train loss: {eva_loss["train"]:.4f}, valid loss: {eva_loss["val"]:.4f}')
                loss_val.append(eva_loss['val'])

        print(f"Epoch {epoch+1}/{max_epoch} finished, Evaluating model...")

        # eva_loss = evaluate(model, dataset_dict, eva_num=10)
        # loss_val.append(eva_loss['val'])
        # print( f'Iter {iter + 1}/{len(dataloader_train)}, train loss: {eva_loss["train"]:.4f}, valid loss: {eva_loss["val"]:.4f}')

        # 保存模型配置
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(
                save_path, f'model_{model_type}_{params}_{str_config}.pth'))
            print(f'>>>>>>>>Model saved to {save_path}model_{model_type}_{params}_{str_config}.pth')

    # 保存模型
    torch.save(model.state_dict(), os.path.join(
        save_path, f'model_{model_type}_{params}_{str_config}.pth'))
    print(f'训练结束。Model saved to {save_path}model_{model_type}_{params}_{str_config}.pth')

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
    max_epoch = 50
    batch_size = 8
    accumulation_steps = 4
    learning_rate = 1e-3
    eva_interval = 30

    save_path = '../model_save/'
    save_interval = None

    # 设置划分比例
    train_ratio = 0.9  # 90% 用作训练集
    val_ratio = 0.1  # 10% 用作验证集

    # 调整参数------------------------------------------------------------------------------------
    element_size = 64  # 输入数据的长度, transformer24,   conv和 lstm不需要设置，conv64表示输入分段
    model_type = 'conv'  # 选择模型类型
    is_transform = True  # 是否对dataset中的训练数据进行标准化变换
    is_norm = False  # True: 对dataset中的训练数据进行直接归一化 False: 对dataset中的训练数据进行log归一化

    load_model = False  # 是否加载已训练模型
    model_name = "model_conv_886529__mel_128_seq_len_64_hidden_s_128_layers_2_dropout_0.4.pth"

    dataset_path = '../dataset/aidataset/'

    torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 加载模型,设置参数
    if model_type == 'transformer':
        model = TransformerMelModel(seq_length=element_size, d_model=128, n_head=2, mel_bins=128, num_decoder_layers=8, dropout=0.4)
    elif model_type == 'lstm':
        model = SimpleLSTMModel(mel_bins=128, lstm_hidden_size=256, lstm_layers=4, dropout_rate=0.4)
    elif model_type == 'conv':
        model = BreathToSpeechModel(seq_len=element_size, lstm_hidden_size=128, lstm_layers=2, dropout_rate=0.2)
    else:
        raise ValueError("Unsupported model type")

    train(model, load_model=load_model, model_type=model_type)

    # 09241202效果较好
