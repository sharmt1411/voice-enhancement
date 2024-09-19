import torch
import torch.nn as nn
import torch.nn.functional as F


class BreathToSpeechModel(nn.Module) :
    """
    用于声学模型的深度卷积神经网络
    把mel图当作图片处理，用深度卷积神经网络提取特征，然后用LSTM进行时间序列建模，最后用全连接层进行输出映射
    """
    def __init__(self, input_channels=1, output_channels=1, mel_bins=128, seq_len=128, lstm_hidden_size=128, lstm_layers=2,
                 dropout_rate=0.1) :
        super(BreathToSpeechModel, self).__init__()
        self.seq_len = seq_len
        self.mel_bins = mel_bins
        self.config = {
            'mel': mel_bins,
            'seq_len': seq_len,
            'hidden_s': lstm_hidden_size,
            'layers': lstm_layers,
            'dropout': dropout_rate
            }

        # 卷积层用于特征提取
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

        # 用于去噪的U-Net跳跃连接
        # self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), padding=1)
        # self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), padding=1)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), padding=1)

        # RNN用于时间序列建模
        self.lstm = nn.LSTM(input_size= 128, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # 全连接层将LSTM输出映射到Mel频带数
        self.fc = nn.Linear(lstm_hidden_size*2, mel_bins)

        # 最后的去噪层
        self.output_conv = nn.Conv2d(16, output_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x) :
        """
        输入形状为 (batch_size, frames, mel_bins)
        """
        # 输入形状应为 (batch_size, channels, height, width)
        # print("input.shape:", x.shape)  # torch.Size([8, 64, 128])
        x = x.unsqueeze(1)  # 增加通道维度(batch_size,1, frames, mel_bins)
        x = x.permute(0, 1, 3, 2)  # (batch_size,1, mel_bins, frames,)
        # print("input-processed.shape:", x.shape)  # torch.Size([8, 1, 128, 64])

        # 卷积层特征提取
        x1 = F.relu(self.conv1(x))
        # print(x1.shape)    # torch.Size([8, 16, 128, 64])
        x2 = F.relu(self.conv2(x1))   # torch.Size([8, 32, 128, 64])
        # print(x2.shape)
        # x3 = F.relu(self.conv3(x2))
        # x4 = F.relu(self.conv4(x3))
        # print(x3.shape)   # torch.Size([8, 64, 128, 128])
        # print(x4.shape)   # torch.Size([8, 128, 128, 128])

        x4 = x2
        # 获取卷积层输出的维度
        batch_size, channels, height, width = x4.size()

        # 将卷积层输出重塑为LSTM输入形状
        x4_reshaped = x4.permute(0, 1, 3, 2).contiguous()  # 交换维度以便将时间序列放在第二维度 (batch_size,channels, width, height)
        x4_reshaped = x4_reshaped.view(batch_size*channels,  width, height)  # 变形为 (batch_size*channels, sequence_length, input_size)
        # print("x4_reshaped.shape:", x4_reshaped.shape)   # [256, 64, 128]
        # LSTM进行时间序列建模
        rnn_output, _ = self.lstm(x4_reshaped)   # [256, 64, 256]
        # print("rnn_output.shape:", rnn_output.shape)

        # 使用全连接层进行输出映射
        rnn_output2 = self.fc(rnn_output)  # 将LSTM输出映射回mel_bins的维度
        # print("rnn_output2.shape:", rnn_output2.shape)  # torch.Size([256, 64, 128])
        # 还原形状用于上采样和去噪
        rnn_output_reshaped = rnn_output2.permute(0, 2, 1).view(batch_size, -1, height, width)   # [8, 32, 128, 64]
        # print("rnn_output_reshaped.shape:", rnn_output_reshaped.shape)
        # U-Net反卷积进行去噪
        # x = F.relu(self.upconv3(rnn_output_reshaped) + x3)
        # x = F.relu(self.upconv2(rnn_output_reshaped) + x2)
        x = F.relu(self.upconv1(rnn_output_reshaped) + x1)
        # print("x.shape:", x.shape)  # torch.Size([8, 16, 128, 64])

        # 最后的卷积层进行输出映射
        output = self.output_conv(x) # [8, 1, 128, 64]
        output = output.squeeze(1).permute(0, 2, 1)  # 去掉通道维度
        # print("output.shape:", output.shape)  # torch.Size([8, 64, 128]),对应输入
        return output


if __name__ == '__main__' :
    # 实例化模型
    model = BreathToSpeechModel()

    # 打印模型参数数量
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 打印模型结构
    print(model)
