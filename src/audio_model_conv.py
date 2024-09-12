import torch
import torch.nn as nn
import torch.nn.functional as F


class BreathToSpeechModel(nn.Module) :
    """
    用于声学模型的深度卷积神经网络
    把mel图当作图片处理，用深度卷积神经网络提取特征，然后用LSTM进行时间序列建模，最后用全连接层进行输出映射
    """
    def __init__(self, input_channels=1, output_channels=1, mel_bins=128, lstm_hidden_size=128, lstm_layers=2,
                 dropout_rate=0.3) :
        super(BreathToSpeechModel, self).__init__()

        # 卷积层用于特征提取
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

        # 用于去噪的U-Net跳跃连接
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), padding=1)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), padding=1)

        # RNN用于时间序列建模
        self.lstm = nn.LSTM(input_size=mel_bins, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # 全连接层将LSTM输出映射到Mel频带数
        self.fc = nn.Linear(lstm_hidden_size * 2, mel_bins)

        # 最后的去噪层
        self.output_conv = nn.Conv2d(16, output_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x) :
        # 输入形状为 (batch_size, channels, height, width)
        # 卷积层特征提取
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        # 获取卷积层输出的维度
        batch_size, channels, height, width = x4.size()

        # 将卷积层输出重塑为LSTM输入形状
        x4_reshaped = x4.permute(0, 3, 1, 2).contiguous()  # 交换维度以便将时间序列放在第二维度
        x4_reshaped = x4_reshaped.view(batch_size, width, -1)  # 变形为 (batch_size, sequence_length, input_size)

        # LSTM进行时间序列建模
        rnn_output, _ = self.lstm(x4_reshaped)

        # 使用全连接层进行输出映射
        rnn_output = self.fc(rnn_output)  # 将LSTM输出映射回mel_bins的维度

        # 还原形状用于上采样和去噪
        rnn_output_reshaped = rnn_output.permute(0, 2, 1).view(batch_size, -1, height, width)

        # U-Net反卷积进行去噪
        x = F.relu(self.upconv3(rnn_output_reshaped) + x3)
        x = F.relu(self.upconv2(x) + x2)
        x = F.relu(self.upconv1(x) + x1)

        # 最后的卷积层进行输出映射
        output = self.output_conv(x)
        return output


# 实例化模型
model = BreathToSpeechModel()

# 打印模型参数数量
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# 打印模型结构
print(model)
