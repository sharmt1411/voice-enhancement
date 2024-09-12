# 首先模型输入mel图，按照时间帧顺序进入n层LSTM或者其他RNN网络中，然后经过n层线性层，得到输出mel帧
import torch
import torch.nn as nn


class SimpleLSTMModel(nn.Module) :
    def __init__(self, mel_bins=128, lstm_hidden_size=128, lstm_layers=2, dropout_rate=0.3) :
        super(SimpleLSTMModel, self).__init__()

        # LSTM用于时间序列建模
        self.lstm = nn.LSTM(input_size=mel_bins, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # 全连接层将LSTM输出映射到Mel频带数
        self.fc = nn.Linear(lstm_hidden_size * 2, mel_bins)

    def forward(self, x) :
        # x: 输入形状为 (batch_size, 时间帧数量, mel_bins)

        # LSTM进行时间序列建模
        rnn_output, _ = self.lstm(x)

        # 使用全连接层进行输出映射
        output = self.fc(rnn_output)  # 输出形状为 (batch_size, 时间帧数量, mel_bins)

        return output


# 示例输入
batch_size = 1
time_frames = 128
mel_bins = 128

# 实例化模型
model = SimpleLSTMModel(mel_bins=mel_bins)

# 打印模型参数数量
print("模型参数数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# 输入张量 (batch_size, 时间帧数量, mel_bins)
x = torch.randn(batch_size, time_frames, mel_bins)

# 前向传播
output = model(x)

print("输出形状:", output.shape)  # 输出形状应为 (1, 128, 128)
