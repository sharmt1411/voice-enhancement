import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


# 实现带因果遮罩自注意力的Transformer模型块
class TransformerBlock(nn.Module) :
    def __init__(self, d_model, n_head, seq_length, dropout=0.1) :
        super(TransformerBlock, self).__init__()

        # 多头注意力层
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        ff_dim = 4 * d_model
        # 前向传播网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 创建因果遮罩
        self.seq_length = seq_length
        self.register_buffer('causal_mask', self.create_causal_mask())

    def create_causal_mask(self) :
        # 创建因果遮罩 (seq_length, seq_length)
        # mask = torch.triu(torch.ones(self.seq_length, self.seq_length), diagonal=1).bool()
        # print("因果遮罩:", mask[0:2,0:2])

        mask = nn.Transformer.generate_square_subsequent_mask(self.seq_length)
        # print("因果遮罩:", mask[0 :2, 0 :2])
        return mask

    def forward(self, x) :
        x_norm = self.layer_norm1(x)  # 层归一化
        # print("TransformerBlock输入:", x_norm.shape)  # 1, 64, 256
        # 多头自注意力
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, is_causal=False)
        # attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self.causal_mask, is_causal=False)
        # print("多头自注意力输出:", attn_output.shape)
        proj_output = self.proj(attn_output)  # 线性变换
        # 残差连接和层归一化
        x = x + self.dropout(proj_output)

        x_norm = self.layer_norm2(x)  # 层归一化
        # 前向传播网络
        ffn_output = self.ffn(x_norm)

        # 残差连接和层归一化
        x = x + self.dropout(ffn_output)

        return x


class TransformerMelModel(nn.Module) :
    def __init__(self, mel_bins=128, d_model=64, seq_length=64, n_head=2, num_decoder_layers=8, dropout=0.2) :
        super(TransformerMelModel, self).__init__()
        self.position_embedding_table = nn.Embedding(seq_length, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_head, seq_length, dropout)
                                      for _ in range(num_decoder_layers)])
        # 线性层将 mel_bins 映射到 d_model 维度
        self.input_linear = nn.Linear(mel_bins, d_model)
        # 输出层将 d_model 映射回 mel_bins 维度
        self.output_linear = nn.Linear(d_model, mel_bins)

        self.config = {
            "mel": mel_bins,
            "d": d_model,
            "seq_len": seq_length,
            "n_h": n_head,
            "layers": num_decoder_layers,
            "drop": dropout
        }

    def forward(self, x) :
        # x: 输入形状为 (batch_size, 时间帧数量seq_length, mel_bins)，而mel图的shape为(batch_size， mel_bins， 时间帧数量)需要注意变换
        # 将 mel_bins 维度的输入映射到 d_model 维度
        x = self.input_linear(x)  # (batch_size, 时间帧数量, d_model)
        # print("输入形状:", x.shape)
        pos = self.position_embedding_table(torch.arange(x.shape[1]).to(x.device))  # (时间帧数量, d_model)
        # print("位置编码:", pos.shape)
        x = x + pos  # (batch_size, 时间帧数量, d_model)
        # print("位置编码后:", x.shape)
        """此处注意，默认多头注意力序列长度，batchsize，dim形状"""
        x = self.blocks(x)  # (batch_size,时间帧数量, d_model)
        # print("TransformerBlock输出:", x.shape)

        x = F.relu(self.output_linear(x))  # (时间帧数量, batch_size, mel_bins)
        # print("输出形状:", x.shape)
        return x

    def custom_init(self, m):
        if isinstance(m, nn.Linear):
            init.uniform_(m.weight, -0.1, 0.1)  # 使用均匀分布初始化权重
            if m.bias is not None:
                init.constant_(m.bias, 0)  # 将偏置初始化为 0


if __name__ == '__main__':
    # 测试TransformerBlock

    # 模型参数设置
    d_model=256
    n_head=4
    num_decoder_layers=4
    dropout=0.2

    # 输入
    batch_size = 1
    time_frames = 64
    mel_bins = 128
    torch.manual_seed(0)  # 设置随机种子

    # 实例化模型
    model = TransformerMelModel(mel_bins=mel_bins,seq_length=time_frames,d_model=d_model,n_head=n_head,num_decoder_layers=num_decoder_layers,dropout=dropout)
    # 初始化模型参数权重
    model.apply(model.custom_init)

    # 输出模型参数数量
    print("模型参数数量:", sum(p.numel() for p in model.parameters()))  # 模型参数数量应为 110,208

    # 输入张量 (batch_size, 时间帧数量, mel_bins)
    x = torch.randn(batch_size, time_frames, mel_bins)

    # 前向传播
    output = model(x)

    print("输出形状:", output.shape)  # 输出形状应为 (1, 128, 128)
    print("输出张量:", output)
    # 测试decoder，memory zeros 输出：tensor([[[ 0.6477,  0.9012, -0.7328,  ...,  0.9868, -1.0214, -0.3526],
    # 测试decoder， memory ones 输出：tensor([[[ 0.6718, -0.3628, -0.6519,  ...,  1.6491, -1.1979,  0.2694],
    # 测试上三角矩阵输出：tensor([[[ 1.4437,  2.9433,  5.4295,  ..., -0.2881,  2.6514, -6.7360]
    # 测试官方提供的因果遮罩输出：tensor([[[ 1.4437,  2.9433,  5.4295,  ..., -0.2881,  2.6514, -6.7360]
