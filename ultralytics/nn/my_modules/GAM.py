import torch.nn as nn  # 从 PyTorch 导入神经网络模块，简称 nn
import torch  # 导入 PyTorch 库，用于深度学习和张量操作


# 定义 GAM_Attention 类，该类继承自 nn.Module
class GAM(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM, self).__init__()
        # 初始化 GAM_Attention 模块
        # in_channels: 输入特征图的通道数
        # rate: 缩减比率，用于通道和空间注意力的通道缩减

        # 定义通道注意力模块 (Channel Attention)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),  # 线性层，将通道数缩减为 1/rate
            nn.ReLU(inplace=True),  # 激活函数，ReLU
            nn.Linear(int(in_channels / rate), in_channels)  # 线性层，将通道数恢复为原始通道数
        )

        # 定义空间注意力模块 (Spatial Attention)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  # 卷积层，卷积核大小为 7x7，填充为 3
            nn.BatchNorm2d(int(in_channels / rate)),  # 批量归一化
            nn.ReLU(inplace=True),  # 激活函数，ReLU
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  # 卷积层，将通道数恢复为原始通道数
            nn.BatchNorm2d(in_channels)  # 批量归一化
        )

    def forward(self, x):
        # 前向传播函数，定义了数据通过网络时的计算过程
        b, c, h, w = x.shape  # 获取输入 x 的形状 (batch_size, channels, height, width)

        # 计算通道注意力
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  # 将输入 x 的维度从 (b, c, h, w) 变换为 (b, h*w, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  # 通过通道注意力模块，输出形状为 (b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  # 变换回原始形状 (b, c, h, w)，并通过 Sigmoid 函数

        # 应用通道注意力
        x = x * x_channel_att  # 将原始输入 x 与通道注意力权重相乘

        # 计算空间注意力
        x_spatial_att = self.spatial_attention(x).sigmoid()  # 通过空间注意力模块，输出形状为 (b, c, h, w)，并通过 Sigmoid 函数
        out = x * x_spatial_att  # 将通道加权后的输入与空间注意力权重相乘

        return out  # 返回最终的注意力加权输出


# 测试代码块
if __name__ == '__main__':
    x = torch.randn(1, 64, 7, 7)  # 生成一个随机张量，形状为 (1, 64, 7, 7)
    b, c, h, w = x.shape  # 获取输入张量的形状
    net = GAM(in_channels=c)  # 创建 GAM_Attention 实例，输入通道数为 64
    y = net(x)  # 通过 GAM_Attention 模块进行前向传播
    print(y.size())  # 打印输出张量的形状，验证输出是否正确
