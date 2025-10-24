import torch
import torch.nn as nn
import torch.nn.functional as F


class CAB(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAB, self).__init__()

        # 第一部分：双卷积路径
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 第二部分：注意力路径
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv3 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        # 第一部分处理
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        residual = out  # 保留原始输入作为残差

        # 第二部分注意力分支
        attention = self.global_pool(out)
        attention = self.conv3(attention)
        attention = F.relu(attention)
        attention = self.conv4(attention)
        attention = torch.sigmoid(attention)  # 生成0-1的注意力权重

        # 残差连接和注意力相乘
        out = residual * attention

        return out


if __name__ == "__main__":
    # 测试与验证
    model = CAB(64)
    x = torch.randn(2, 64, 32, 32)  # 使用batch_size=2测试

    # 前向传播
    out = model(x)
    print(f"Output shape: {out.shape}")  # 应为 torch.Size([2, 64, 32, 32])

    # 反向传播测试
    loss = out.sum()
    loss.backward()
    print("Backward pass completed successfully")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e3:.1f}K")