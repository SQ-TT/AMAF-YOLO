import torch
import torch.nn as nn


# 跨层拼接残差
class EnhancedBlock(nn.Module):
    def __init__(self, c1, c2):  # 改为接受 c1 和 c2
        super().__init__()
        assert c2 % 2 == 0, "Channel number must be even"
        self.C = c2  # 使用 c2 作为输出通道

        # 主路径（输入通道为 c1）
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 分支1：输入通道为 c1//2（假设 c1 == c2）
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1 // 2, c2 // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.ReLU(inplace=True)
        )

        # 分支2：改进的MaxPool路径
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c1 // 2, c2 // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.ReLU(inplace=True)
        )

        # 残差路径：输入通道为 c1，输出为 c2
        self.residual = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(c2)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.residual(x)
        x = self.avgpool(x)
        x1, x2 = x[:, :self.C // 2], x[:, self.C // 2:]
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.fusion(torch.cat([out, residual], dim=1))
        return out


if __name__ == "__main__":
    # 测试与验证
    model = EnhancedBlock(64, 64)
    x = torch.randn(2, 64, 32, 32)  # 使用batch_size=2测试

    # 前向传播
    out = model(x)
    print(f"Output shape: {out.shape}")  # 应为 torch.Size([2, 64, 16, 16])

    # 反向传播测试
    loss = out.sum()
    loss.backward()
    print("Backward pass completed successfully")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e3:.1f}K")