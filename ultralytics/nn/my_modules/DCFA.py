import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class W_MSA(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(8, 8),
        num_heads=12,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # (2*Wh-1 * 2*Ww-1, nH)

        # Relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, H, W)  # 4D input
            mask: (0/-inf) mask with shape (num_windows, Wh*Ww, Wh*Ww) or None
        Returns:
            x: (B, C, H, W)  # 4D output
        """
        B, C, H, W = x.shape
        Wh, Ww = self.window_size

        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (Wh - W % Wh) % Wh
        pad_b = (Ww - H % Ww) % Ww
        if pad_r > 0 or pad_b > 0:
            x = nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        # Reshape to windows
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = x.view(B, Hp // Wh, Wh, Wp // Ww, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, num_windows_h, num_windows_w, Wh, Ww, C)
        x = x.view(-1, Wh * Ww, C)  # (B*num_windows, Wh*Ww, C)

        # Standard attention computation
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Wh * Ww, Wh * Ww, -1)  # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back to (B, C, H, W)
        x = x.view(B, Hp // Wh, Wp // Ww, Wh, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, Hp, Wp, C)
        x = x.view(B, Hp, Wp, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, Hp, Wp)

        # Remove padding if needed
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W]

        return x


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


class CADP_Block(nn.Module):
    def __init__(self, dim, window_size=(8, 8), num_heads=12, reduction_ratio=16,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        # Part 1: Main path
        self.norm1 = nn.LayerNorm(dim)
        self.cab = CAB(dim, reduction_ratio)
        self.w_msa = W_MSA(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

        # Part 2: MLP path
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)  # 4D input
        Returns:
            x: (B, C, H, W)  # 4D output
        """
        B, C, H, W = x.shape

        # Part 1: Dual attention path
        residual1 = x

        # Process path 2
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Split path 2 into 2.1 (CAB) and 2.2 (W-MSA)
        x_cab = self.cab(x_norm)
        x_wmsa = self.w_msa(x_norm)

        # Concatenate all three paths
        x_out = torch.cat([residual1.unsqueeze(1), x_cab.unsqueeze(1), x_wmsa.unsqueeze(1)], dim=1)
        x_out = torch.sum(x_out, dim=1)  # Sum instead of mean for better gradient flow

        # Part 2: MLP path
        residual2 = x_out
        x_out = self.norm2(x_out.permute(0, 2, 3, 1))  # (B, H, W, C)
        x_out = self.mlp(x_out)
        x_out = x_out.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Final output
        output = residual2 + x_out

        return output


class GS_EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(GS_EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class DCFA(nn.Module):
    def __init__(self, dim, window_size=(7, 7), num_heads=8, reduction_ratio=32):
        super().__init__()
        self.gs_ema = GS_EMA(dim)
        self.cadp = CADP_Block(dim, window_size, num_heads, reduction_ratio)

    def forward(self, x):
        x_gs = self.gs_ema(x)
        x_cadp = self.cadp(x)
        return x_gs + x_cadp   # 特征相加


if __name__ == "__main__":
    # 测试与验证
    model = DCFA(512)
    x = torch.randn(16, 512, 80, 80)  # 使用batch_size=2测试

    # 前向传播
    out = model(x)
    print(f"Output shape: {out.shape}")  # 应为 torch.Size([2, 64, 32, 32])

    # 反向传播测试
    loss = out.sum()
    loss.backward()
    print("Backward pass completed successfully")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")