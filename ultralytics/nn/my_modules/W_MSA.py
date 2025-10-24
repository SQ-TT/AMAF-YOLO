import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class W_MSA(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(8, 8),
        num_heads=8,
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


if __name__ == "__main__":
    # Test with 4D input (B, C, H, W)
    model = W_MSA(dim=256, window_size=(8, 8), num_heads=8)
    x = torch.randn(16, 256, 80, 80)  # (B, C, H, W)

    # Forward pass
    out = model(x)
    print(f"Input shape: {x.shape}")  # torch.Size([2, 64, 32, 32])
    print(f"Output shape: {out.shape}")  # torch.Size([2, 64, 32, 32])

    # Backward pass test
    loss = out.sum()
    loss.backward()
    print("Backward pass completed successfully")

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e3:.1f}K")