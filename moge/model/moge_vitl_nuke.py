"""
Hardcoded TorchScript-compatible MoGe ViT-L model for Nuke.

This module is specifically designed for moge-2-vitl models with:
- DINOv2 ViT-L backbone (embed_dim=1024, depth=24, num_heads=16)
- 4 intermediate layers at indices [5, 11, 17, 23]
- 5-level neck/head architecture

All loops are unrolled and dimensions are hardcoded for TorchScript compatibility.
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Geometry utilities (pure PyTorch, no numpy/utils3d)
# ============================================================================

def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float,
    dtype: torch.dtype,
    device: torch.device
) -> Tensor:
    """UV coordinates normalized by diagonal."""
    span_x = aspect_ratio / (1.0 + aspect_ratio ** 2) ** 0.5
    span_y = 1.0 / (1.0 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)

    u_grid = u.unsqueeze(0).expand(height, -1)
    v_grid = v.unsqueeze(1).expand(-1, width)
    return torch.stack([u_grid, v_grid], dim=-1)


# ============================================================================
# ViT-L Components (hardcoded for DINOv2 ViT-L)
# ============================================================================

class PatchEmbed(nn.Module):
    """Patch embedding for ViT-L."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head attention for ViT-L (16 heads, dim=1024)."""

    scale: float

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 16
        self.scale = 64 ** -0.5  # head_dim ** -0.5
        self.qkv = nn.Linear(1024, 1024 * 3, bias=True)
        self.proj = nn.Linear(1024, 1024, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 16, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Manual attention (Nuke doesn't support F.scaled_dot_product_attention)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)

        x = x.permute(0, 2, 1, 3).reshape(B, N, 1024)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """MLP for ViT-L."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """Transformer block for ViT-L."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(1024, eps=1e-6)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(1024, eps=1e-6)
        self.mlp = Mlp()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTLBackbone(nn.Module):
    """
    DINOv2 ViT-L backbone with hardcoded architecture.

    - 24 transformer blocks
    - embed_dim = 1024
    - num_heads = 16
    - patch_size = 14
    - 4 register tokens
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 1024))  # 37*37 + 1 = 1370 for 518x518
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, 1024))

        # 24 blocks - explicitly created
        self.block0 = Block()
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.block4 = Block()
        self.block5 = Block()
        self.block6 = Block()
        self.block7 = Block()
        self.block8 = Block()
        self.block9 = Block()
        self.block10 = Block()
        self.block11 = Block()
        self.block12 = Block()
        self.block13 = Block()
        self.block14 = Block()
        self.block15 = Block()
        self.block16 = Block()
        self.block17 = Block()
        self.block18 = Block()
        self.block19 = Block()
        self.block20 = Block()
        self.block21 = Block()
        self.block22 = Block()
        self.block23 = Block()

        self.norm = nn.LayerNorm(1024, eps=1e-6)

    def interpolate_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        """Interpolate position embeddings."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0:1, :]
        patch_pos_embed = pos_embed[:, 1:, :]

        h0 = h // 14
        w0 = w // 14
        M = int(math.sqrt(float(N)))

        patch_pos_embed = patch_pos_embed.reshape(1, M, M, 1024).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h0 * w0, 1024)

        return torch.cat([class_pos_embed, patch_pos_embed], dim=1).to(x.dtype)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass returning 4 intermediate features at layers [5, 11, 17, 23].

        Returns: (feat5, feat11, feat17, feat23) each of shape (B, N, 1024)
        """
        B, C, H, W = x.shape

        # Patch embed
        x = self.patch_embed(x)

        # Add cls token and position encoding
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)

        # Add register tokens after cls
        x = torch.cat([x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]], dim=1)

        # Run blocks and capture intermediate outputs
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        feat5 = self.norm(x)

        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        feat11 = self.norm(x)

        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        feat17 = self.norm(x)

        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        feat23 = self.norm(x)

        return feat5, feat11, feat17, feat23


# ============================================================================
# Encoder (projects 4 intermediate features to 1024 channels and sums)
# ============================================================================

class Encoder(nn.Module):
    """Encoder that projects 4 ViT features and sums them."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ViTLBackbone()

        # 4 projection layers (one per intermediate layer)
        self.proj0 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.proj1 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.proj2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.proj3 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: Tensor, token_h: int, token_w: int) -> Tensor:
        """
        Args:
            image: (B, 3, H, W) in [0, 1]
            token_h: number of patch rows
            token_w: number of patch cols

        Returns:
            features: (B, 1024, token_h, token_w)
        """
        B = image.shape[0]

        # Resize to patch grid
        image_14 = F.interpolate(image, size=(token_h * 14, token_w * 14), mode='bilinear', align_corners=False)
        image_14 = (image_14 - self.image_mean) / self.image_std

        # Get 4 intermediate features
        feat5, feat11, feat17, feat23 = self.backbone(image_14)

        # Remove cls and register tokens, reshape to 2D
        # Tokens are: [cls, reg0, reg1, reg2, reg3, patches...]
        feat5 = feat5[:, 5:, :].permute(0, 2, 1).reshape(B, 1024, token_h, token_w)
        feat11 = feat11[:, 5:, :].permute(0, 2, 1).reshape(B, 1024, token_h, token_w)
        feat17 = feat17[:, 5:, :].permute(0, 2, 1).reshape(B, 1024, token_h, token_w)
        feat23 = feat23[:, 5:, :].permute(0, 2, 1).reshape(B, 1024, token_h, token_w)

        # Project and sum
        out = self.proj0(feat5) + self.proj1(feat11) + self.proj2(feat17) + self.proj3(feat23)

        return out


# ============================================================================
# Neck/Head building blocks
# ============================================================================

class ResBlock(nn.Module):
    """Residual conv block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class Upsample(nn.Module):
    """2x upsample with conv transpose."""

    def __init__(self, in_ch: int, out_ch: int, mode: str = 'conv_transpose') -> None:
        super().__init__()
        self.mode = mode
        if mode == 'conv_transpose':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='replicate')
        else:  # bilinear
            self.up = nn.Identity()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'conv_transpose':
            x = self.up(x)
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
            x = self.conv(x)
        return x


# ============================================================================
# Neck (5 levels: 1024 -> 256 -> 128 -> 64 -> 32)
# ============================================================================

class Neck(nn.Module):
    """
    5-level neck.

    Level 0: 1026 (1024 + 2 UV) -> 1024, no res blocks
    Level 1: 2 (UV) + prev -> 256, 2 res blocks
    Level 2: 2 (UV) + prev -> 128, 2 res blocks
    Level 3: 2 (UV) + prev -> 64, 2 res blocks
    Level 4: 2 (UV) + prev -> 32, no res blocks
    """

    def __init__(self) -> None:
        super().__init__()
        # Level 0: input projection only
        self.input0 = nn.Conv2d(1026, 1024, kernel_size=1)
        self.up0 = Upsample(1024, 256, 'conv_transpose')

        # Level 1
        self.input1 = nn.Conv2d(2, 256, kernel_size=1)
        self.res1_0 = ResBlock(256)
        self.res1_1 = ResBlock(256)
        self.up1 = Upsample(256, 128, 'conv_transpose')

        # Level 2
        self.input2 = nn.Conv2d(2, 128, kernel_size=1)
        self.res2_0 = ResBlock(128)
        self.res2_1 = ResBlock(128)
        self.up2 = Upsample(128, 64, 'conv_transpose')

        # Level 3
        self.input3 = nn.Conv2d(2, 64, kernel_size=1)
        self.res3_0 = ResBlock(64)
        self.res3_1 = ResBlock(64)
        self.up3 = Upsample(64, 32, 'bilinear')

        # Level 4: no res blocks
        self.input4 = nn.Conv2d(2, 32, kernel_size=1)

    def forward(
        self,
        feat0: Tensor,  # (B, 1026, H0, W0)
        feat1: Tensor,  # (B, 2, H1, W1)
        feat2: Tensor,  # (B, 2, H2, W2)
        feat3: Tensor,  # (B, 2, H3, W3)
        feat4: Tensor,  # (B, 2, H4, W4)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns features at all 5 levels."""
        # Level 0
        x0 = self.input0(feat0)
        out0 = x0
        x = self.up0(x0)

        # Level 1
        x1 = self.input1(feat1)
        x = x + x1
        x = self.res1_0(x)
        x = self.res1_1(x)
        out1 = x
        x = self.up1(x)

        # Level 2
        x2 = self.input2(feat2)
        x = x + x2
        x = self.res2_0(x)
        x = self.res2_1(x)
        out2 = x
        x = self.up2(x)

        # Level 3
        x3 = self.input3(feat3)
        x = x + x3
        x = self.res3_0(x)
        x = self.res3_1(x)
        out3 = x
        x = self.up3(x)

        # Level 4
        x4 = self.input4(feat4)
        x = x + x4
        out4 = x

        return out0, out1, out2, out3, out4


# ============================================================================
# Output heads (points and mask)
# ============================================================================

class PointsHead(nn.Module):
    """Points prediction head - outputs 3 channels at level 4."""

    def __init__(self) -> None:
        super().__init__()
        # Level 0: just pass through
        self.input0 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.up0 = Upsample(1024, 256, 'conv_transpose')

        # Level 1
        self.input1 = nn.Conv2d(256, 256, kernel_size=1)
        self.res1 = ResBlock(256)
        self.up1 = Upsample(256, 128, 'conv_transpose')

        # Level 2
        self.input2 = nn.Conv2d(128, 128, kernel_size=1)
        self.res2 = ResBlock(128)
        self.up2 = Upsample(128, 64, 'conv_transpose')

        # Level 3
        self.input3 = nn.Conv2d(64, 64, kernel_size=1)
        self.res3 = ResBlock(64)
        self.up3 = Upsample(64, 32, 'bilinear')

        # Level 4: output
        self.input4 = nn.Conv2d(32, 32, kernel_size=1)
        self.output = nn.Conv2d(32, 3, kernel_size=1)

    def forward(
        self,
        f0: Tensor,
        f1: Tensor,
        f2: Tensor,
        f3: Tensor,
        f4: Tensor,
    ) -> Tensor:
        # Level 0
        x = self.input0(f0)
        x = self.up0(x)

        # Level 1
        x = x + self.input1(f1)
        x = self.res1(x)
        x = self.up1(x)

        # Level 2
        x = x + self.input2(f2)
        x = self.res2(x)
        x = self.up2(x)

        # Level 3
        x = x + self.input3(f3)
        x = self.res3(x)
        x = self.up3(x)

        # Level 4
        x = x + self.input4(f4)
        return self.output(x)


class MaskHead(nn.Module):
    """Mask prediction head - outputs 1 channel at level 4."""

    def __init__(self) -> None:
        super().__init__()
        self.input0 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.up0 = Upsample(1024, 256, 'conv_transpose')

        self.input1 = nn.Conv2d(256, 256, kernel_size=1)
        self.res1 = ResBlock(256)
        self.up1 = Upsample(256, 128, 'conv_transpose')

        self.input2 = nn.Conv2d(128, 128, kernel_size=1)
        self.res2 = ResBlock(128)
        self.up2 = Upsample(128, 64, 'conv_transpose')

        self.input3 = nn.Conv2d(64, 64, kernel_size=1)
        self.res3 = ResBlock(64)
        self.up3 = Upsample(64, 32, 'bilinear')

        self.input4 = nn.Conv2d(32, 32, kernel_size=1)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(
        self,
        f0: Tensor,
        f1: Tensor,
        f2: Tensor,
        f3: Tensor,
        f4: Tensor,
    ) -> Tensor:
        x = self.input0(f0)
        x = self.up0(x)

        x = x + self.input1(f1)
        x = self.res1(x)
        x = self.up1(x)

        x = x + self.input2(f2)
        x = self.res2(x)
        x = self.up2(x)

        x = x + self.input3(f3)
        x = self.res3(x)
        x = self.up3(x)

        x = x + self.input4(f4)
        return self.output(x)


# ============================================================================
# Main model
# ============================================================================

class MoGeViTL(nn.Module):
    """
    Complete MoGe ViT-L model for Nuke export.

    Input: (1, 3, H, W) RGB image in [0, 1]
    Output: (1, 4, H, W) where channels are [x, y, z, mask]
    """

    output_mode: int

    def __init__(self, output_mode: int = 0) -> None:
        """
        Args:
            output_mode: 0=depth only (1ch), 1=points (3ch), 2=points+mask (4ch)
        """
        super().__init__()
        self.output_mode = output_mode

        self.encoder = Encoder()
        self.neck = Neck()
        self.points_head = PointsHead()
        self.mask_head = MaskHead()

    def forward(self, image: Tensor) -> Tensor:
        """
        Args:
            image: (B, 3, H, W) in [0, 1]

        Returns:
            Based on output_mode:
            - 0: (B, 1, H, W) depth
            - 1: (B, 3, H, W) points
            - 2: (B, 4, H, W) points + mask
        """
        B = image.shape[0]
        H = image.shape[2]
        W = image.shape[3]
        device = image.device
        dtype = image.dtype
        aspect_ratio = float(W) / float(H)

        # Calculate token grid size (target ~2400 tokens)
        num_tokens = 2400
        token_h = int(round((float(num_tokens) / aspect_ratio) ** 0.5))
        token_w = int(round((float(num_tokens) * aspect_ratio) ** 0.5))

        # Encoder
        enc_feat = self.encoder(image, token_h, token_w)  # (B, 1024, token_h, token_w)

        # Build feature pyramid with UV
        h0, w0 = token_h, token_w
        uv0 = normalized_view_plane_uv(w0, h0, aspect_ratio, dtype, device)
        uv0 = uv0.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        feat0 = torch.cat([enc_feat, uv0], dim=1)  # (B, 1026, h0, w0)

        h1, w1 = h0 * 2, w0 * 2
        uv1 = normalized_view_plane_uv(w1, h1, aspect_ratio, dtype, device)
        feat1 = uv1.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        h2, w2 = h1 * 2, w1 * 2
        uv2 = normalized_view_plane_uv(w2, h2, aspect_ratio, dtype, device)
        feat2 = uv2.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        h3, w3 = h2 * 2, w2 * 2
        uv3 = normalized_view_plane_uv(w3, h3, aspect_ratio, dtype, device)
        feat3 = uv3.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        h4, w4 = h3 * 2, w3 * 2
        uv4 = normalized_view_plane_uv(w4, h4, aspect_ratio, dtype, device)
        feat4 = uv4.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        # Neck
        n0, n1, n2, n3, n4 = self.neck(feat0, feat1, feat2, feat3, feat4)

        # Heads
        points_raw = self.points_head(n0, n1, n2, n3, n4)  # (B, 3, h4, w4)
        mask_raw = self.mask_head(n0, n1, n2, n3, n4)  # (B, 1, h4, w4)

        # Resize to original
        points = F.interpolate(points_raw, size=(H, W), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask_raw, size=(H, W), mode='bilinear', align_corners=False)

        # Apply exp remapping (as per model config)
        xy = points[:, :2, :, :]
        z = points[:, 2:3, :, :]
        z_exp = torch.exp(z)
        points = torch.cat([xy * z_exp, z_exp], dim=1)

        # Sigmoid for mask
        mask = torch.sigmoid(mask)

        # Format output
        if self.output_mode == 0:
            # Depth only
            return points[:, 2:3, :, :]
        elif self.output_mode == 1:
            # Points only
            return points
        else:
            # Points + mask
            return torch.cat([points, mask], dim=1)


def load_weights_from_checkpoint(model: MoGeViTL, checkpoint_path: str) -> None:
    """
    Load weights from original MoGe checkpoint into hardcoded model.

    This maps the original model's state dict keys to the new hardcoded structure.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in ckpt:
        src_state = ckpt['model']
    else:
        src_state = ckpt

    # Build mapping from original to new keys
    dst_state = model.state_dict()
    new_state = {}

    for dst_key in dst_state.keys():
        # Try to find matching source key
        src_key = None

        # Encoder backbone blocks
        if 'encoder.backbone.block' in dst_key:
            # Extract block number
            parts = dst_key.split('.')
            block_idx = int(parts[2].replace('block', ''))
            rest = '.'.join(parts[3:])
            src_key = f'encoder.backbone.blocks.{block_idx}.{rest}'

        # Encoder projections
        elif dst_key.startswith('encoder.proj'):
            idx = int(dst_key[12])  # proj0, proj1, etc.
            rest = dst_key[14:]  # weight or bias
            src_key = f'encoder.output_projections.{idx}.{rest}'

        # Encoder backbone other
        elif dst_key.startswith('encoder.backbone.'):
            src_key = dst_key

        # Encoder buffers
        elif dst_key.startswith('encoder.image_'):
            src_key = dst_key

        # Neck - need to map carefully
        elif dst_key.startswith('neck.'):
            # This requires manual mapping based on the original ConvStack structure
            # Original neck has: input_blocks, res_blocks, resamplers, output_blocks
            src_key = _map_neck_key(dst_key)

        # Points head
        elif dst_key.startswith('points_head.'):
            src_key = _map_head_key(dst_key, 'points_head')

        # Mask head
        elif dst_key.startswith('mask_head.'):
            src_key = _map_head_key(dst_key, 'mask_head')

        if src_key and src_key in src_state:
            new_state[dst_key] = src_state[src_key]
        elif dst_key in src_state:
            new_state[dst_key] = src_state[dst_key]
        else:
            print(f"Warning: Could not find source for {dst_key}")
            new_state[dst_key] = dst_state[dst_key]

    model.load_state_dict(new_state)


def _map_neck_key(dst_key: str) -> Optional[str]:
    """Map new neck key to original ConvStack key."""
    # Neck structure:
    # input0 -> input_blocks.0
    # up0 -> resamplers.0
    # input1 -> input_blocks.1
    # res1_0, res1_1 -> res_blocks.1.0, res_blocks.1.1
    # etc.

    parts = dst_key.split('.')
    name = parts[1]  # input0, up0, res1_0, etc.
    rest = '.'.join(parts[2:])

    if name.startswith('input'):
        idx = int(name[5:])
        return f'neck.input_blocks.{idx}.{rest}'
    elif name.startswith('up'):
        idx = int(name[2:])
        # Resamplers have conv1 (up) and conv2 (conv)
        if 'up.' in dst_key:
            return f'neck.resamplers.{idx}.conv1.{rest}'
        elif 'conv.' in dst_key:
            return f'neck.resamplers.{idx}.conv2.{rest}'
    elif name.startswith('res'):
        # res1_0 -> level 1, block 0
        level = int(name[3])
        block = int(name[5])
        # Map conv1/conv2 to layers in Sequential
        if 'conv1' in rest:
            layer_idx = 2  # norm, act, conv
            return f'neck.res_blocks.{level}.{block}.layers.{layer_idx}.{rest.replace("conv1.", "")}'
        elif 'conv2' in rest:
            layer_idx = 5
            return f'neck.res_blocks.{level}.{block}.layers.{layer_idx}.{rest.replace("conv2.", "")}'

    return None


def _map_head_key(dst_key: str, head_name: str) -> Optional[str]:
    """Map new head key to original ConvStack key."""
    parts = dst_key.split('.')
    name = parts[1]
    rest = '.'.join(parts[2:])

    if name.startswith('input'):
        idx = int(name[5:])
        return f'{head_name}.input_blocks.{idx}.{rest}'
    elif name.startswith('up'):
        idx = int(name[2:])
        if 'up.' in dst_key:
            return f'{head_name}.resamplers.{idx}.conv1.{rest}'
        elif 'conv.' in dst_key:
            return f'{head_name}.resamplers.{idx}.conv2.{rest}'
    elif name.startswith('res'):
        level = int(name[3])
        if 'conv1' in rest:
            return f'{head_name}.res_blocks.{level}.0.layers.2.{rest.replace("conv1.", "")}'
        elif 'conv2' in rest:
            return f'{head_name}.res_blocks.{level}.0.layers.5.{rest.replace("conv2.", "")}'
    elif name == 'output':
        return f'{head_name}.output_blocks.4.{rest}'

    return None
