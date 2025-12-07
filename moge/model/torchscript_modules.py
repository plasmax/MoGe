"""
TorchScript-compatible modules for MoGe model.

These modules are re-implementations of the original modules that are fully
compatible with torch.jit.script().

Key changes from original:
- No isinstance() checks
- No dynamic imports
- No external libraries (xformers)
- All type annotations
- All class attributes declared in __init__
- Fixed tensor shapes with explicit typing
"""

from typing import Tuple, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# TorchScript-compatible DINOv2 Layers
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample - TorchScript compatible."""

    drop_prob: float
    scale_by_keep: bool

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    """Layer scale module - TorchScript compatible."""

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """MLP as used in Vision Transformer - TorchScript compatible."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = 'gelu',
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        if act_layer == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLUFFNFused(nn.Module):
    """SwiGLU FFN - TorchScript compatible."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = 'silu',
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        # Round to multiple of 256 for efficiency
        hidden_features = (hidden_features + 255) // 256 * 256

        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1 = x12[..., : x12.shape[-1] // 2]
        x2 = x12[..., x12.shape[-1] // 2 :]
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding - TorchScript compatible."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: bool = False,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten_embedding = flatten_embedding
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, C, H/P, W/P)
        if self.flatten_embedding:
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head attention using PyTorch native SDPA - TorchScript compatible."""

    num_heads: int
    scale: float

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, C // H)

        # Use PyTorch native scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block - TorchScript compatible."""

    sample_drop_ratio: float

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        ffn_layer_type: str = 'mlp',  # 'mlp' or 'swiglu'
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if init_values is not None and init_values > 0:
            self.ls1 = LayerScale(dim, init_values=init_values)
        else:
            self.ls1 = nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if ffn_layer_type == 'swiglu':
            self.mlp = SwiGLUFFNFused(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                bias=ffn_bias,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=drop,
                bias=ffn_bias,
            )

        if init_values is not None and init_values > 0:
            self.ls2 = LayerScale(dim, init_values=init_values)
        else:
            self.ls2 = nn.Identity()

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        # Simple forward without stochastic depth complexity (inference mode)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DinoVisionTransformerTS(nn.Module):
    """
    DINOv2 Vision Transformer - TorchScript compatible version.

    This is a simplified version that:
    - Uses only the TorchScript-compatible modules above
    - Removes chunked blocks (not needed for inference)
    - Fixed intermediate layer extraction
    """

    num_features: int
    embed_dim: int
    num_tokens: int
    n_blocks: int
    num_heads: int
    patch_size: int
    num_register_tokens: int
    interpolate_antialias: bool
    interpolate_offset: float

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = None,
        ffn_layer_type: str = 'mlp',
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ):
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = nn.Parameter(torch.zeros(1, 0, embed_dim))

        # Linear drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                init_values=init_values,
                ffn_layer_type=ffn_layer_type,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if num_register_tokens > 0:
            nn.init.normal_(self.register_tokens, std=1e-6)

    def interpolate_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        """Interpolate position embeddings to match input size."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, C)
        patch_pos_embed = pos_embed[:, 1:, :]   # (1, N, C)
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        M = int(math.sqrt(float(N)))

        # Reshape and interpolate
        patch_pos_embed = patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h0, w0),
            mode='bicubic',
            align_corners=False,
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h0 * w0, dim)

        return torch.cat([class_pos_embed, patch_pos_embed], dim=1).to(previous_dtype)

    def prepare_tokens(self, x: Tensor) -> Tensor:
        """Prepare tokens with position encoding."""
        B, nc, h, w = x.shape
        x = self.patch_embed(x)

        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, h, w)

        if self.num_register_tokens > 0:
            x = torch.cat([
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ], dim=1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning class token output."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int = 1,
        return_class_token: bool = False
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Get intermediate layer outputs.

        Args:
            x: Input tensor (B, C, H, W)
            n: Number of last layers to return
            return_class_token: If True, returns (features, class_token) tuples

        Returns:
            List of (features, class_token) tuples for each requested layer
        """
        x = self.prepare_tokens(x)

        outputs: List[Tuple[Tensor, Tensor]] = []
        total_blocks = len(self.blocks)
        start_idx = total_blocks - n

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= start_idx:
                x_norm = self.norm(x)
                features = x_norm[:, self.num_register_tokens + 1:]  # Skip cls and register tokens
                class_token = x_norm[:, 0]
                outputs.append((features, class_token))

        return outputs


# ============================================================================
# TorchScript-compatible Modules for MoGe
# ============================================================================

class ResidualConvBlockTS(nn.Module):
    """Residual Conv Block - TorchScript compatible."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding_mode: str = 'replicate',
        activation: str = 'relu',
        in_norm: str = 'layer_norm',
        hidden_norm: str = 'group_norm',
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        # Build activation
        if activation == 'relu':
            act = nn.ReLU(inplace=False)
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        elif activation == 'silu':
            act = nn.SiLU(inplace=False)
        elif activation == 'elu':
            act = nn.ELU(inplace=False)
        else:
            act = nn.ReLU(inplace=False)

        # Build normalization layers
        if in_norm == 'group_norm':
            norm1 = nn.GroupNorm(in_channels // 32, in_channels)
        elif in_norm == 'layer_norm':
            norm1 = nn.GroupNorm(1, in_channels)
        elif in_norm == 'instance_norm':
            norm1 = nn.InstanceNorm2d(in_channels)
        else:
            norm1 = nn.Identity()

        if hidden_norm == 'group_norm':
            norm2 = nn.GroupNorm(hidden_channels // 32, hidden_channels)
        elif hidden_norm == 'layer_norm':
            norm2 = nn.GroupNorm(1, hidden_channels)
        elif hidden_norm == 'instance_norm':
            norm2 = nn.InstanceNorm2d(hidden_channels)
        else:
            norm2 = nn.Identity()

        self.layers = nn.Sequential(
            norm1,
            act,
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
            norm2,
            act,
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode)
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class ResamplerTS(nn.Module):
    """Resampler module - TorchScript compatible."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: str,  # 'pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose'
        scale_factor: int = 2,
    ):
        super().__init__()
        self.type_ = type_
        self.scale_factor = scale_factor

        if type_ == 'pixel_shuffle':
            self.conv1 = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            self.shuffle = nn.PixelShuffle(scale_factor)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        elif type_ == 'conv_transpose':
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
            self.shuffle = nn.Identity()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        else:  # nearest or bilinear
            self.conv1 = nn.Identity()
            self.shuffle = nn.Identity()
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')

    def forward(self, x: Tensor) -> Tensor:
        if self.type_ == 'pixel_shuffle':
            x = self.conv1(x)
            x = self.shuffle(x)
            x = self.conv2(x)
        elif self.type_ == 'conv_transpose':
            x = self.conv1(x)
            x = self.conv2(x)
        elif self.type_ == 'nearest':
            x = F.interpolate(x, scale_factor=float(self.scale_factor), mode='nearest')
            x = self.conv2(x)
        elif self.type_ == 'bilinear':
            x = F.interpolate(x, scale_factor=float(self.scale_factor), mode='bilinear', align_corners=False)
            x = self.conv2(x)
        else:
            x = F.interpolate(x, scale_factor=float(self.scale_factor), mode='nearest')
            x = self.conv2(x)
        return x


class ConvStackTS(nn.Module):
    """ConvStack - TorchScript compatible version."""

    def __init__(
        self,
        dim_in: List[int],
        dim_res_blocks: List[int],
        dim_out: List[int],
        resamplers: List[str],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: List[int] = [1, 1, 1, 1, 1],
        res_block_in_norm: str = 'layer_norm',
        res_block_hidden_norm: str = 'group_norm',
        activation: str = 'relu',
    ):
        super().__init__()
        num_levels = len(dim_res_blocks)

        # Input blocks
        input_blocks: List[nn.Module] = []
        for i in range(num_levels):
            if dim_in[i] > 0:
                input_blocks.append(nn.Conv2d(dim_in[i], dim_res_blocks[i], kernel_size=1, stride=1, padding=0))
            else:
                input_blocks.append(nn.Identity())
        self.input_blocks = nn.ModuleList(input_blocks)

        # Resamplers
        resampler_modules: List[nn.Module] = []
        for i in range(num_levels - 1):
            resampler_modules.append(ResamplerTS(dim_res_blocks[i], dim_res_blocks[i + 1], scale_factor=2, type_=resamplers[i]))
        self.resamplers = nn.ModuleList(resampler_modules)

        # Residual blocks
        res_block_modules: List[nn.Module] = []
        for i in range(num_levels):
            blocks: List[nn.Module] = []
            for j in range(num_res_blocks[i]):
                blocks.append(ResidualConvBlockTS(
                    dim_res_blocks[i],
                    dim_res_blocks[i],
                    dim_times_res_block_hidden * dim_res_blocks[i],
                    activation=activation,
                    in_norm=res_block_in_norm,
                    hidden_norm=res_block_hidden_norm
                ))
            res_block_modules.append(nn.Sequential(*blocks))
        self.res_blocks = nn.ModuleList(res_block_modules)

        # Output blocks
        output_blocks: List[nn.Module] = []
        for i in range(num_levels):
            if dim_out[i] > 0:
                output_blocks.append(nn.Conv2d(dim_res_blocks[i], dim_out[i], kernel_size=1, stride=1, padding=0))
            else:
                output_blocks.append(nn.Identity())
        self.output_blocks = nn.ModuleList(output_blocks)

        self.num_levels = num_levels

    def forward(self, in_features: List[Tensor]) -> List[Tensor]:
        out_features: List[Tensor] = []
        x = torch.zeros(1, device=in_features[0].device, dtype=in_features[0].dtype)  # placeholder

        for i in range(self.num_levels):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            else:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < self.num_levels - 1:
                x = self.resamplers[i](x)

        return out_features


class MLPTS(nn.Module):
    """Simple MLP - TorchScript compatible."""

    def __init__(self, dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DINOv2EncoderTS(nn.Module):
    """
    DINOv2 Encoder - TorchScript compatible.

    Wraps the ViT backbone and handles image preprocessing.
    """

    dim_features: int
    num_features: int
    intermediate_layers_n: int

    def __init__(
        self,
        backbone: str,
        intermediate_layers: int,  # Can be int or list, but we convert to int
        dim_out: int,
    ):
        super().__init__()
        # Handle intermediate_layers being either int or list
        if isinstance(intermediate_layers, list):
            self.intermediate_layers_n = len(intermediate_layers)
        else:
            self.intermediate_layers_n = intermediate_layers
        self.num_features = self.intermediate_layers_n

        # Create backbone based on name
        if 'vit_small' in backbone or 'vitb14' in backbone.lower():
            embed_dim = 384 if 'small' in backbone else 768
            depth = 12
            num_heads = 6 if 'small' in backbone else 12
            ffn_type = 'mlp'
            num_register_tokens = 4 if 'reg' in backbone else 0
        elif 'vit_large' in backbone or 'vitl14' in backbone.lower():
            embed_dim = 1024
            depth = 24
            num_heads = 16
            ffn_type = 'mlp'
            num_register_tokens = 4 if 'reg' in backbone else 0
        elif 'vit_giant' in backbone or 'vitg14' in backbone.lower():
            embed_dim = 1536
            depth = 40
            num_heads = 24
            ffn_type = 'swiglu'
            num_register_tokens = 4 if 'reg' in backbone else 0
        else:
            # Default to ViT-B
            embed_dim = 768
            depth = 12
            num_heads = 12
            ffn_type = 'mlp'
            num_register_tokens = 0

        self.backbone = DinoVisionTransformerTS(
            img_size=518,  # Standard DINOv2 size
            patch_size=14,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            ffn_layer_type=ffn_type,
            num_register_tokens=num_register_tokens,
        )

        self.dim_features = embed_dim

        # Output projections
        projections: List[nn.Module] = []
        for _ in range(self.intermediate_layers_n):
            projections.append(nn.Conv2d(embed_dim, dim_out, kernel_size=1, stride=1, padding=0))
        self.output_projections = nn.ModuleList(projections)

        # Image normalization
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        image: Tensor,
        token_rows: int,
        token_cols: int,
        return_class_token: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            image: (B, 3, H, W) input image in [0, 1]
            token_rows: number of patch rows
            token_cols: number of patch cols
            return_class_token: whether to return class token

        Returns:
            features: (B, C, H, W) projected features
            cls_token: (B, C) class token (if return_class_token=True)
        """
        # Resize to patch grid size
        image_14 = F.interpolate(
            image,
            size=(token_rows * 14, token_cols * 14),
            mode='bilinear',
            align_corners=False
        )
        # Normalize
        image_14 = (image_14 - self.image_mean) / self.image_std

        # Get intermediate layers
        features_list = self.backbone.get_intermediate_layers(
            image_14,
            n=self.intermediate_layers_n,
            return_class_token=True
        )

        # Project and sum features
        batch_size = image.shape[0]
        x_sum: Optional[Tensor] = None
        cls_token_out: Tensor = torch.zeros(1)  # placeholder

        for i, (feat, clstoken) in enumerate(features_list):
            # feat: (B, N, C) -> reshape to (B, C, H, W)
            feat_2d = feat.permute(0, 2, 1).reshape(batch_size, -1, token_rows, token_cols).contiguous()
            proj_feat = self.output_projections[i](feat_2d)
            if x_sum is None:
                x_sum = proj_feat
            else:
                x_sum = x_sum + proj_feat
            cls_token_out = clstoken

        if x_sum is None:
            x_sum = torch.zeros(batch_size, 1, token_rows, token_cols, device=image.device, dtype=image.dtype)

        return x_sum, cls_token_out
