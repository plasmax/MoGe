"""
TorchScript-compatible wrapper for MoGe model.

This module provides a TorchScript-compatible wrapper that can be exported to
a .pt file for use in Nuke's Inference node via CatFileCreator.

Requirements for TorchScript/Nuke compatibility:
- No external libraries (numpy, utils3d, etc.)
- All type annotations
- All attributes declared in __init__
- No inheritance (use composition)
- Handle device/dtype dynamically from input tensor
- Input: (1, 3, H, W) tensor, Output: (1, outChan, H, W) tensor
"""

from typing import Tuple, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Pure PyTorch geometry utilities (no numpy/utils3d dependencies)
# ============================================================================

def normalized_view_plane_uv_ts(
    width: int,
    height: int,
    aspect_ratio: float,
    dtype: torch.dtype,
    device: torch.device
) -> Tensor:
    """
    UV coordinates with left-top corner as (-width/diagonal, -height/diagonal)
    and right-bottom corner as (width/diagonal, height/diagonal).

    TorchScript compatible version.
    """
    span_x = aspect_ratio / (1.0 + aspect_ratio ** 2) ** 0.5
    span_y = 1.0 / (1.0 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device
    )
    # meshgrid with indexing='xy'
    u_grid = u.unsqueeze(0).expand(height, -1)
    v_grid = v.unsqueeze(1).expand(-1, width)
    uv = torch.stack([u_grid, v_grid], dim=-1)
    return uv


def intrinsics_from_focal_center_ts(
    fx: Tensor,
    fy: Tensor,
    cx: Tensor,
    cy: Tensor
) -> Tensor:
    """
    Build intrinsics matrix from focal lengths and principal point.

    TorchScript compatible version of utils3d.pt.intrinsics_from_focal_center.
    """
    batch_shape = fx.shape
    device = fx.device
    dtype = fx.dtype

    # Create 3x3 identity-like tensor
    intrinsics = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    intrinsics[..., 2, 2] = 1.0

    return intrinsics


def depth_map_to_point_map_ts(
    depth: Tensor,
    intrinsics: Tensor
) -> Tensor:
    """
    Convert depth map to point map using camera intrinsics.

    TorchScript compatible version of utils3d.pt.depth_map_to_point_map.

    Args:
        depth: (B, H, W) depth values
        intrinsics: (B, 3, 3) normalized intrinsics matrix

    Returns:
        points: (B, H, W, 3) camera-space point map
    """
    batch_size = depth.shape[0]
    height = depth.shape[1]
    width = depth.shape[2]
    device = depth.device
    dtype = depth.dtype

    # Create UV grid (0 to 1)
    u = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype)
    v = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype)
    u_grid = u.unsqueeze(0).expand(height, -1)  # (H, W)
    v_grid = v.unsqueeze(1).expand(-1, width)   # (H, W)

    # Stack to get (H, W, 2) and add batch dim -> (1, H, W, 2)
    uv = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(0)

    # Extract intrinsics components
    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]  # (B,)
    cx = intrinsics[:, 0, 2]  # (B,)
    cy = intrinsics[:, 1, 2]  # (B,)

    # Compute (u - cx) / fx and (v - cy) / fy
    # Shape: (B, 1, 1) for broadcasting
    fx_inv = (1.0 / fx).view(batch_size, 1, 1)
    fy_inv = (1.0 / fy).view(batch_size, 1, 1)
    cx_b = cx.view(batch_size, 1, 1)
    cy_b = cy.view(batch_size, 1, 1)

    x = (uv[..., 0] - cx_b) * fx_inv * depth  # (B, H, W)
    y = (uv[..., 1] - cy_b) * fy_inv * depth  # (B, H, W)
    z = depth  # (B, H, W)

    points = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)
    return points


def recover_focal_shift_ts(
    points: Tensor,
    mask: Optional[Tensor],
    focal: Optional[Tensor] = None,
    downsample_size: Tuple[int, int] = (64, 64)
) -> Tuple[Tensor, Tensor]:
    """
    Recover the focal length and Z-shift from a point map.
    Pure PyTorch implementation for TorchScript compatibility.

    Args:
        points: (B, H, W, 3) point map
        mask: (B, H, W) binary mask or None
        focal: (B,) known focal or None to estimate
        downsample_size: size for downsampled computation

    Returns:
        focal: (B,) estimated focal length
        shift: (B,) Z-axis shift
    """
    batch_size = points.shape[0]
    height = points.shape[1]
    width = points.shape[2]
    device = points.device
    dtype = points.dtype

    aspect_ratio = float(width) / float(height)

    # Compute normalized UV
    uv = normalized_view_plane_uv_ts(width, height, aspect_ratio, dtype, device)  # (H, W, 2)

    # Downsample points and uv
    # Reshape points from (B, H, W, 3) to (B, 3, H, W) for interpolation
    points_bhwc = points
    points_bchw = points_bhwc.permute(0, 3, 1, 2)  # (B, 3, H, W)
    points_lr = F.interpolate(
        points_bchw,
        size=(downsample_size[0], downsample_size[1]),
        mode='nearest'
    )  # (B, 3, h, w)
    points_lr = points_lr.permute(0, 2, 3, 1)  # (B, h, w, 3)

    # Downsample UV
    uv_hw2 = uv.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 2, H, W)
    uv_lr = F.interpolate(
        uv_hw2,
        size=(downsample_size[0], downsample_size[1]),
        mode='nearest'
    )  # (1, 2, h, w)
    uv_lr = uv_lr.squeeze(0).permute(1, 2, 0)  # (h, w, 2)

    # Handle mask
    if mask is not None:
        mask_bhw = mask.unsqueeze(1).float()  # (B, 1, H, W)
        mask_lr = F.interpolate(
            mask_bhw,
            size=(downsample_size[0], downsample_size[1]),
            mode='nearest'
        )  # (B, 1, h, w)
        mask_lr = mask_lr.squeeze(1) > 0.5  # (B, h, w)
    else:
        mask_lr = torch.ones(
            batch_size, downsample_size[0], downsample_size[1],
            dtype=torch.bool, device=device
        )

    # Solve for focal and shift using least squares (vectorized)
    # The problem: focal * xy / (z + shift) = uv
    # Rearranged: focal * xy = uv * z + uv * shift
    # Let's solve: |focal * xy - uv * (z + shift)|^2 -> min

    out_focal = torch.zeros(batch_size, device=device, dtype=dtype)
    out_shift = torch.zeros(batch_size, device=device, dtype=dtype)

    for b in range(batch_size):
        m = mask_lr[b]  # (h, w)
        pts = points_lr[b][m]  # (N, 3)
        xy = pts[:, :2]  # (N, 2)
        z = pts[:, 2]  # (N,)
        uv_masked = uv_lr[m]  # (N, 2)

        n_points = xy.shape[0]
        if n_points < 10:
            out_focal[b] = 1.0
            out_shift[b] = 0.0
            continue

        if focal is not None:
            # Focal is known, only solve for shift
            f = focal[b]
            # focal * xy / (z + shift) = uv
            # focal * xy = uv * z + uv * shift
            # uv * shift = focal * xy - uv * z
            # A * shift = b where A = uv, b = focal * xy - uv * z
            # shift = sum(uv * (focal * xy - uv * z)) / sum(uv * uv)

            # Actually, let's solve: min |f * xy - uv * (z + s)|^2
            # d/ds = -2 * sum(uv * (f * xy - uv * (z + s))) = 0
            # sum(uv * f * xy) = sum(uv * uv * (z + s))
            # sum(uv * f * xy) = sum(uv * uv * z) + s * sum(uv * uv)
            # s = (sum(uv * f * xy) - sum(uv * uv * z)) / sum(uv * uv)

            uv_sq_sum = (uv_masked * uv_masked).sum()
            if uv_sq_sum < 1e-8:
                out_focal[b] = f
                out_shift[b] = 0.0
            else:
                numerator = (uv_masked * f * xy).sum() - (uv_masked * uv_masked * z.unsqueeze(-1)).sum()
                s = numerator / uv_sq_sum
                out_focal[b] = f
                out_shift[b] = s
        else:
            # Solve for both focal and shift
            # min |f * xy - uv * (z + s)|^2
            # Variables: f, s
            # d/df = sum(xy * (f * xy - uv * (z + s))) = 0
            # d/ds = sum(-uv * (f * xy - uv * (z + s))) = 0

            # f * sum(xy^2) - sum(xy * uv * z) - s * sum(xy * uv) = 0  ... (1)
            # -f * sum(xy * uv) + sum(uv^2 * z) + s * sum(uv^2) = 0     ... (2)

            # Let:
            # a = sum(xy^2), b = sum(xy * uv * z), c = sum(xy * uv)
            # d = sum(uv^2 * z), e = sum(uv^2)
            #
            # f * a - c * s = b   ... (1)
            # -f * c + e * s = -d  ... (2)
            #
            # From (1): f = (b + c*s) / a
            # Sub into (2): -(b + c*s) * c / a + e * s = -d
            # -b*c/a - c^2*s/a + e*s = -d
            # s * (e - c^2/a) = b*c/a - d
            # s = (b*c/a - d) / (e - c^2/a)
            # s = (b*c - d*a) / (e*a - c^2)

            xy_sq = (xy * xy).sum()  # a
            xy_uv_z = (xy * uv_masked * z.unsqueeze(-1)).sum()  # b
            xy_uv = (xy * uv_masked).sum()  # c
            uv_sq_z = (uv_masked * uv_masked * z.unsqueeze(-1)).sum()  # d
            uv_sq = (uv_masked * uv_masked).sum()  # e

            denom = uv_sq * xy_sq - xy_uv * xy_uv
            if torch.abs(denom) < 1e-8:
                out_focal[b] = 1.0
                out_shift[b] = 0.0
            else:
                s = (xy_uv_z * xy_uv - uv_sq_z * xy_sq) / denom
                f = (xy_uv_z + xy_uv * s) / (xy_sq + 1e-8)

                # Ensure positive focal
                if f <= 0:
                    f = torch.tensor(1.0, device=device, dtype=dtype)

                out_focal[b] = f
                out_shift[b] = s

    return out_focal, out_shift


# ============================================================================
# TorchScript-compatible MoGe Wrapper for Nuke
# ============================================================================

class MoGeForNuke(nn.Module):
    """
    TorchScript-compatible wrapper for MoGe model for use in Nuke.

    This wrapper:
    - Takes a single RGB input tensor (1, 3, H, W)
    - Returns depth map as (1, 1, H, W) by default
    - Can return multiple outputs concatenated along channel dimension
    - Handles device and dtype dynamically from input

    Custom knobs can be added as __init__ parameters following Nuke conventions.
    """

    # Declare all buffer/attribute types for TorchScript
    image_mean: Tensor
    image_std: Tensor
    remap_output: str
    num_tokens_min: int
    num_tokens_max: int
    resolution_level: int
    output_mode: int  # 0=depth, 1=points, 2=depth+mask

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        remap_output: str = 'linear',
        num_tokens_range: Tuple[int, int] = (1200, 2500),
        resolution_level: int = 9,
        output_mode: int = 0
    ):
        """
        Initialize the TorchScript-compatible MoGe wrapper.

        Args:
            encoder: The backbone encoder module
            head: The prediction head module
            remap_output: Output remapping mode ('linear', 'sinh', 'exp', 'sinh_exp')
            num_tokens_range: (min, max) tokens for resolution control
            resolution_level: Integer knob (0-9) for resolution control
            output_mode: 0=depth only, 1=points (3ch), 2=depth+mask (2ch)
        """
        super(MoGeForNuke, self).__init__()

        # Store modules
        self.encoder = encoder
        self.head = head

        # Store configuration as instance attributes (declared above)
        self.remap_output = remap_output
        self.num_tokens_min = num_tokens_range[0]
        self.num_tokens_max = num_tokens_range[1]
        self.resolution_level = resolution_level
        self.output_mode = output_mode

        # Register normalization buffers
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # Set eval mode
        self.eval()

    def _remap_points(self, points: Tensor) -> Tensor:
        """Apply output remapping transformation."""
        if self.remap_output == 'linear':
            return points
        elif self.remap_output == 'sinh':
            return torch.sinh(points)
        elif self.remap_output == 'exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            z_exp = torch.exp(z)
            return torch.cat([xy * z_exp, z_exp], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            return points

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for Nuke inference.

        Args:
            input: RGB image tensor of shape (1, 3, H, W) in range [0, 1]

        Returns:
            Output tensor of shape (1, C, H, W) where C depends on output_mode:
            - output_mode=0: (1, 1, H, W) depth only
            - output_mode=1: (1, 3, H, W) points (x, y, z)
            - output_mode=2: (1, 2, H, W) depth and mask
        """
        device = input.device
        dtype = input.dtype

        # Clamp input to valid range
        input = torch.clamp(input, min=0.0, max=1.0)

        batch_size = input.shape[0]
        original_height = input.shape[2]
        original_width = input.shape[3]
        aspect_ratio = float(original_width) / float(original_height)

        # Calculate number of tokens based on resolution_level
        num_tokens = self.num_tokens_min + (self.resolution_level * (self.num_tokens_max - self.num_tokens_min)) // 9

        # Resize to resolution based on num_tokens
        resize_factor = ((num_tokens * 14 * 14) / (original_height * original_width)) ** 0.5
        resized_height = int(original_height * resize_factor)
        resized_width = int(original_width * resize_factor)

        image = F.interpolate(
            input,
            size=(resized_height, resized_width),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )

        # Normalize for encoder
        image_mean = self.image_mean.to(dtype=dtype, device=device)
        image_std = self.image_std.to(dtype=dtype, device=device)
        image = (image - image_mean) / image_std

        # Pad to multiple of 14
        pad_h = (14 - resized_height % 14) % 14
        pad_w = (14 - resized_width % 14) % 14
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')

        padded_height = image.shape[2]
        padded_width = image.shape[3]
        patch_h = padded_height // 14
        patch_w = padded_width // 14

        # Get features from encoder
        features = self.encoder.get_intermediate_layers(image, n=4, return_class_token=True)

        # Run head
        output = self.head(features, image)
        points_raw, mask_raw = output[0], output[1]

        # Resize to original resolution
        points = F.interpolate(
            points_raw,
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        )
        mask = F.interpolate(
            mask_raw,
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        )

        # Post-process
        # points is (B, 3, H, W), convert to (B, H, W, 3)
        points = points.permute(0, 2, 3, 1)
        points = self._remap_points(points)
        mask = mask.squeeze(1).sigmoid()  # (B, H, W)

        # Recover focal and shift
        mask_binary = mask > 0.5
        focal, shift = recover_focal_shift_ts(points, mask_binary)

        # Compute depth
        depth = points[..., 2] + shift.view(batch_size, 1, 1)

        # Compute intrinsics and reproject
        fx = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5
        cx = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        cy = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        intrinsics = intrinsics_from_focal_center_ts(fx, fy, cx, cy)

        # Recompute points from depth for projection consistency
        points = depth_map_to_point_map_ts(depth, intrinsics)

        # Apply mask
        depth = torch.where(mask_binary, depth, torch.tensor(float('inf'), device=device, dtype=dtype))

        # Format output based on mode
        if self.output_mode == 0:
            # Depth only
            output_tensor = depth.unsqueeze(1)  # (B, 1, H, W)
        elif self.output_mode == 1:
            # Points (x, y, z)
            output_tensor = points.permute(0, 3, 1, 2)  # (B, 3, H, W)
        else:
            # Depth + mask
            output_tensor = torch.stack([depth, mask], dim=1)  # (B, 2, H, W)

        return output_tensor


class MoGeForNukeV2(nn.Module):
    """
    TorchScript-compatible wrapper for MoGe V2 model for use in Nuke.

    This is specifically designed for the V2 architecture which has a different
    structure (DINOv2Encoder + ConvStack neck + heads).
    """

    image_mean: Tensor
    image_std: Tensor
    remap_output: str
    num_tokens_min: int
    num_tokens_max: int
    resolution_level: int
    output_mode: int
    mask_threshold: float

    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        points_head: nn.Module,
        mask_head: nn.Module,
        remap_output: str = 'linear',
        num_tokens_range: Tuple[int, int] = (1200, 3600),
        resolution_level: int = 9,
        output_mode: int = 0,
        mask_threshold: float = 0.5
    ):
        """
        Initialize the TorchScript-compatible MoGe V2 wrapper.

        Args:
            encoder: DINOv2Encoder module
            neck: ConvStack neck module
            points_head: Points prediction head
            mask_head: Mask prediction head
            remap_output: Output remapping mode
            num_tokens_range: (min, max) tokens for resolution
            resolution_level: Integer knob (0-9) for resolution
            output_mode: 0=depth, 1=points, 2=depth+mask
            mask_threshold: Threshold for binary mask
        """
        super(MoGeForNukeV2, self).__init__()

        self.encoder = encoder
        self.neck = neck
        self.points_head = points_head
        self.mask_head = mask_head

        self.remap_output = remap_output
        self.num_tokens_min = num_tokens_range[0]
        self.num_tokens_max = num_tokens_range[1]
        self.resolution_level = resolution_level
        self.output_mode = output_mode
        self.mask_threshold = mask_threshold

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.eval()

    def _remap_points(self, points: Tensor) -> Tensor:
        """Apply output remapping transformation."""
        if self.remap_output == 'linear':
            return points
        elif self.remap_output == 'sinh':
            return torch.sinh(points)
        elif self.remap_output == 'exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            z_exp = torch.exp(z)
            return torch.cat([xy * z_exp, z_exp], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            return points

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for Nuke inference.

        Args:
            input: RGB image tensor of shape (1, 3, H, W) in range [0, 1]

        Returns:
            Output tensor of shape (1, C, H, W)
        """
        device = input.device
        dtype = input.dtype

        input = torch.clamp(input, min=0.0, max=1.0)

        batch_size = input.shape[0]
        img_h = input.shape[2]
        img_w = input.shape[3]
        aspect_ratio = float(img_w) / float(img_h)

        # Calculate tokens
        num_tokens = self.num_tokens_min + (self.resolution_level * (self.num_tokens_max - self.num_tokens_min)) // 9

        # Calculate base dimensions
        base_h = int(round((float(num_tokens) / aspect_ratio) ** 0.5))
        base_w = int(round((float(num_tokens) * aspect_ratio) ** 0.5))

        # Encoder forward
        features, cls_token = self.encoder(input, base_h, base_w, return_class_token=True)

        # Build feature pyramid with UV coordinates
        feature_list: List[Tensor] = [features]
        for level in range(1, 5):
            h_level = base_h * (2 ** level)
            w_level = base_w * (2 ** level)
            uv = normalized_view_plane_uv_ts(w_level, h_level, aspect_ratio, dtype, device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            feature_list.append(uv)

        # Add UV to first feature
        uv0 = normalized_view_plane_uv_ts(base_w, base_h, aspect_ratio, dtype, device)
        uv0 = uv0.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        feature_list[0] = torch.cat([feature_list[0], uv0], dim=1)

        # Neck
        neck_features = self.neck(feature_list)

        # Heads
        points_out = self.points_head(neck_features)[-1]  # Get last level
        mask_out = self.mask_head(neck_features)[-1]

        # Resize to original
        points = F.interpolate(points_out, size=(img_h, img_w), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask_out, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # Post-process
        points = points.permute(0, 2, 3, 1)  # (B, H, W, 3)
        points = self._remap_points(points)
        mask = mask.squeeze(1).sigmoid()  # (B, H, W)

        mask_binary = mask > self.mask_threshold

        # Recover focal and shift
        focal, shift = recover_focal_shift_ts(points, mask_binary)

        # Compute depth
        depth = points[..., 2] + shift.view(batch_size, 1, 1)
        mask_binary = mask_binary & (depth > 0)

        # Compute intrinsics
        fx = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5
        cx = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        cy = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        intrinsics = intrinsics_from_focal_center_ts(fx, fy, cx, cy)

        # Reproject
        points = depth_map_to_point_map_ts(depth, intrinsics)

        # Apply mask
        depth = torch.where(mask_binary, depth, torch.tensor(float('inf'), device=device, dtype=dtype))

        # Format output
        if self.output_mode == 0:
            output_tensor = depth.unsqueeze(1)
        elif self.output_mode == 1:
            output_tensor = points.permute(0, 3, 1, 2)
        else:
            output_tensor = torch.stack([depth, mask], dim=1)

        return output_tensor


# ============================================================================
# Self-contained simple wrapper (for direct weight loading)
# ============================================================================

class MoGeForNukeSimple(nn.Module):
    """
    Self-contained TorchScript-compatible MoGe wrapper.

    This class includes all necessary components inline without external dependencies.
    It is designed to be directly scriptable with torch.jit.script().
    """

    image_mean: Tensor
    image_std: Tensor
    remap_output: str
    num_tokens_min: int
    num_tokens_max: int
    resolution_level: int
    output_mode: int
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        remap_output: str = 'linear',
        num_tokens_range: Tuple[int, int] = (1200, 2500),
        resolution_level: int = 9,
        output_mode: int = 0,
    ):
        """
        Initialize self-contained MoGe wrapper.

        Args:
            embed_dim: Embedding dimension (768 for ViT-B, 1024 for ViT-L, 1536 for ViT-G)
            depth: Number of transformer blocks (12 for B, 24 for L, 40 for G)
            num_heads: Number of attention heads
            remap_output: Output remapping mode
            num_tokens_range: (min, max) tokens for resolution
            resolution_level: Integer knob (0-9) for resolution
            output_mode: 0=depth, 1=points, 2=depth+mask
        """
        super(MoGeForNukeSimple, self).__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = 14
        self.remap_output = remap_output
        self.num_tokens_min = num_tokens_range[0]
        self.num_tokens_max = num_tokens_range[1]
        self.resolution_level = resolution_level
        self.output_mode = output_mode

        # Build encoder (simplified DINOv2-style ViT)
        self.encoder = self._build_encoder(embed_dim, depth, num_heads)

        # Build head (simplified prediction head)
        self.head = self._build_head(embed_dim)

        # Normalization buffers
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.eval()

    def _build_encoder(self, embed_dim: int, depth: int, num_heads: int) -> nn.Module:
        """Build the encoder backbone."""
        from .torchscript_modules import DinoVisionTransformerTS
        return DinoVisionTransformerTS(
            img_size=518,
            patch_size=14,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            ffn_layer_type='mlp' if embed_dim < 1536 else 'swiglu',
        )

    def _build_head(self, embed_dim: int) -> nn.Module:
        """Build the prediction head."""
        # Simplified head: project features and predict points + mask
        return nn.ModuleDict({
            'proj': nn.Conv2d(embed_dim, 256, kernel_size=1),
            'upsample1': nn.Sequential(
                nn.ConvTranspose2d(258, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
            ),
            'upsample2': nn.Sequential(
                nn.ConvTranspose2d(130, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ),
            'upsample3': nn.Sequential(
                nn.ConvTranspose2d(66, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
            ),
            'points_out': nn.Conv2d(34, 3, kernel_size=1),
            'mask_out': nn.Conv2d(34, 1, kernel_size=1),
        })

    def _remap_points(self, points: Tensor) -> Tensor:
        """Apply output remapping transformation."""
        if self.remap_output == 'linear':
            return points
        elif self.remap_output == 'sinh':
            return torch.sinh(points)
        elif self.remap_output == 'exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            z_exp = torch.exp(z)
            return torch.cat([xy * z_exp, z_exp], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            return points

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for Nuke inference.

        Args:
            input: RGB image tensor of shape (1, 3, H, W) in range [0, 1]

        Returns:
            Output tensor of shape (1, C, H, W)
        """
        device = input.device
        dtype = input.dtype

        # Clamp input
        input = torch.clamp(input, min=0.0, max=1.0)

        batch_size = input.shape[0]
        img_h = input.shape[2]
        img_w = input.shape[3]
        aspect_ratio = float(img_w) / float(img_h)

        # Calculate tokens
        num_tokens = self.num_tokens_min + (self.resolution_level * (self.num_tokens_max - self.num_tokens_min)) // 9

        # Calculate patch grid size
        base_h = int(round((float(num_tokens) / aspect_ratio) ** 0.5))
        base_w = int(round((float(num_tokens) * aspect_ratio) ** 0.5))

        # Resize for encoder
        enc_h = base_h * self.patch_size
        enc_w = base_w * self.patch_size
        image = F.interpolate(input, size=(enc_h, enc_w), mode='bilinear', align_corners=False)

        # Normalize
        image = (image - self.image_mean.to(device=device, dtype=dtype)) / self.image_std.to(device=device, dtype=dtype)

        # Encoder forward
        features = self.encoder.get_intermediate_layers(image, n=4, return_class_token=True)

        # Get last layer features and reshape
        feat, _ = features[-1]  # (B, N, C)
        feat = feat.permute(0, 2, 1).reshape(batch_size, -1, base_h, base_w)  # (B, C, H, W)

        # Simple head forward
        x = self.head['proj'](feat)

        # Add UV coordinates and upsample
        for i, key in enumerate(['upsample1', 'upsample2', 'upsample3']):
            h, w = x.shape[2], x.shape[3]
            uv = normalized_view_plane_uv_ts(w, h, aspect_ratio, dtype, device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            x = self.head[key](x)

        # Final outputs
        h, w = x.shape[2], x.shape[3]
        uv = normalized_view_plane_uv_ts(w, h, aspect_ratio, dtype, device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        points_raw = self.head['points_out'](x)
        mask_raw = self.head['mask_out'](x)

        # Resize to original
        points = F.interpolate(points_raw, size=(img_h, img_w), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask_raw, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # Post-process
        points = points.permute(0, 2, 3, 1)  # (B, H, W, 3)
        points = self._remap_points(points)
        mask = mask.squeeze(1).sigmoid()  # (B, H, W)

        mask_binary = mask > 0.5

        # Recover focal and shift
        focal, shift = recover_focal_shift_ts(points, mask_binary)

        # Compute depth
        depth = points[..., 2] + shift.view(batch_size, 1, 1)
        mask_binary = mask_binary & (depth > 0)

        # Compute intrinsics
        fx = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5
        cx = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        cy = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        intrinsics = intrinsics_from_focal_center_ts(fx, fy, cx, cy)

        # Reproject
        points = depth_map_to_point_map_ts(depth, intrinsics)

        # Apply mask
        depth = torch.where(mask_binary, depth, torch.tensor(float('inf'), device=device, dtype=dtype))

        # Format output
        if self.output_mode == 0:
            output_tensor = depth.unsqueeze(1)
        elif self.output_mode == 1:
            output_tensor = points.permute(0, 3, 1, 2)
        else:
            output_tensor = torch.stack([depth, mask], dim=1)

        return output_tensor


class MoGeForNukeV2Simple(nn.Module):
    """
    Self-contained TorchScript-compatible MoGe V2 wrapper.
    """

    image_mean: Tensor
    image_std: Tensor
    remap_output: str
    num_tokens_min: int
    num_tokens_max: int
    resolution_level: int
    output_mode: int
    mask_threshold: float

    def __init__(
        self,
        encoder_config: dict,
        neck_config: dict,
        points_head_config: dict,
        mask_head_config: dict,
        remap_output: str = 'linear',
        num_tokens_range: Tuple[int, int] = (1200, 3600),
        resolution_level: int = 9,
        output_mode: int = 0,
        mask_threshold: float = 0.5,
    ):
        super(MoGeForNukeV2Simple, self).__init__()

        self.remap_output = remap_output
        self.num_tokens_min = num_tokens_range[0]
        self.num_tokens_max = num_tokens_range[1]
        self.resolution_level = resolution_level
        self.output_mode = output_mode
        self.mask_threshold = mask_threshold

        # Build modules from configs
        from .torchscript_modules import DINOv2EncoderTS, ConvStackTS

        self.encoder = DINOv2EncoderTS(
            backbone=encoder_config.get('backbone', 'dinov2_vitl14'),
            intermediate_layers=encoder_config.get('intermediate_layers', 4),
            dim_out=encoder_config.get('dim_out', 256),
        )

        # Build neck and heads
        self.neck = self._build_convstack(neck_config)
        self.points_head = self._build_convstack(points_head_config)
        self.mask_head = self._build_convstack(mask_head_config)

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.eval()

    def _build_convstack(self, config: dict) -> nn.Module:
        """Build a ConvStack from config dict."""
        from .torchscript_modules import ConvStackTS

        # Get values with proper defaults, handling None values
        dim_in = config.get('dim_in') or [258, 2, 2, 2, 2]
        dim_res_blocks = config.get('dim_res_blocks') or [256, 128, 128, 64, 32]
        dim_out = config.get('dim_out') or [0, 0, 0, 0, 3]
        resamplers = config.get('resamplers') or ['pixel_shuffle'] * 4
        dim_times_res_block_hidden = config.get('dim_times_res_block_hidden') or 1
        num_res_blocks = config.get('num_res_blocks') or [1, 1, 1, 1, 1]

        # Convert None values in lists to 0
        dim_in = [x if x is not None else 0 for x in dim_in]
        dim_out = [x if x is not None else 0 for x in dim_out]

        return ConvStackTS(
            dim_in=dim_in,
            dim_res_blocks=dim_res_blocks,
            dim_out=dim_out,
            resamplers=resamplers,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
        )

    def _remap_points(self, points: Tensor) -> Tensor:
        if self.remap_output == 'linear':
            return points
        elif self.remap_output == 'sinh':
            return torch.sinh(points)
        elif self.remap_output == 'exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            z_exp = torch.exp(z)
            return torch.cat([xy * z_exp, z_exp], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy = points[..., :2]
            z = points[..., 2:3]
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            return points

    def forward(self, input: Tensor) -> Tensor:
        device = input.device
        dtype = input.dtype

        input = torch.clamp(input, min=0.0, max=1.0)

        batch_size = input.shape[0]
        img_h = input.shape[2]
        img_w = input.shape[3]
        aspect_ratio = float(img_w) / float(img_h)

        # Calculate tokens
        num_tokens = self.num_tokens_min + (self.resolution_level * (self.num_tokens_max - self.num_tokens_min)) // 9

        # Calculate base dimensions
        base_h = int(round((float(num_tokens) / aspect_ratio) ** 0.5))
        base_w = int(round((float(num_tokens) * aspect_ratio) ** 0.5))

        # Encoder forward
        features, cls_token = self.encoder(input, base_h, base_w, return_class_token=True)

        # Build feature pyramid with UV coordinates
        feature_list: List[Tensor] = []

        # Level 0: encoder features + UV
        uv0 = normalized_view_plane_uv_ts(base_w, base_h, aspect_ratio, dtype, device)
        uv0 = uv0.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        feature_list.append(torch.cat([features, uv0], dim=1))

        # Levels 1-4: UV only
        for level in range(1, 5):
            h_level = base_h * (2 ** level)
            w_level = base_w * (2 ** level)
            uv = normalized_view_plane_uv_ts(w_level, h_level, aspect_ratio, dtype, device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            feature_list.append(uv)

        # Neck
        neck_features = self.neck(feature_list)

        # Heads
        points_out = self.points_head(neck_features)[-1]
        mask_out = self.mask_head(neck_features)[-1]

        # Resize to original
        points = F.interpolate(points_out, size=(img_h, img_w), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask_out, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # Post-process
        points = points.permute(0, 2, 3, 1)
        points = self._remap_points(points)
        mask = mask.squeeze(1).sigmoid()

        mask_binary = mask > self.mask_threshold

        # Recover focal and shift
        focal, shift = recover_focal_shift_ts(points, mask_binary)

        # Compute depth
        depth = points[..., 2] + shift.view(batch_size, 1, 1)
        mask_binary = mask_binary & (depth > 0)

        # Compute intrinsics
        fx = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2.0 * (1.0 + aspect_ratio ** 2) ** 0.5
        cx = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        cy = torch.tensor(0.5, device=device, dtype=dtype).expand(batch_size)
        intrinsics = intrinsics_from_focal_center_ts(fx, fy, cx, cy)

        # Reproject
        points = depth_map_to_point_map_ts(depth, intrinsics)

        # Apply mask
        depth = torch.where(mask_binary, depth, torch.tensor(float('inf'), device=device, dtype=dtype))

        # Format output
        if self.output_mode == 0:
            output_tensor = depth.unsqueeze(1)
        elif self.output_mode == 1:
            output_tensor = points.permute(0, 3, 1, 2)
        else:
            output_tensor = torch.stack([depth, mask], dim=1)

        return output_tensor


def export_moge_to_torchscript(
    model: nn.Module,
    output_path: str,
    output_mode: int = 0,
    resolution_level: int = 9
) -> None:
    """
    Export a MoGe model to TorchScript format.

    Args:
        model: Loaded MoGe model (v1 or v2)
        output_path: Path to save the .pt file
        output_mode: 0=depth, 1=points, 2=depth+mask
        resolution_level: Default resolution level (0-9)
    """
    model.eval()

    # Determine model version and create wrapper
    if hasattr(model, 'neck'):
        # V2 model
        wrapper = MoGeForNukeV2(
            encoder=model.encoder,
            neck=model.neck,
            points_head=model.points_head,
            mask_head=model.mask_head,
            remap_output=model.remap_output,
            num_tokens_range=model.num_tokens_range,
            resolution_level=resolution_level,
            output_mode=output_mode
        )
    else:
        # V1 model
        wrapper = MoGeForNuke(
            encoder=model.backbone,
            head=model.head,
            remap_output=model.remap_output,
            num_tokens_range=model.num_tokens_range,
            resolution_level=resolution_level,
            output_mode=output_mode
        )

    wrapper.eval()

    # Script the model
    scripted = torch.jit.script(wrapper)

    # Save
    scripted.save(output_path)
    print(f"Model exported to {output_path}")
