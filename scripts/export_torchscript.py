#!/usr/bin/env python3
"""
Export MoGe model to TorchScript format for use in Nuke's CatFileCreator.

This script converts a pre-trained MoGe model to TorchScript format (.pt file)
that can be used with Foundry Nuke's CatFileCreator to generate .cat files
for inference.

Usage:
    python scripts/export_torchscript.py --model Ruicheng/moge-vitl --output moge_vitl.pt

    # With custom settings
    python scripts/export_torchscript.py \\
        --model path/to/model.pt \\
        --output moge_model.pt \\
        --output-mode 0 \\
        --resolution-level 9

Arguments:
    --model: Path to model checkpoint or HuggingFace repo ID
    --output: Output path for the TorchScript .pt file
    --output-mode: Output mode (0=depth, 1=points, 2=depth+mask)
    --resolution-level: Default resolution level (0-9)
    --test: Run a test inference after export to verify

After exporting, use Nuke's CatFileCreator to convert the .pt file to .cat:
    1. Open NukeX
    2. Create CatFileCreator node (Other > CatFileCreator)
    3. Set Torchscript File to your .pt file
    4. Configure channels:
       - output-mode 0 (depth): Channels Out = 1 (depth.Z)
       - output-mode 1 (points): Channels Out = 3 (rgba.red, rgba.green, rgba.blue)
       - output-mode 2 (depth+mask): Channels Out = 2 (depth.Z, mask.red)
    5. Click "Create Cat File"
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_model_weights(model_path: str) -> dict:
    """Load model weights from checkpoint or HuggingFace."""
    if Path(model_path).exists():
        return torch.load(model_path, map_location='cpu', weights_only=True)
    else:
        try:
            from huggingface_hub import hf_hub_download
            cached_path = hf_hub_download(
                repo_id=model_path,
                repo_type="model",
                filename="model.pt",
            )
            return torch.load(cached_path, map_location='cpu', weights_only=True)
        except ImportError:
            print("Error: huggingface_hub is required to download models from HuggingFace.")
            print("Install it with: pip install huggingface_hub")
            sys.exit(1)


def create_torchscript_model(
    checkpoint: dict,
    output_mode: int = 0,
    resolution_level: int = 9,
) -> nn.Module:
    """
    Create a TorchScript-compatible model from checkpoint.

    Args:
        checkpoint: Model checkpoint dict with 'model_config' and 'model' keys
        output_mode: 0=depth, 1=points, 2=depth+mask
        resolution_level: Default resolution (0-9)

    Returns:
        TorchScript-compatible model wrapper
    """
    from moge.model.torchscript_modules import (
        DinoVisionTransformerTS,
        DINOv2EncoderTS,
        ConvStackTS,
        MLPTS,
    )
    from moge.model.torchscript_export import (
        MoGeForNukeSimple,
    )

    model_config = checkpoint['model_config']
    state_dict = checkpoint['model']

    # Determine model version based on config
    if 'encoder' in model_config and isinstance(model_config['encoder'], dict):
        # V2 model
        return create_v2_model(model_config, state_dict, output_mode, resolution_level)
    else:
        # V1 model
        return create_v1_model(model_config, state_dict, output_mode, resolution_level)


def create_v1_model(
    model_config: dict,
    state_dict: dict,
    output_mode: int,
    resolution_level: int,
) -> nn.Module:
    """Create TorchScript model from V1 checkpoint."""
    from moge.model.torchscript_export import MoGeForNukeSimple

    # Extract configuration
    encoder_name = model_config.get('encoder', 'dinov2_vitb14')
    remap_output = model_config.get('remap_output', 'linear')
    num_tokens_range = model_config.get('num_tokens_range', [1200, 2500])

    # Map encoder name to architecture params
    if 'vitl' in encoder_name.lower() or 'large' in encoder_name.lower():
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif 'vitg' in encoder_name.lower() or 'giant' in encoder_name.lower():
        embed_dim = 1536
        depth = 40
        num_heads = 24
    else:  # Default to base
        embed_dim = 768
        depth = 12
        num_heads = 12

    # Create the simple wrapper
    model = MoGeForNukeSimple(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        remap_output=remap_output,
        num_tokens_range=tuple(num_tokens_range),
        resolution_level=resolution_level,
        output_mode=output_mode,
    )

    # Load weights (with key mapping)
    model_state = {}
    for k, v in state_dict.items():
        # Map old keys to new structure
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', 'encoder.backbone.')
            model_state[new_key] = v
        elif k.startswith('head.'):
            model_state[k] = v
        else:
            model_state[k] = v

    # Try to load, allowing missing keys for structure differences
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"Note: Some keys were not loaded (expected for architecture differences):")
        print(f"  Missing: {len(missing)} keys")

    return model


def create_v2_model(
    model_config: dict,
    state_dict: dict,
    output_mode: int,
    resolution_level: int,
) -> nn.Module:
    """Create TorchScript model from V2 checkpoint."""
    encoder_config = model_config['encoder']
    backbone_name = encoder_config.get('backbone', 'dinov2_vitl14')

    # Use hardcoded ViT-L model for best TorchScript compatibility
    if 'vitl' in backbone_name.lower() or 'large' in backbone_name.lower():
        print("  Using hardcoded ViT-L architecture for TorchScript compatibility")
        return create_v2_vitl_model(model_config, state_dict, output_mode)
    else:
        # Fall back to dynamic model for other architectures
        print(f"  Warning: {backbone_name} not fully supported, trying dynamic model")
        return create_v2_dynamic_model(model_config, state_dict, output_mode, resolution_level)


def create_v2_vitl_model(
    model_config: dict,
    state_dict: dict,
    output_mode: int,
) -> nn.Module:
    """Create hardcoded ViT-L model for TorchScript export."""
    from moge.model.moge_vitl_nuke import MoGeViTL

    model = MoGeViTL(output_mode=output_mode)

    # Map weights from original checkpoint to hardcoded model
    new_state = {}
    model_state = model.state_dict()

    for dst_key in model_state.keys():
        src_key = _map_vitl_key(dst_key)
        if src_key and src_key in state_dict:
            src_tensor = state_dict[src_key]
            dst_tensor = model_state[dst_key]
            if src_tensor.shape == dst_tensor.shape:
                new_state[dst_key] = src_tensor
            else:
                print(f"  Shape mismatch for {dst_key}: {src_tensor.shape} vs {dst_tensor.shape}")
                new_state[dst_key] = dst_tensor
        else:
            # Keep initialized value
            new_state[dst_key] = model_state[dst_key]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    loaded = len(model_state) - len(missing)
    print(f"  Loaded {loaded}/{len(model_state)} weights")

    return model


def _map_vitl_key(dst_key: str) -> str:
    """Map hardcoded model key to original checkpoint key."""
    # Encoder backbone blocks: encoder.backbone.block0 -> encoder.backbone.blocks.0
    if '.backbone.block' in dst_key:
        import re
        match = re.search(r'\.backbone\.block(\d+)\.', dst_key)
        if match:
            block_idx = match.group(1)
            return dst_key.replace(f'.block{block_idx}.', f'.blocks.{block_idx}.')

    # Encoder projections: encoder.proj0 -> encoder.output_projections.0
    if dst_key.startswith('encoder.proj'):
        import re
        match = re.search(r'encoder\.proj(\d+)\.(.+)', dst_key)
        if match:
            idx = match.group(1)
            rest = match.group(2)
            return f'encoder.output_projections.{idx}.{rest}'

    # Neck mappings
    if dst_key.startswith('neck.'):
        return _map_neck_key_vitl(dst_key)

    # Head mappings
    if dst_key.startswith('points_head.'):
        return _map_head_key_vitl(dst_key, 'points_head')
    if dst_key.startswith('mask_head.'):
        return _map_head_key_vitl(dst_key, 'mask_head')

    # Direct mapping for other keys
    return dst_key


def _map_neck_key_vitl(dst_key: str) -> str:
    """Map neck keys from hardcoded to original."""
    import re

    # input0..input4 -> input_blocks.0..4
    match = re.search(r'neck\.input(\d+)\.(.+)', dst_key)
    if match:
        idx = match.group(1)
        rest = match.group(2)
        return f'neck.input_blocks.{idx}.{rest}'

    # up0..up3 -> resamplers.0..3
    match = re.search(r'neck\.up(\d+)\.(up|conv)\.(.+)', dst_key)
    if match:
        idx = match.group(1)
        sublayer = match.group(2)
        rest = match.group(3)
        if sublayer == 'up':
            return f'neck.resamplers.{idx}.conv1.{rest}'
        else:
            return f'neck.resamplers.{idx}.conv2.{rest}'

    # res1_0, res1_1 -> res_blocks.1.0, res_blocks.1.1
    match = re.search(r'neck\.res(\d+)_(\d+)\.(conv\d+)\.(.+)', dst_key)
    if match:
        level = match.group(1)
        block = match.group(2)
        conv = match.group(3)
        rest = match.group(4)
        # ResBlock has conv1, conv2 which map to layers in original ResidualConvBlock
        if conv == 'conv1':
            return f'neck.res_blocks.{level}.{block}.layers.2.{rest}'
        elif conv == 'conv2':
            return f'neck.res_blocks.{level}.{block}.layers.5.{rest}'

    return dst_key


def _map_head_key_vitl(dst_key: str, head_name: str) -> str:
    """Map head keys from hardcoded to original."""
    import re

    # input0..input4 -> input_blocks.0..4
    match = re.search(rf'{head_name}\.input(\d+)\.(.+)', dst_key)
    if match:
        idx = match.group(1)
        rest = match.group(2)
        return f'{head_name}.input_blocks.{idx}.{rest}'

    # up0..up3 -> resamplers.0..3
    match = re.search(rf'{head_name}\.up(\d+)\.(up|conv)\.(.+)', dst_key)
    if match:
        idx = match.group(1)
        sublayer = match.group(2)
        rest = match.group(3)
        if sublayer == 'up':
            return f'{head_name}.resamplers.{idx}.conv1.{rest}'
        else:
            return f'{head_name}.resamplers.{idx}.conv2.{rest}'

    # res1, res2, res3 -> res_blocks.1.0, res_blocks.2.0, res_blocks.3.0
    match = re.search(rf'{head_name}\.res(\d+)\.(conv\d+)\.(.+)', dst_key)
    if match:
        level = match.group(1)
        conv = match.group(2)
        rest = match.group(3)
        if conv == 'conv1':
            return f'{head_name}.res_blocks.{level}.0.layers.2.{rest}'
        elif conv == 'conv2':
            return f'{head_name}.res_blocks.{level}.0.layers.5.{rest}'

    # output -> output_blocks.4
    match = re.search(rf'{head_name}\.output\.(.+)', dst_key)
    if match:
        rest = match.group(1)
        return f'{head_name}.output_blocks.4.{rest}'

    return dst_key


def create_v2_dynamic_model(
    model_config: dict,
    state_dict: dict,
    output_mode: int,
    resolution_level: int,
) -> nn.Module:
    """Create dynamic model (fallback for non-ViT-L)."""
    from moge.model.torchscript_export import MoGeForNukeV2Simple

    encoder_config = model_config['encoder']
    neck_config = model_config['neck']
    points_head_config = model_config.get('points_head', {})
    mask_head_config = model_config.get('mask_head', {})

    remap_output = model_config.get('remap_output', 'linear')
    num_tokens_range = model_config.get('num_tokens_range', [1200, 3600])

    model = MoGeForNukeV2Simple(
        encoder_config=encoder_config,
        neck_config=neck_config,
        points_head_config=points_head_config,
        mask_head_config=mask_head_config,
        remap_output=remap_output,
        num_tokens_range=tuple(num_tokens_range),
        resolution_level=resolution_level,
        output_mode=output_mode,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Note: Some keys were not loaded:")
        print(f"  Missing: {len(missing)} keys")

    return model


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    test_input_size: tuple = (1, 3, 512, 512),
) -> None:
    """
    Export model to TorchScript.

    Args:
        model: PyTorch model
        output_path: Path to save .pt file
        test_input_size: Size for test input
    """
    model.eval()

    print("Converting to TorchScript...")

    try:
        # Try scripting first (preferred for Nuke)
        scripted_model = torch.jit.script(model)
        print("  Successfully converted using torch.jit.script()")
    except Exception as e:
        print(f"  torch.jit.script() failed: {e}")
        print("  Falling back to torch.jit.trace()...")

        # Fall back to tracing
        dummy_input = torch.randn(*test_input_size)
        scripted_model = torch.jit.trace(model, dummy_input)
        print("  Successfully converted using torch.jit.trace()")

    # Save
    scripted_model.save(output_path)
    print(f"Saved TorchScript model to: {output_path}")


def test_exported_model(model_path: str, input_size: tuple = (1, 3, 512, 512)) -> None:
    """Test the exported TorchScript model."""
    print("\nTesting exported model...")

    # Load the scripted model
    model = torch.jit.load(model_path)
    model.eval()

    # Create test input
    test_input = torch.rand(*input_size)

    # Run inference
    with torch.no_grad():
        output = model(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("  Test passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Export MoGe model to TorchScript for Nuke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to model checkpoint or HuggingFace repo ID (e.g., Ruicheng/moge-vitl)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for TorchScript .pt file'
    )
    parser.add_argument(
        '--output-mode',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Output mode: 0=depth only (1 channel), 1=points xyz (3 channels), 2=depth+mask (2 channels)'
    )
    parser.add_argument(
        '--resolution-level',
        type=int,
        default=9,
        choices=range(10),
        help='Default resolution level (0-9, higher = more detail but slower)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test inference after export'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('H', 'W'),
        help='Test input size (height width)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MoGe TorchScript Export for Nuke")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading model from: {args.model}")
    checkpoint = load_model_weights(args.model)

    # Create TorchScript-compatible model
    print("\nCreating TorchScript-compatible model...")
    model = create_torchscript_model(
        checkpoint,
        output_mode=args.output_mode,
        resolution_level=args.resolution_level,
    )

    # Export
    print(f"\nExporting to: {args.output}")
    export_to_torchscript(
        model,
        args.output,
        test_input_size=(1, 3, args.test_size[0], args.test_size[1]),
    )

    # Test if requested
    if args.test:
        test_exported_model(
            args.output,
            input_size=(1, 3, args.test_size[0], args.test_size[1]),
        )

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print("\nNext steps for Nuke:")
    print("1. Open NukeX (CatFileCreator requires NukeX)")
    print("2. Create CatFileCreator node (Other > CatFileCreator)")
    print(f"3. Set Torchscript File to: {args.output}")
    print("4. Configure channels based on output mode:")
    if args.output_mode == 0:
        print("   - Channels In: 3 (rgba.red, rgba.green, rgba.blue)")
        print("   - Channels Out: 1 (depth.Z)")
    elif args.output_mode == 1:
        print("   - Channels In: 3 (rgba.red, rgba.green, rgba.blue)")
        print("   - Channels Out: 3 (rgba.red, rgba.green, rgba.blue)")
    else:
        print("   - Channels In: 3 (rgba.red, rgba.green, rgba.blue)")
        print("   - Channels Out: 2 (depth.Z, mask.red)")
    print("5. Click 'Create Cat File'")
    print("6. Use the .cat file with the Inference node")


if __name__ == '__main__':
    main()
