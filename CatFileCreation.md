# CAT File Creation Guide for Foundry Nuke

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Workflow Overview](#workflow-overview)
4. [Preparing Your PyTorch Model](#preparing-your-pytorch-model)
5. [Converting to TorchScript](#converting-to-torchscript)
6. [Creating the .cat File in Nuke](#creating-the-cat-file-in-nuke)
7. [Using the Inference Node](#using-the-inference-node)
8. [Advanced Topics](#advanced-topics)
9. [Complete Examples](#complete-examples)
10. [Appendix: Reference Tables](#appendix-reference-tables)

---

## Introduction

This guide details the process of preparing machine learning architectures for use in Foundry Nuke. The workflow involves converting PyTorch models to TorchScript (`.pt` files), which are then converted to `.cat` files for inference within Nuke's Inference node.

### Purpose
- Convert PyTorch models to TorchScript for Nuke compatibility
- Enable ML model inference directly within Nuke's compositing workflow
- Allow dynamic control of model parameters through custom Nuke knobs

---

## Requirements

### PyTorch Version Requirements

**For Nuke 13.x:**
- `torch==1.6`
- `torchvision==0.7`

**For Nuke 14:**
- `torch==1.12.1`
- `torchvision==0.13.1`

**Important Notes:**
- `.cat` files are version-specific and must match your Nuke version
- Ensure your development environment uses the correct PyTorch version
- Images must be in sRGB color space

---

## Workflow Overview

The complete workflow consists of 5 main steps:

1. **Develop PyTorch Model** - Create and train your model using standard PyTorch
2. **Prepare Model for TorchScript** - Modify model to meet TorchScript requirements
3. **Convert to TorchScript** - Save model as `.pt` file using `torch.jit.script()`
4. **Generate .cat File** - Use Nuke's CatFileCreator node to convert `.pt` to `.cat`
5. **Deploy in Nuke** - Use the Inference node to run your model on image data

---

## Preparing Your PyTorch Model

### Basic Model Structure

Your model must inherit from `torch.nn.Module` and implement the required methods:

```python
import torch
import torch.nn as nn

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize your model architecture here
        self.model = YourNetworkArchitecture()
        self.model.eval()  # Set to evaluation mode

    def forward(self, input):
        # Process input and return output
        output = self.model(input)
        return output
```

### Tensor Size Requirements

**Input Tensor:**
- Must be size: `1 x inChan x inH x inW`
  - `1` = batch dimension (always 1)
  - `inChan` = number of input channels
  - `inH` = input height
  - `inW` = input width

**Output Tensor:**
- Must be size: `1 x outChan x outH x outW`
  - `1` = batch dimension (always 1)
  - `outChan` = number of output channels
  - `outH` = output height (can be scaled)
  - `outW` = output width (can be scaled)

**Examples:**
- RGB input, binary output: `1 x 3 x H x W` → `1 x 1 x H x W`
- 5-channel input, 3-channel output with 2x scaling: `1 x 5 x inH x inW` → `1 x 3 x (2*inH) x (2*inW)`

### Pixel Range Handling

**Key Considerations:**
- Tensor values typically correspond to image pixel values
- Most values will be in the range `[0, 1]`
- Superblack/superwhite pixels can produce values outside `[0, 1]`

**Preprocessing Options:**

**Option 1: In-model clamping**
```python
def forward(self, input):
    # Clamp input values to [0, 1]
    input = torch.clamp(input, min=0.0, max=1.0)
    output = self.model(input)
    return output
```

**Option 2: Pre-processing in Nuke**
Use a Clamp node before the Inference node in your Nuke script.

### Device Assignment (CPU/GPU)

Models must handle both CPU and GPU execution dynamically:

```python
def forward(self, input):
    # Detect device from input tensor
    if input.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create all tensors on the correct device
    output = torch.ones(1, 1, input.shape[2], input.shape[3], device=device)
    return output
```

**Important Notes:**
- All tensors in the `forward()` function must be on the same device
- The Inference node has a "Use GPU if available" checkbox
- Only single GPU usage is currently supported

### Precision Handling (Full/Half Precision)

When "Optimize for Speed and Memory" is enabled in the Inference node, tensors are converted to half precision:

```python
def forward(self, input):
    # Match input tensor's dtype
    output = torch.ones(1, 1, input.shape[2], input.shape[3], dtype=input.dtype)
    return output
```

**Best Practices:**
- Always use `dtype=input.dtype` when creating new tensors
- Ensures compatibility with both full and half precision modes
- Improves speed and memory efficiency when optimization is enabled

### Combining Device and Precision Handling

**Complete example:**
```python
def forward(self, input):
    # Handle both device and precision
    output = torch.ones(
        1, 1, input.shape[2], input.shape[3],
        device=input.device,
        dtype=input.dtype
    )
    return output
```

---

## Converting to TorchScript

### TorchScript Requirements

TorchScript is a statically-typed subset of Python. Your model must comply with several restrictions:

#### 1. Type Annotations

All functions must have explicit type annotations:

```python
def normalize(self, input, mean, std):
    # type: (Tensor, Tuple[float, float, float], Tuple[float, float, float]) -> Tensor
    # Normalization code here
    return normalized_input
```

#### 2. Class Attributes

All class attributes must be declared in `__init__`:

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # All attributes MUST be initialized here
        self.my_attribute = some_value
```

**Invalid:**
```python
class MyModel(torch.nn.Module):
    class_attribute = 10  # This will cause errors!
```

#### 3. Static Variables

Variables must maintain a single static type:

```python
# Invalid - variable changes type
def forward(self, input):
    result = 0
    if condition:
        result = torch.tensor([1.0])  # Changes from int to Tensor!
    return result

# Valid - consistent type
def forward(self, input):
    result = torch.zeros(1)
    if condition:
        result = torch.tensor([1.0])
    return result
```

#### 4. None and Optional Types

Use type annotations for Optional types:

```python
def __init__(self):
    super().__init__()
    self.optional_value: Optional[int] = None
```

To use Optional class attributes outside `__init__`, assign to a local variable first:

```python
def forward(self, input):
    local_value = self.optional_value
    if local_value is not None:
        # Use local_value here
        pass
```

#### 5. External Libraries

**Supported:**
- Python built-in functions
- `math` module
- PyTorch operations

**Not Supported:**
- NumPy
- OpenCV
- Other external libraries

Must rewrite operations using PyTorch-equivalent functions.

#### 6. Inheritance Limitations

**TorchScript does not support inheritance.** You cannot create subclasses.

**Invalid:**
```python
class BaseModel(torch.nn.Module):
    pass

class MyModel(BaseModel):  # This won't work!
    pass
```

**Workaround - Use composition:**
```python
class FirstModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize first model

    def forward(self, input):
        return output

class SecondModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Embed first model as an attribute
        self.first_model = FirstModel()

    def forward(self, input):
        intermediate = self.first_model(input)
        output = self.process(intermediate)
        return output
```

### Converting and Saving

Once your model meets all TorchScript requirements:

```python
# Create model instance
model = MyModel()

# Convert to TorchScript using scripting
scripted_model = torch.jit.script(model)

# Save as .pt file
scripted_model.save('my_model.pt')
```

**Note:** Use `torch.jit.script()` rather than `torch.jit.trace()` for better compatibility with Nuke.

---

## Creating the .cat File in Nuke

### Using the CatFileCreator Node

1. **Open NukeX** (CatFileCreator requires NukeX, not standard Nuke)

2. **Create CatFileCreator node:**
   - In the Node Graph, right-click and select `Other > CatFileCreator`

3. **Configure the node:**
   - **Torchscript File:** Browse to your `.pt` file
   - **Model ID:** Enter a unique identifier for your model
   - **Channels In:** Specify input channel count (e.g., 3 for RGB)
   - **Channels Out:** Specify output channel count
   - **Output Scale:** Set if your model changes image dimensions

4. **Add Custom Knobs (Optional):**
   - Integer Knob
   - Float Knob
   - Position Knob (2D or 3D)
   - Boolean Knob
   - Enumeration Knob

   See [Custom Knobs](#custom-knobs) section for details.

5. **Generate .cat file:**
   - **Cat File:** Specify output path for `.cat` file
   - Click **Create Cat File** button

### Channel Configuration

**Available Channels (19 total):**
- `rgba.red`, `rgba.green`, `rgba.blue`, `rgba.alpha`
- `depth.Z`
- `forward.u`, `forward.v`
- `backward.u`, `backward.v`
- `disparityLeft.x`, `disparityRight.x`
- `mask.red`, `mask.green`, `mask.blue`, `mask.alpha`
- And additional motion/deep channels

**Example Configurations:**

**Single RGB input:**
```
Channels In: rgba.red, rgba.green, rgba.blue
```

**Multiple inputs (6 channels):**
```
Channels In: rgba.red, rgba.green, rgba.blue, forward.u, forward.v, backward.u
```

**Multiple outputs:**
```
Channels Out: rgba.red, rgba.green, forward.u, forward.v
```

---

## Using the Inference Node

### Basic Setup

1. **Create Inference node:**
   - Connect your source image(s) to the Inference node

2. **Load model:**
   - **Model File:** Browse to your `.cat` file

3. **Configure settings:**
   - **Use GPU if available:** Enable for GPU acceleration
   - **Optimize for Speed and Memory:** Enable for half-precision inference

4. **Adjust custom knobs:** If your model has custom knobs, they will appear in the node properties

5. **View output:** Connect a Viewer node to see results

### Multi-Input Workflow

For models requiring multiple inputs:

1. **Combine images using Shuffle nodes:**
   - Use Shuffle nodes to combine multiple images into one multi-channel image
   - Example: Combine two RGB images into a 6-channel image

2. **Configure CatFileCreator:**
   - Set appropriate channel count in Channels In

3. **Split outputs:**
   - Use Shuffle nodes after Inference to separate multi-channel outputs

---

## Advanced Topics

### Custom Knobs

Custom knobs allow dynamic control of model parameters directly from Nuke's interface.

#### Implementation Requirements

**1. Define attribute in model's `__init__`:**
```python
class MyModel(torch.nn.Module):
    def __init__(self, userLabel=0):
        super().__init__()
        self.userLabel = userLabel  # Attribute name must match knob name
```

**2. Use attribute in `forward` method:**
```python
    def forward(self, input):
        # Use self.userLabel in your logic
        if condition == self.userLabel:
            output = process_a(input)
        else:
            output = process_b(input)
        return output
```

**3. Add knob in CatFileCreator:**
- Add corresponding knob type in CatFileCreator
- Knob name must exactly match the attribute name in your model
- Configure knob options/values

#### Supported Knob Types

**Integer Knob:**
```python
def __init__(self, int_knob=0):
    super().__init__()
    self.int_knob = int_knob
```

**Float Knob:**
```python
def __init__(self, float_knob=1.0):
    super().__init__()
    self.float_knob = float_knob
```

**2D Position Knob:**
```python
def __init__(self, xy_knob=torch.tensor([0, 0])):
    super().__init__()
    self.xy_knob = xy_knob
```

**3D Position Knob:**
```python
def __init__(self, xyz_knob=torch.tensor([0, 0, 0])):
    super().__init__()
    self.xyz_knob = xyz_knob
```

**Boolean Knob (Checkbox):**
```python
def __init__(self, bool_knob=0):  # Must use 0/1, not True/False
    super().__init__()
    self.bool_knob = bool_knob
```

**Enumeration Knob (Pulldown):**
```python
def __init__(self, enum_knob=0):  # Integer between 0 and (num_options - 1)
    super().__init__()
    self.enum_knob = enum_knob
```

#### Important Limitations

- Cannot use `Optional[]` type attributes
- Knob name must exactly match attribute name
- Limited to supported Nuke knob types listed above

### Accessing Nested Attributes

When one model contains another model as an attribute, you cannot directly control nested attributes with knobs (Nuke doesn't allow '.' in knob names).

**Solution:** Pass parameters through the parent model:

```python
class NestedModel(torch.nn.Module):
    def __init__(self, label=0):
        super().__init__()
        self.label = label

    def forward(self, input):
        # Use self.label
        return output

class ParentModel(torch.nn.Module):
    def __init__(self, userLabel=0):
        super().__init__()
        self.userLabel = userLabel
        self.nested = NestedModel()

    def forward(self, input):
        # Assign userLabel to nested model's label
        self.nested.label = self.userLabel
        output = self.nested(input)
        return output
```

Now you can control the nested attribute via the `userLabel` knob in the parent model.

### Models with Multiple Inputs

The Inference node only accepts one input image. To use models requiring multiple inputs:

**1. Modify the forward function to accept a single wide tensor:**

**Original:**
```python
def forward(self, img1, img2):
    output = self.model.forward(img1, img2)
    return output
```

**Modified:**
```python
def forward(self, input):
    # Split the combined input tensor
    input_split = torch.split(input, 3, 1)  # Split along channel dimension
    img1 = input_split[0]  # First 3 channels
    img2 = input_split[1]  # Second 3 channels

    output = self.model.forward(img1, img2)
    return output
```

**2. In Nuke:**
- Use Shuffle nodes to combine multiple images into one multi-channel image
- Example: Combine two RGB images into a 6-channel image
- Configure CatFileCreator with 6 input channels

**Example Channel Configuration:**
```
Channels In (6 channels):
- rgba.red
- rgba.green
- rgba.blue
- forward.u
- forward.v
- backward.u
```

### Models with Multiple Outputs

The Inference node returns only one output image. To output multiple results:

**1. Concatenate outputs in the forward function:**

```python
def forward(self, input):
    # Model produces multiple outputs
    [img1, img2] = self.model.forward(input)

    # Concatenate along channel dimension
    output = torch.cat((img1, img2), 1)
    return output
```

**2. Configure CatFileCreator:**
- Set Channels Out to map concatenated channels to specific Nuke channels
- Example: First 3 channels to RGB, next 2 to forward.u/v

**Example:**
```
Channels Out (5 channels):
- rgba.red
- rgba.green
- rgba.blue
- forward.u
- forward.v
```

**3. In Nuke:**
- Use Shuffle nodes after Inference to split multi-channel output
- Separate channels into individual images as needed

---

## Complete Examples

### Example 1: Object Detection with Cifar10ResNet

This example demonstrates a complete object detection model with custom knobs and normalization.

**Model Code:**
```python
import torch
import torch.nn as nn
from typing import Tuple

class Cifar10ResNet(torch.nn.Module):
    def __init__(self, userLabel=0):
        super().__init__()
        # Custom knob attribute
        self.userLabel = userLabel

        # Load pre-trained ResNet model
        self.model = ResNet(pretrained=True)
        self.model.eval()

        # Normalization parameters for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def normalize(self, input, mean, std):
        # type: (Tensor, Tuple[float, float, float], Tuple[float, float, float]) -> Tensor
        """Normalize input tensor using mean and std."""
        mean_tensor = torch.tensor(mean, device=input.device, dtype=input.dtype)
        std_tensor = torch.tensor(std, device=input.device, dtype=input.dtype)

        # Reshape for broadcasting
        mean_tensor = mean_tensor.view(1, 3, 1, 1)
        std_tensor = std_tensor.view(1, 3, 1, 1)

        return (input - mean_tensor) / std_tensor

    def forward(self, input):
        # Clamp input to valid range
        input = torch.clamp(input, min=0.0, max=1.0)

        # Normalize input
        normalized = self.normalize(
            input,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

        # Run inference
        modelOutput = self.model.forward(normalized)

        # Get predicted label
        modelLabel = int(torch.argmax(modelOutput[0]))

        # Compare with user-selected label
        if modelLabel == self.userLabel:
            output = torch.ones(
                1, 1, input.shape[2], input.shape[3],
                device=input.device,
                dtype=input.dtype
            )
        else:
            output = torch.zeros(
                1, 1, input.shape[2], input.shape[3],
                device=input.device,
                dtype=input.dtype
            )

        return output

# Convert to TorchScript
resnet = Cifar10ResNet()
module = torch.jit.script(resnet)
module.save('cifar10_resnet.pt')
```

**Nuke Setup:**
1. Create CatFileCreator node
2. Load `cifar10_resnet.pt`
3. Configure:
   - Channels In: 3 (RGB)
   - Channels Out: 1 (Alpha)
4. Add Enumeration Knob:
   - Name: `userLabel`
   - Options: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
5. Create Cat File
6. Use Inference node with RGB input
7. Output appears in alpha channel

### Example 2: Color Transfer with Multiple Inputs

This example shows a color transfer model that takes two images and transfers the color distribution from one to the other.

**Model Code:**
```python
import torch
import torch.nn as nn

class LinearColourTransfer(torch.nn.Module):
    """Core linear color transfer model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, input):
        # input shape: [B, 3, H, W]
        B, C, H, W = input.shape

        # Reshape for linear layer: [B*H*W, 3]
        reshaped = input.permute(0, 2, 3, 1).reshape(-1, 3)

        # Apply linear transformation
        transformed = self.linear(reshaped)

        # Reshape back: [B, 3, H, W]
        output = transformed.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return output

class ColourTransfer(torch.nn.Module):
    """Wrapper model with multiple inputs and mix control."""
    def __init__(self, mixValue=1.0):
        super().__init__()
        self.mixValue = mixValue
        self.colour_transfer = LinearColourTransfer()

        # Normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def normalize(self, input):
        mean = self.mean.to(device=input.device, dtype=input.dtype)
        std = self.std.to(device=input.device, dtype=input.dtype)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        return (input - mean) / std

    def denormalize(self, input):
        mean = self.mean.to(device=input.device, dtype=input.dtype)
        std = self.std.to(device=input.device, dtype=input.dtype)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        return (input * std) + mean

    def forward(self, input):
        # Split combined input into two images
        input_split = torch.split(input, 3, 1)
        img1 = input_split[0]  # Source image
        img2 = input_split[1]  # Target image (for color reference)

        # Clamp inputs
        img1 = torch.clamp(img1, min=0.0, max=1.0)
        img2 = torch.clamp(img2, min=0.0, max=1.0)

        # Normalize both images
        img1_norm = self.normalize(img1)
        img2_norm = self.normalize(img2)

        # Apply color transfer to img1
        transferred = self.colour_transfer(img1_norm)

        # Denormalize
        transferred = self.denormalize(transferred)

        # Mix between original and transferred based on mixValue
        output = (1.0 - self.mixValue) * img1 + self.mixValue * transferred

        # Clamp output
        output = torch.clamp(output, min=0.0, max=1.0)

        return output

# Convert to TorchScript
model = ColourTransfer()
module = torch.jit.script(model)
module.save('colour_transfer.pt')
```

**Nuke Setup:**
1. Prepare two RGB images in Nuke
2. Use Shuffle nodes to combine them into 6 channels:
   - Channels 0-2: First image (rgba.red, rgba.green, rgba.blue)
   - Channels 3-5: Second image (forward.u, forward.v, backward.u)
3. Create CatFileCreator node
4. Load `colour_transfer.pt`
5. Configure:
   - Channels In: 6 (two RGB images)
   - Channels Out: 3 (RGB output)
6. Add Float Knob:
   - Name: `mixValue`
   - Range: 0.0 to 1.0
   - Default: 1.0
7. Create Cat File
8. Use Inference node
9. Adjust `mixValue` slider to blend between original and color-transferred images

---

## Appendix: Reference Tables

### Supported Nuke Channels (19 Total)

| Category | Channels |
|----------|----------|
| **RGBA** | `rgba.red`, `rgba.green`, `rgba.blue`, `rgba.alpha` |
| **Depth** | `depth.Z` |
| **Forward Motion** | `forward.u`, `forward.v` |
| **Backward Motion** | `backward.u`, `backward.v` |
| **Disparity** | `disparityLeft.x`, `disparityRight.x` |
| **Mask** | `mask.red`, `mask.green`, `mask.blue`, `mask.alpha` |
| **Deep** | `front`, `back` |

### Supported Custom Knob Types

| Knob Type | Python Type | Example Definition | Notes |
|-----------|-------------|-------------------|-------|
| **Integer** | `int` | `int_knob = 0` | Standard integer values |
| **Float** | `float` | `float_knob = 1.0` | Floating-point values |
| **2D Position** | `torch.Tensor` | `xy_knob = torch.tensor([0, 0])` | 2-element tensor |
| **3D Position** | `torch.Tensor` | `xyz_knob = torch.tensor([0, 0, 0])` | 3-element tensor |
| **Boolean** | `int` | `bool_knob = 0` | Must use 0 or 1, not True/False |
| **Enumeration** | `int` | `enum_knob = 0` | Integer index (0 to num_options-1) |

### TorchScript Restrictions Summary

| Feature | Supported | Notes |
|---------|-----------|-------|
| **Inheritance** | ❌ No | Use composition instead |
| **Optional Types** | ⚠️ Limited | Requires type annotation and careful handling |
| **Class Attributes** | ✅ Yes | Must be declared in `__init__` |
| **Static Variables** | ✅ Yes | Must maintain single type throughout |
| **Built-in Functions** | ✅ Yes | Python built-ins and `math` module |
| **External Libraries** | ❌ No | NumPy, OpenCV not supported - use PyTorch equivalents |
| **Type Annotations** | ✅ Required | All functions must have type hints |
| **Multiple Inputs** | ⚠️ Workaround | Combine into single tensor, split in forward() |
| **Multiple Outputs** | ⚠️ Workaround | Concatenate outputs into single tensor |

### Best Practices Checklist

- [ ] Model inherits from `torch.nn.Module`
- [ ] All attributes declared in `__init__`
- [ ] Forward function handles device assignment dynamically
- [ ] Forward function handles precision (dtype) dynamically
- [ ] Input/output tensors have correct dimensions (batch=1)
- [ ] Pixel values clamped or normalized appropriately
- [ ] No inheritance used (composition instead)
- [ ] All functions have type annotations
- [ ] No external libraries (NumPy, OpenCV) used
- [ ] Model tested with `torch.jit.script()` successfully
- [ ] Custom knob names match model attribute names exactly
- [ ] Model set to evaluation mode (`model.eval()`)

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Type error during scripting | Missing type annotations | Add type hints to all functions |
| Attribute not found | Attribute not in `__init__` | Move all attributes to `__init__` |
| Device mismatch error | Tensors on different devices | Use `device=input.device` for all tensors |
| Precision error | dtype mismatch | Use `dtype=input.dtype` for all tensors |
| Inheritance error | Class uses inheritance | Refactor to use composition |
| Optional type error | Improper Optional handling | Add type annotation, assign to local variable |
| Knob not appearing | Name mismatch | Ensure knob name exactly matches attribute name |
| Import error | Unsupported library | Rewrite using PyTorch operations |

---

## Additional Resources

**Official Documentation:**
- Foundry Nuke CAT File Creation Guide: https://learn.foundry.com/nuke/developers/latest/catfilecreationreferenceguide/
- PyTorch TorchScript Documentation: https://pytorch.org/docs/stable/jit.html

**Key Takeaways:**
1. Always use the correct PyTorch version for your Nuke version
2. Test TorchScript conversion early in development
3. Handle device and precision dynamically in all models
4. Follow TorchScript restrictions carefully
5. Use composition instead of inheritance
6. Add type annotations to all functions
7. Initialize all attributes in `__init__`
8. Test with both CPU and GPU, full and half precision
9. Ensure input/output tensor dimensions match expectations
10. Custom knobs provide powerful runtime control of model behavior
