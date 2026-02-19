"""
PyTorch Transforms Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

Transforms are used to modify data for training. Common uses:
- Converting images to tensors
- Normalizing pixel values
- Data augmentation
- Converting labels to one-hot encoding
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize

# =============================================================================
# BASIC TRANSFORMS
# =============================================================================

# ToTensor: Converts a PIL Image or numpy array to a FloatTensor
# and scales pixel values from [0, 255] to [0.0, 1.0]

# Lambda: Applies any user-defined lambda function
# Here we convert integer labels to one-hot encoded tensors

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)

# =============================================================================
# UNDERSTANDING THE TRANSFORMS
# =============================================================================

# Get a sample
image, label = ds[0]

print("Image transform (ToTensor):")
print(f"  Type: {type(image)}")
print(f"  Shape: {image.shape}")
print(f"  Dtype: {image.dtype}")
print(f"  Value range: [{image.min():.2f}, {image.max():.2f}]")

print("\nLabel transform (one-hot encoding):")
print(f"  Original label would be an integer (e.g., 9 for Ankle Boot)")
print(f"  Transformed label: {label}")
print(f"  Shape: {label.shape}")

# =============================================================================
# LAMBDA TRANSFORM EXPLAINED
# =============================================================================

# The lambda transform creates a one-hot encoded vector:
# 1. torch.zeros(10, dtype=torch.float) - creates a zero tensor of size 10
# 2. scatter_(0, torch.tensor(y), value=1) - puts 1 at index y

# Example breakdown:
target_transform = Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(
        dim=0, index=torch.tensor(y), value=1
    )
)

# Manual demonstration:
print("\nOne-hot encoding demonstration:")
for class_idx in range(10):
    one_hot = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(class_idx), 1)
    print(f"  Class {class_idx}: {one_hot.tolist()}")

# =============================================================================
# COMPOSING TRANSFORMS
# =============================================================================

# You can chain multiple transforms using Compose
composed_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

ds_normalized = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=composed_transform,
)

image_normalized, label = ds_normalized[0]
print(f"\nAfter normalization:")
print(f"  Value range: [{image_normalized.min():.2f}, {image_normalized.max():.2f}]")

# =============================================================================
# COMMON TRANSFORMS FOR IMAGE DATA
# =============================================================================

print("\nCommon torchvision transforms:")
print("  ToTensor() - Convert PIL/numpy to tensor, scale to [0,1]")
print("  Normalize(mean, std) - Normalize with mean and std")
print("  Resize(size) - Resize image")
print("  CenterCrop(size) - Crop center of image")
print("  RandomCrop(size) - Random crop (data augmentation)")
print("  RandomHorizontalFlip() - Random horizontal flip")
print("  RandomRotation(degrees) - Random rotation")
print("  ColorJitter() - Random brightness, contrast, etc.")
print("  Compose([...]) - Chain multiple transforms")
