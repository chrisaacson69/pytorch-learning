"""
PyTorch Build Model Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

Neural networks are composed of layers/modules that perform operations on data.
The torch.nn namespace provides all the building blocks for your own networks.
Every module in PyTorch subclasses nn.Module.
"""

import torch
from torch import nn

# =============================================================================
# GET DEVICE FOR TRAINING
# =============================================================================

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

# =============================================================================
# DEFINE THE MODEL CLASS
# =============================================================================


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Create an instance and move to device
model = NeuralNetwork().to(device)
print(model)

# =============================================================================
# USING THE MODEL
# =============================================================================

# Pass input data to the model (calls forward() automatically)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"\nPredicted class: {y_pred}")

# =============================================================================
# MODEL LAYERS BREAKDOWN
# =============================================================================

# Let's break down the layers in NeuralNetwork

# Sample input: minibatch of 3 images (28x28)
input_image = torch.rand(3, 28, 28)
print(f"\nInput image shape: {input_image.size()}")

# --- nn.Flatten ---
# Converts 2D (28x28) image into contiguous array of 784 values
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f"After Flatten: {flat_image.size()}")

# --- nn.Linear ---
# Applies linear transformation using weights and biases
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(f"After Linear(784, 20): {hidden1.size()}")

# --- nn.ReLU ---
# Non-linear activation that introduces non-linearity
# ReLU(x) = max(0, x)
print(f"\nBefore ReLU (first 5 values of first sample): {hidden1[0][:5]}")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU (first 5 values of first sample): {hidden1[0][:5]}")

# --- nn.Sequential ---
# Ordered container of modules. Data passes through in order.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(f"\nSequential output shape: {logits.size()}")

# --- nn.Softmax ---
# Scales logits to [0, 1] representing predicted probabilities
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Softmax output (probabilities sum to 1): {pred_probab[0].sum():.4f}")

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# nn.Module tracks all parameters automatically
# Parameters are optimized during training

print(f"\nModel structure:\n{model}\n")

print("Model parameters:")
for name, param in model.named_parameters():
    print(f"  Layer: {name} | Size: {param.size()}")

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
