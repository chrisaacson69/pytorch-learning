"""
PyTorch Automatic Differentiation (Autograd) Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

Autograd is PyTorch's automatic differentiation engine that powers neural network
training. It computes gradients automatically for backpropagation.
"""

import torch

# =============================================================================
# TENSORS, FUNCTIONS, AND COMPUTATIONAL GRAPH
# =============================================================================

# Simple neural network computation:
# Input: x
# Parameters: weights w, bias b
# Output: prediction z
# Loss: comparing z to y

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)  # weights (trainable)
b = torch.randn(3, requires_grad=True)  # bias (trainable)

# Forward pass
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("Forward pass:")
print(f"  x: {x}")
print(f"  w shape: {w.shape}")
print(f"  b: {b}")
print(f"  z (prediction): {z}")
print(f"  loss: {loss}")

# =============================================================================
# GRADIENT FUNCTIONS
# =============================================================================

# PyTorch builds a computational graph that tracks operations
# Each tensor has a grad_fn that references the function that created it

print(f"\nGradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# =============================================================================
# COMPUTING GRADIENTS
# =============================================================================

# backward() computes gradients of loss w.r.t. all tensors with requires_grad=True
loss.backward()

print(f"\nGradients after backward():")
print(f"  w.grad:\n{w.grad}")
print(f"  b.grad: {b.grad}")

# Note: x doesn't have gradients because requires_grad=False (default)

# =============================================================================
# DISABLING GRADIENT TRACKING
# =============================================================================

# Sometimes we don't need gradients (inference, frozen parameters)
# Two ways to disable:

# Method 1: torch.no_grad() context manager
z = torch.matmul(x, w) + b
print(f"\nWith gradient tracking: z.requires_grad = {z.requires_grad}")

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(f"Inside no_grad(): z.requires_grad = {z.requires_grad}")

# Method 2: detach() - creates a new tensor without gradient tracking
z = torch.matmul(x, w) + b
z_det = z.detach()
print(f"After detach(): z_det.requires_grad = {z_det.requires_grad}")

# =============================================================================
# GRADIENT ACCUMULATION
# =============================================================================

# By default, gradients ACCUMULATE (add up) on each backward() call
# You must zero them before each training iteration

inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()

# First backward call
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nFirst backward call:\n{inp.grad}")

# Second backward call - gradients accumulate!
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call (accumulated):\n{inp.grad}")

# Zero gradients, then backward
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nAfter zeroing gradients:\n{inp.grad}")

# =============================================================================
# KEY CONCEPTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. requires_grad=True tells PyTorch to track operations for gradients

2. Forward pass builds a computational graph (DAG)

3. backward() computes gradients via chain rule (backpropagation)

4. Gradients accumulate by default - call .zero_grad() before each step

5. Use torch.no_grad() or .detach() to disable gradient tracking:
   - During inference
   - For frozen parameters
   - When you don't need gradients

6. Only leaf tensors (created directly, not from operations) retain gradients
""")
