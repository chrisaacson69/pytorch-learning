"""
PyTorch Tensors Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

Tensors are the fundamental data structure in PyTorch - similar to arrays and
matrices. They're used for encoding inputs, outputs, and model parameters.
"""

import torch
import numpy as np

# =============================================================================
# INITIALIZING TENSORS
# =============================================================================

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data:\n{x_data}\n")

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from numpy:\n{x_np}\n")

# From another tensor (retains shape and datatype)
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# =============================================================================
# TENSOR ATTRIBUTES
# =============================================================================

tensor = torch.rand(3, 4)
print(f"\nShape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Move tensor to GPU if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Tensor moved to: {tensor.device}")

# =============================================================================
# TENSOR OPERATIONS
# =============================================================================

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"\nOriginal tensor:\n{tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

# Modify a column
tensor[:, 1] = 0
print(f"After setting column 1 to 0:\n{tensor}")

# Joining tensors with torch.cat
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"\nConcatenated tensor (dim=1):\n{t1}")

# =============================================================================
# ARITHMETIC OPERATIONS
# =============================================================================

# Matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"\nMatrix multiplication result:\n{y1}")

# Element-wise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"Element-wise multiplication:\n{z1}")

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(f"\nSum as Python number: {agg_item}, type: {type(agg_item)}")

# In-place operations (denoted by _ suffix)
print(f"\nBefore add_: \n{tensor}")
tensor.add_(5)
print(f"After add_(5): \n{tensor}")

# =============================================================================
# BRIDGE WITH NUMPY
# =============================================================================

# Tensor to NumPy array (share memory on CPU)
t = torch.ones(5)
print(f"\nt: {t}")
n = t.numpy()
print(f"n: {n}")

# Changes in tensor reflect in numpy array
t.add_(1)
print(f"After t.add_(1):")
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor (share memory)
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in numpy array reflect in tensor
np.add(n, 1, out=n)
print(f"\nAfter np.add(n, 1):")
print(f"t: {t}")
print(f"n: {n}")
