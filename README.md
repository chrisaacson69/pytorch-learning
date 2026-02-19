# PyTorch Learning

**Vault:** `C:\Users\Chris.Isaacson\Vault\projects\pytorch-learning\README.md`

Tutorial files based on the official PyTorch "Learn the Basics" series:
https://docs.pytorch.org/tutorials/beginner/basics/intro.html

## Setup

Python 3.14.2 with PyTorch nightly (CUDA 12.8):
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Tutorial Files

Run these in order for the best learning experience:

| File | Topic | Key Concepts |
|------|-------|--------------|
| `01_quickstart.py` | Complete ML workflow | End-to-end training and inference |
| `02_tensors.py` | Tensors | Data structures, operations, NumPy bridge |
| `03_datasets_dataloaders.py` | Data handling | Loading, batching, custom datasets |
| `04_transforms.py` | Data transforms | Preprocessing, augmentation, one-hot encoding |
| `05_build_model.py` | Neural networks | nn.Module, layers, forward pass |
| `06_autograd.py` | Automatic differentiation | Gradients, computational graph |
| `07_optimization.py` | Training | Loss functions, optimizers, training loop |
| `08_save_load_model.py` | Persistence | Save/load weights, checkpoints |

## Running

```bash
cd pytorch-learning
python 01_quickstart.py
python 02_tensors.py
# ... etc
```

## Learning Path

**If you're new to ML:** Start with `02_tensors.py` and work through sequentially.

**If you know ML but not PyTorch:** Start with `01_quickstart.py` for a quick overview, then dive into specific topics.

## Generated Files

- `data/` - FashionMNIST dataset (auto-downloaded)
- `model.pth` - Trained model weights
- `*.png` - Visualization outputs
