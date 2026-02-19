# CLAUDE.md

**Vault:** `C:\Users\Chris.Isaacson\Vault\projects\pytorch-learning\README.md`

## Overview

Tutorial scripts following the official PyTorch "Learn the Basics" series. Eight
sequential files covering tensors through model persistence, all using the
FashionMNIST dataset. Learning project, not a library.

## Tech Stack

- **Python 3.14.2**
- **PyTorch nightly** (CUDA 12.8): `torch`, `torchvision`
- **matplotlib** -- dataset visualization (03, outputs PNGs)
- **numpy** -- tensor bridge demos (02)
- **pandas** -- custom dataset example (03, import only)
- Git repo with GitHub remote

## Setup

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Project Structure

```
01_quickstart.py             # Full ML workflow: load data, train, infer (FashionMNIST)
02_tensors.py                # Tensor creation, attributes, ops, numpy bridge
03_datasets_dataloaders.py   # Dataset/DataLoader, visualization, custom Dataset class
04_transforms.py             # ToTensor, Lambda (one-hot), Compose, Normalize
05_build_model.py            # nn.Module, Sequential, layer breakdown, param counting
06_autograd.py               # Computational graph, backward(), gradient accumulation
07_optimization.py           # Train/test loops, loss functions, optimizers, 10 epochs
08_save_load_model.py        # state_dict save/load, full model save, checkpointing
data/FashionMNIST/           # Auto-downloaded dataset (~30 MB)
model.pth                    # Saved model weights (from 01)
model_weights.pth            # Saved weights (from 08)
model_complete.pth           # Full model pickle (from 08)
checkpoint.pth               # Training checkpoint (from 08)
README.md
.gitignore
```

## How to Run

```bash
# Run any tutorial individually
python 01_quickstart.py
python 02_tensors.py
# etc.
```

Scripts auto-download FashionMNIST to `data/` on first run. Scripts 01 and 07
train for 5 and 10 epochs respectively -- expect a few minutes on CPU.

## Key Notes

- Uses `torch.accelerator` API (not legacy `torch.cuda`) for device detection
- Model architecture: Flatten -> Linear(784,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,10)
- 01 and 07 both train the same NeuralNetwork but 07 has the detailed loop breakdown
- 03 saves `fashion_mnist_samples.png` and `single_sample.png`
- 06 demonstrates gradient accumulation bug and the fix (zero_grad)
- 08 shows both recommended (state_dict) and discouraged (full pickle) save methods
- All scripts are self-contained -- no imports between tutorial files
