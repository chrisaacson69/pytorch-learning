"""
PyTorch Datasets & DataLoaders Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

PyTorch provides two data primitives:
- torch.utils.data.Dataset: stores samples and their labels
- torch.utils.data.DataLoader: wraps an iterable around Dataset for easy access
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# =============================================================================
# LOADING A DATASET
# =============================================================================

# FashionMNIST is a dataset of Zalando's article images
# 60,000 training examples and 10,000 test examples
# Each example is a 28x28 grayscale image with a label from 10 classes

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

print(f"Training data size: {len(training_data)}")
print(f"Test data size: {len(test_data)}")

# =============================================================================
# VISUALIZING THE DATASET
# =============================================================================

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.suptitle("Random samples from FashionMNIST")
plt.tight_layout()
plt.savefig("fashion_mnist_samples.png")
print("\nSaved visualization to fashion_mnist_samples.png")
plt.show()

# =============================================================================
# CREATING A CUSTOM DATASET
# =============================================================================

# Custom datasets must implement three functions:
# __init__, __len__, and __getitem__

import os
import pandas as pd
from torchvision.io import decode_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Note: This custom dataset example requires an annotations CSV file
# with columns: [image_filename, label]

# =============================================================================
# PREPARING DATA FOR TRAINING WITH DATALOADERS
# =============================================================================

# DataLoader wraps a dataset and provides:
# - Automatic batching
# - Shuffling
# - Multiprocess data loading

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# =============================================================================
# ITERATE THROUGH THE DATALOADER
# =============================================================================

# Get a batch of training data
train_features, train_labels = next(iter(train_dataloader))
print(f"\nFeature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Display a single image from the batch
img = train_features[0].squeeze()
label = train_labels[0]
plt.figure()
plt.imshow(img, cmap="gray")
plt.title(f"Label: {labels_map[label.item()]}")
plt.savefig("single_sample.png")
print(f"\nSaved single sample to single_sample.png")
print(f"Label: {label} ({labels_map[label.item()]})")
plt.show()
