"""
PyTorch Optimization Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

Training a model is an iterative process. Each iteration (epoch):
1. Forward pass: compute predictions
2. Compute loss: measure error
3. Backward pass: compute gradients
4. Update parameters: adjust weights to reduce loss
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# =============================================================================
# PREREQUISITE: LOAD DATA AND DEFINE MODEL
# =============================================================================

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


model = NeuralNetwork()

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Hyperparameters are adjustable parameters that control training
# They affect model performance and convergence

learning_rate = 1e-3  # How much to update parameters each step
batch_size = 64  # Number of samples per gradient update
epochs = 10  # Number of times to iterate over the dataset

print("Hyperparameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")

# =============================================================================
# LOSS FUNCTION
# =============================================================================

# Loss function measures how wrong the model's predictions are
# Common loss functions:
#   - nn.MSELoss (Mean Squared Error) - for regression
#   - nn.NLLLoss (Negative Log Likelihood) - for classification
#   - nn.CrossEntropyLoss - combines LogSoftmax + NLLLoss

loss_fn = nn.CrossEntropyLoss()
print(f"\nLoss function: {loss_fn}")

# =============================================================================
# OPTIMIZER
# =============================================================================

# Optimizer adjusts model parameters to reduce loss
# Common optimizers:
#   - SGD (Stochastic Gradient Descent)
#   - Adam (Adaptive Moment Estimation)
#   - RMSprop

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f"Optimizer: {optimizer}")

# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # Set model to training mode

    for batch, (X, y) in enumerate(dataloader):
        # Forward pass: compute prediction
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass: compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients for next iteration
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


# =============================================================================
# TESTING LOOP
# =============================================================================


def test_loop(dataloader, model, loss_fn):
    model.eval()  # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# =============================================================================
# FULL TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")

# =============================================================================
# KEY CONCEPTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING LOOP ANATOMY:")
print("=" * 60)
print("""
Each training iteration:

1. model.train()      - Enable training mode (dropout, batch norm)
2. pred = model(X)    - Forward pass
3. loss = loss_fn()   - Compute loss
4. loss.backward()    - Compute gradients (backpropagation)
5. optimizer.step()   - Update parameters
6. optimizer.zero_grad() - Clear gradients for next iteration

Each testing iteration:

1. model.eval()       - Enable evaluation mode
2. torch.no_grad()    - Disable gradient computation
3. pred = model(X)    - Forward pass
4. Compute metrics    - Accuracy, loss, etc.
""")
