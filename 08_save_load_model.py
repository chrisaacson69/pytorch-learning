"""
PyTorch Save and Load Model Tutorial
https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

Learn how to persist model state for later use - saving weights,
loading trained models, and running inference.
"""

import torch
from torch import nn

# =============================================================================
# DEFINE A MODEL (same as previous tutorials)
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


model = NeuralNetwork()
print("Original model:")
print(model)

# =============================================================================
# SAVING AND LOADING MODEL WEIGHTS (RECOMMENDED)
# =============================================================================

# PyTorch models store learned parameters in state_dict
# This is the recommended approach for saving/loading

# View the state_dict
print("\nModel state_dict keys:")
for key in model.state_dict().keys():
    print(f"  {key}")

# Save the model weights
torch.save(model.state_dict(), "model_weights.pth")
print("\nSaved model weights to model_weights.pth")

# Load weights into a new model instance
# IMPORTANT: You must create the model architecture first
model2 = NeuralNetwork()
model2.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model2.eval()  # Set to evaluation mode for inference

print("Loaded model weights into new model instance")

# Verify weights are the same
for (name1, param1), (name2, param2) in zip(
    model.named_parameters(), model2.named_parameters()
):
    assert torch.equal(param1, param2), f"Mismatch in {name1}"
print("Verified: weights are identical")

# =============================================================================
# SAVING AND LOADING ENTIRE MODEL (NOT RECOMMENDED)
# =============================================================================

# You can save the entire model (architecture + weights)
# However, this approach has limitations:
# - Pickled data is bound to specific classes and directory structure
# - Code changes can break loading

torch.save(model, "model_complete.pth")
print("\nSaved complete model to model_complete.pth")

# Load the complete model
# weights_only=False is required but less secure
model3 = torch.load("model_complete.pth", weights_only=False)
model3.eval()

print("Loaded complete model")

# =============================================================================
# SAVING FOR INFERENCE VS TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("SAVING BEST PRACTICES:")
print("=" * 60)
print("""
FOR INFERENCE (deployment):
    # Save
    torch.save(model.state_dict(), 'model_weights.pth')

    # Load
    model = MyModelClass()
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.eval()  # IMPORTANT: set to eval mode

FOR RESUMING TRAINING:
    # Save (include optimizer state and epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pth')

    # Load
    checkpoint = torch.load('checkpoint.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()  # Set to training mode to continue training
""")

# =============================================================================
# EXAMPLE: CHECKPOINT FOR RESUMING TRAINING
# =============================================================================

# Simulate having an optimizer and training state
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epoch = 5
loss = 0.25

# Save checkpoint
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}
torch.save(checkpoint, "checkpoint.pth")
print("\nSaved training checkpoint to checkpoint.pth")

# Load checkpoint
loaded_checkpoint = torch.load("checkpoint.pth", weights_only=False)
print(f"Loaded checkpoint from epoch {loaded_checkpoint['epoch']}")
print(f"Loss at checkpoint: {loaded_checkpoint['loss']}")

# =============================================================================
# RUNNING INFERENCE
# =============================================================================

print("\n" + "=" * 60)
print("RUNNING INFERENCE:")
print("=" * 60)

# Always call model.eval() before inference
model.eval()

# Create dummy input (batch of 1 image, 28x28)
dummy_input = torch.randn(1, 28, 28)

# Run inference with no gradient computation
with torch.no_grad():
    output = model(dummy_input)
    prediction = output.argmax(1)

print(f"Input shape: {dummy_input.shape}")
print(f"Output logits: {output}")
print(f"Predicted class: {prediction.item()}")
