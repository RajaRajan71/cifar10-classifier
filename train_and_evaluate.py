import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# --- 1. Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Data Pipeline ---
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# --- 3. Model Architecture (Custom CNN from scratch) ---
class CustomCNN(nn.Module):

  def __init__(self):
    super(CustomCNN, self).__init__()
    self.features = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Output: 64 x 16 x 16
        nn.Dropout(0.25),
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Output: 128 x 8 x 8
        nn.Dropout(0.25),
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


model = CustomCNN().to(device)


# --- 4. Training Utilities & Part 2 Option B (Early Stopping & Gradient Clipping) ---
class EarlyStopping:

  def __init__(self, patience=5, path="checkpoint.pth"):
    self.patience = patience
    self.counter = 0
    self.best_loss = float("inf")
    self.early_stop = False
    self.path = path

  def __call__(self, val_loss, model):
    if val_loss < self.best_loss:
      self.best_loss = val_loss
      torch.save(model.state_dict(), self.path)
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
early_stopping = EarlyStopping(patience=5, path="best_cifar_model.pth")

epochs = 15
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for i, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Part 2: Gradient Clipping utility   ###############      PART 2 ############################################
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

  train_loss = running_loss / len(trainloader)
  train_acc = 100.0 * correct / total
  train_losses.append(train_loss)
  train_accuracies.append(train_acc)

  # Evaluation on test set
  model.eval()
  test_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, targets in testloader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

  test_loss = test_loss / len(testloader)
  test_acc = 100.0 * correct / total
  test_losses.append(test_loss)
  test_accuracies.append(test_acc)

  scheduler.step(test_loss)
  early_stopping(test_loss, model)

  print(
      f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc:"
      f" {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc:"
      f" {test_acc:.2f}%"
  )

  if early_stopping.early_stop:
    print("Early stopping triggered!")
    break


# --- 5. Evaluation & Misclassified Samples ---
model.load_state_dict(torch.load("best_cifar_model.pth"))
model.eval()

misclassified_images = []
misclassified_preds = []
true_labels = []

with torch.no_grad():
  for inputs, targets in testloader:
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    _, predicted = outputs.max(1)
    idxs = (predicted != targets).nonzero(as_tuple=True)[0]
    for idx in idxs:
      if len(misclassified_images) < 5:
        misclassified_images.append(inputs[idx].cpu())
        misclassified_preds.append(classes[predicted[idx].item()])
        true_labels.append(classes[targets[idx].item()])
    if len(misclassified_images) >= 5:
      break

# Plot Loss & Accuracy Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss Curves")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.legend()
plt.title("Accuracy Curves")
plt.savefig("metrics_curve.png")
print("Saved performance curves to metrics_curve.png")