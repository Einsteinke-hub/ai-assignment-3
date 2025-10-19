# buggy_pytorch_code.py - Contains intentional bugs for debugging practice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("Loading CIFAR-10 dataset...")

# BUG 1: Missing normalization in transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # Missing: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define CNN model with BUGS
class BuggyCNN(nn.Module):
    def __init__(self):
        super(BuggyCNN, self).__init__()
        # BUG 2: Wrong input channels (should be 3 for RGB)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # BUG: input should be 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # BUG 3: Wrong calculation of flattened features
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # BUG: Wrong dimensions
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # BUG 4: Incorrect reshaping
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = BuggyCNN()
print("Model initialized (with bugs!)")

# BUG 5: Wrong loss function for multi-class classification
criterion = nn.BCEWithLogitsLoss()  # BUG: Should be CrossEntropyLoss for multi-class
# BUG 6: Missing optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with BUGS
def train_model():
    model.train()
    for epoch in range(2):  # Just 2 epochs for testing
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # BUG 7: No zero_grad
            # optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # BUG 8: Labels not converted to correct format
            loss = criterion(outputs, labels)
            
            # BUG 9: Missing backward pass and optimizer step
            # loss.backward()
            # optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/2], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Attempting to train (this will fail due to bugs)...")
try:
    train_model()
except Exception as e:
    print(f"Error encountered: {e}")
    print("This is expected! Now debug the code.")

print("\n" + "="*50)
print("BUG SUMMARY:")
print("1. Missing normalization in data transforms")
print("2. Wrong input channels in conv1 (1 instead of 3)")
print("3. Wrong flattened feature calculation in fc1")
print("4. Incorrect loss function (BCE instead of CrossEntropy)")
print("5. Missing optimizer initialization")
print("6. Missing zero_grad() in training loop")
print("7. Missing backward() and step() in training")
print("8. Labels not converted to correct format")
print("="*50)