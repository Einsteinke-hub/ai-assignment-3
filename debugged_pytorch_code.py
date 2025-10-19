# debugged_pytorch_code.py - Fixed version of the buggy code
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("Loading CIFAR-10 dataset...")

# FIX 1: Added proper normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # FIXED
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define CNN model - ALL BUGS FIXED
class FixedCNN(nn.Module):
    def __init__(self):
        super(FixedCNN, self).__init__()
        # FIX 2: Correct input channels (3 for RGB)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # FIXED: input channels = 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # FIX 3: Correct calculation of flattened features
        # After two pooling layers: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # FIXED: Correct dimensions
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # This is actually correct
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = FixedCNN()
print("Model initialized successfully!")

# FIX 4: Correct loss function for multi-class classification
criterion = nn.CrossEntropyLoss()  # FIXED: CrossEntropy for multi-class
# FIX 5: Added optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # FIXED

# Training loop - ALL BUGS FIXED
def train_model():
    model.train()
    losses = []
    for epoch in range(5):  # Increased to 5 epochs
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # FIX 6: Zero gradients
            optimizer.zero_grad()  # FIXED
            
            # Forward pass
            outputs = model(images)
            
            # FIX 7: Labels are already in correct format for CrossEntropyLoss
            loss = criterion(outputs, labels)  # FIXED: No conversion needed
            
            # FIX 8: Backward pass and optimizer step
            loss.backward()  # FIXED
            optimizer.step()  # FIXED
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                current_loss = loss.item()
                losses.append(current_loss)
                print(f'Epoch [{epoch+1}/5], Step [{i+1}/{len(train_loader)}], Loss: {current_loss:.4f}')
    
    return losses

print("Training model (this should work now!)...")
losses = train_model()

# Evaluate the model
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

accuracy = evaluate_model()

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss Over Batches')
plt.xlabel('Batch (x100)')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('images/debugged_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("SUCCESS! All bugs have been fixed:")
print(f"- Model trained successfully with {len(losses)} loss recordings")
print(f"- Final test accuracy: {accuracy:.2f}%")
print("- Training loss plot saved to images/debugged_training_loss.png")
print("="*50)

# Save the trained model
torch.save(model.state_dict(), 'debugged_cifar_model.pth')
print("Model saved as 'debugged_cifar_model.pth'")