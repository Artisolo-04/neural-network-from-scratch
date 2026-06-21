import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 1 input channel (grayscale) -> 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 filters
        self.pool = nn.MaxPool2d(2, 2)                             # shrinks image by half each time
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)          # 28x28 -> 14x14
        x = torch.relu(self.conv2(x))
        x = self.pool(x)          # 14x14 -> 7x7
        x = x.view(x.size(0), -1) # flatten before the fully-connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: average loss = {total_loss/len(train_loader):.4f}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

print(f"\nTest accuracy: {100 * correct / total:.2f}%")
