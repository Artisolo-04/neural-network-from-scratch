import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Download and load the MNIST dataset (downloads automatically the first time)
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the network
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()           # turns each 28x28 image into one long row of 784 numbers
        self.hidden = nn.Linear(28*28, 128)   # 784 inputs -> 128 hidden neurons
        self.output = nn.Linear(128, 10)      # 128 hidden -> 10 outputs (one per digit)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

model = DigitNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 5
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

# Check accuracy on images the model has never seen
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

print(f"\nTest accuracy: {100 * correct / total:.2f}%")
