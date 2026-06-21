import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

# Generate one synthetic digit image
def generate_digit_image(digit, size=28):
    img = Image.new('L', (size, size), color=0)  # black background
    draw = ImageDraw.Draw(img)
    font_size = random.randint(18, 24)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
    )
    x_offset = random.randint(-2, 4)
    y_offset = random.randint(-4, 2)
    draw.text((6 + x_offset, 2 + y_offset), str(digit), fill=255, font=font)
    angle = random.randint(-10, 10)
    img = img.rotate(angle, fillcolor=0)
    return np.array(img, dtype=np.float32) / 255.0

# Build the dataset: 200 fake images per digit
images, labels = [], []
for digit in range(10):
    for _ in range(200):
        images.append(generate_digit_image(digit))
        labels.append(digit)
images = np.array(images)
X = torch.tensor(images).unsqueeze(1)   # shape: [2000, 1, 28, 28]
y = torch.tensor(labels)

# Split into train/test
split = int(0.8 * len(X))
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Reuse your CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 300
for epoch in range(epochs):
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 30 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

with torch.no_grad():
    test_preds = model(X_test)
    accuracy = (torch.argmax(test_preds, dim=1) == y_test).float().mean()
    print(f"\nAccuracy on fake test data: {accuracy.item()*100:.2f}%")



from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
real_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
real_test_loader = DataLoader(real_test_data, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images_real, labels_real in real_test_loader:
        predictions = model(images_real)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct += (predicted_labels == labels_real).sum().item()
        total += labels_real.size(0)

print(f"\nAccuracy on REAL MNIST digits: {100 * correct / total:.2f}%")


# Pull a small number of REAL examples to mix in
real_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

real_images, real_labels = [], []
counts = {i: 0 for i in range(10)}
for img, label in real_train_data:
    if counts[label] < 20:
        real_images.append(img.squeeze(0).numpy())
        real_labels.append(label)
        counts[label] += 1
    if all(c >= 20 for c in counts.values()):
        break

real_images = np.array(real_images)
X_real_small = torch.tensor(real_images).unsqueeze(1)
y_real_small = torch.tensor(real_labels)

# Combine synthetic + small real sample
X_combined = torch.cat([X_train, X_real_small], dim=0)
y_combined = torch.cat([y_train, y_real_small], dim=0)

# Train a fresh model on the combined dataset
model2 = CNN()
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

epochs = 300
for epoch in range(epochs):
    predictions = model2(X_combined)
    loss = loss_fn(predictions, y_combined)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    if epoch % 30 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# Re-test on real MNIST
correct, total = 0, 0
with torch.no_grad():
    for images_real, labels_real in real_test_loader:
        predictions = model2(images_real)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct += (predicted_labels == labels_real).sum().item()
        total += labels_real.size(0)

print(f"\nAccuracy on REAL MNIST (after mixing in 200 real examples): {100 * correct / total:.2f}%")
