import torch
import torch.nn as nn

# The dataset: 4 examples instead of 1
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])

# Define the network as a class — PyTorch's standard way
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)   # 2 inputs -> 2 hidden neurons (creates weights+biases for you)
        self.output = nn.Linear(2, 1)   # 2 hidden -> 1 output

    def forward(self, x):
        h = torch.sigmoid(self.hidden(x))
        o = torch.sigmoid(self.output(h))
        return o

model = TinyNet()
loss_fn = nn.MSELoss()                          # same loss formula you used, built in
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # handles the weight updates for you

epochs = 2000
for epoch in range(epochs):
    predictions = model(X)        # forward pass on all 4 examples at once
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()         # same idea as your .grad.zero_() lines
    loss.backward()               # computes every gradient, for every example, automatically
    optimizer.step()              # same idea as your w -= lr * grad lines

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")

print("\nFinal predictions:")
print(model(X))
