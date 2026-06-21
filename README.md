# Neural Network From Scratch

A hands-on journey into understanding neural networks — starting with pen and paper, then raw Python, then PyTorch, then a real image classifier — built to actually understand backpropagation, not just call a library.

## What's in this repo

- **`neuron.py`** — A tiny neural network (2 inputs, 2 hidden neurons, 1 output) built entirely in raw Python using only the `math` module. Includes a manual forward pass, manual backpropagation (the chain rule, computed by hand), and manual gradient descent weight updates. No ML libraries used.
- **`neuron2.py`** — The exact same network, rewritten in PyTorch using `autograd`, so every gradient that was previously derived by hand is now computed automatically by `loss.backward()`. Verified to produce identical results to the manual version. Also extended to train on the XOR problem (4 training examples instead of 1).
- **`MNIST_P.py`** — A real handwritten digit classifier, trained on the MNIST dataset (60,000 training images, 28x28 pixels each). Uses a feedforward network with ReLU activation and cross-entropy loss for multi-class classification. Achieves **96.29% test accuracy** on unseen images.

## The approach

Before writing any code, every calculation in the first project was worked out by hand on paper — the forward pass, the loss, the backward pass, and the weight updates. The Python and PyTorch versions were then written and checked against those hand calculations to confirm real understanding of the math, rather than just trusting a library to "do the right thing." From there, the same core training loop (forward pass → loss → backward pass → update) was scaled up to a real-world image classification task.

## Concepts covered

- Forward propagation
- Sigmoid and ReLU activation functions
- Mean squared error and cross-entropy loss
- Backpropagation (chain rule, gradients, deltas)
- Gradient descent and PyTorch's `optim.SGD`
- PyTorch tensors, autograd, and `loss.backward()`
- Batching with `DataLoader`
- Multi-class image classification

## Running it

\`\`\`bash
python3 manual_backpropagation.py     # raw Python version (single example)
python3 pytorch_xor_classifier.py     # PyTorch version (XOR, multiple examples)
python3 mnist_digit_classifier.py     # MNIST digit classifier (downloads dataset automatically)
\`\`\`

## Results

| Project | Result |
|---|---|
| Manual neuron (1 example) | Loss: 0.125 → 0.0001 over 1000 epochs |
| XOR (PyTorch) | Predictions: [0.05, 0.94, 0.94, 0.08] vs target [0, 1, 1, 0] |
| MNIST digit classifier | 96.29% test accuracy |

## Next steps

- Visualize predictions, including misclassified digits
- Try improving MNIST accuracy (more neurons, more epochs, different optimizer)
- Convolutional neural networks (CNNs) for better image performance
