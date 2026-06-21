# Neural Network From Scratch

A hands-on journey into understanding neural networks — starting with pen and paper, then raw Python, then PyTorch — built to actually understand backpropagation, not just call a library.

## What's in this repo

- **`neuron.py`** — A tiny neural network (2 inputs, 2 hidden neurons, 1 output) built entirely in raw Python using only the `math` module. Includes a manual forward pass, manual backpropagation (the chain rule, computed by hand), and manual gradient descent weight updates. No ML libraries used.
- **`neuron2.py`** — The exact same network, rewritten in PyTorch using `autograd`, so every gradient that was previously derived by hand is now computed automatically by `loss.backward()`. Verified to produce identical results to the manual version.

## The approach

Before writing any code, every calculation was first worked out by hand on paper — the forward pass, the loss, the backward pass, and the weight updates. The Python and PyTorch versions were then written and checked against those hand calculations to confirm real understanding of the math, rather than just trusting a library to "do the right thing."

## Concepts covered

- Forward propagation
- Sigmoid activation function
- Mean squared error loss
- Backpropagation (chain rule, gradients, deltas)
- Gradient descent weight updates
- PyTorch tensors, autograd, and `loss.backward()`

## Running it

\`\`\`bash
python3 neuron.py    # raw Python version
python3 neuron2.py   # PyTorch version
\`\`\`

## Next steps

- Training on multiple examples (XOR problem)
- Image classification (MNIST handwritten digits)
