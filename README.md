# 🟠 Convolutional Neural Network Classification on FashionMNIST (PyTorch)
This is my implementation of a convolutional neural network (CNN) for classifying grayscale images from the FashionMNIST dataset using PyTorch. It closely follows the workflow taught in "PyTorch for Deep Learning & Machine Learning – Full Course" by Daniel Bourke. I started with only using multi-layer perceptrons (with nn.Linear) which only got me around 80% accuracy. After using nn.Conv2d and nn.MaxPool2d, and tweaking the optimizer, epoch, and number of hidden units, I was able to get up to 91.48% accuracy.

## The main purpose of this mini-project is to practice and understand:

- Building CNN architectures from scratch in PyTorch
- Managing model training and evaluation on image data
- Using device-agnostic code (CPU vs CUDA)
- Saving and loading model weights
- Making predictions and visualizing model performance on unseen data

## 🧠 Key Concepts Practiced

- Loading and preparing FashionMNIST using torchvision.datasets
- Building a multi-layer CNN using nn.Conv2d, nn.ReLU, and nn.MaxPool2d
- Writing a custom training loop with loss tracking and accuracy computation
- Understanding underfitting and testing different optimizers (e.g. SGD vs Adam)
- Saving model weights using torch.save() and reloading with torch.load()
- Writing a reusable make_predictions() function for inference
- Visualizing predictions using matplotlib in a 3x3 grid with colored labels

