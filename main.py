import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

train_data = datasets.FashionMNIST(root='data', train=True, download=False, transform=ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root='data', train=False, download=False, transform=ToTensor())

class_names = train_data.classes

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
train_features_batch, train_labels_batch = next(iter(train_dataloader))

flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)

class FashionModelMNISTModelV0(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, output_shape: int):
        super(FashionModelMNISTModelV0, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), ## dim becomes 14*14
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), ## dim becomes 7*7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape),
        )
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

# Accuracy function
def accuracy_fn(y_true, y_pred):
    """Calculate classification accuracy (as percentage)."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_train_time(start: float, end: float, device: device):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_and_save_model(model, train_dataloader, loss_fn, optimizer, epochs=5, model_path="fashion_model.pth"):
    train_time_start = timer()
    for epoch in range(epochs):
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            model.train()
            # Do the forward pass
            y_pred = model(X)
            # Calculate the loss
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
        train_loss /= len(train_dataloader)
        print(f"\nTrain loss: {train_loss:.5f}")
        train_time_end = timer()
        total_train_time_model_0 = print_train_time(start=train_time_start,
                                                    end=train_time_end,
                                                    device=device)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in path: {model_path}")

def load_and_evaluate(model_class, model_path, test_dataloader, loss_fn):
    model = model_class(1, 32, 10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    test_loss, test_accuracy = 0, 0
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_accuracy += accuracy_fn(y, test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)
    print(f"\nTest loss: {test_loss:.5f}, Test acc: {test_accuracy:.2f}%\n")

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, 0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

if __name__ == '__main__':
    model_0 = FashionModelMNISTModelV0(1, 32, 10).to(device)
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_and_save_model(model_0, train_dataloader, loss_fn, optimizer)
    load_and_evaluate(FashionModelMNISTModelV0, "fashion_model.pth", test_dataloader, loss_fn)

    # Make predictions on random test samples
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)
    pred_probs = make_predictions(model_0, test_samples)
    pred_classes = pred_probs.argmax(dim=1)

    # Plot predictions
    plt.figure(figsize=(9, 9))
    nrows, ncols = 3, 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(sample.squeeze(), cmap="gray")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")  # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r")  # red text if wrong
        plt.axis(False)
    plt.show()


