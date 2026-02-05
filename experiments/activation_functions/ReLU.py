import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchview import draw_graph

import json
import os

torch.manual_seed(635215)

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using {device} device")

# Define the neural network


class ReLUNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with ReLU and batch normalization
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # Fully connected layers with ReLU
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f'batch {batch} completed')

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, capture_batches=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0.0
    correct = 0.0
    captured_batches = []

    with torch.no_grad():
        for batch_num, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_cpu = y.detach().cpu()

            pred = model(X)
            pred_cpu = pred.detach().cpu()

            if batch_num < capture_batches:
                captured_batches.append({
                    "batch_index": batch_num,
                    "inputs": X.detach().cpu(),
                    "targets": y_cpu,
                    "logits": pred_cpu,
                    "predicted_classes": pred_cpu.argmax(1)
                })

            test_loss += loss_fn(pred, y).item()
            correct += (pred_cpu.argmax(1) ==
                        y_cpu).type(torch.float).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = correct / size

    print(
        f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "examples": captured_batches
    }


def train_and_evaluate_model(epochs, train_loader, test_loader, model, loss_fn, optimizer, desired_accuracy=1):
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        epoch_metrics = test(test_loader, model, loss_fn)
        # torch.xpu.synchronize()
        if epoch_metrics['accuracy'] >= desired_accuracy:
            print(
                f"Desired accuracy of {desired_accuracy} reached, stopping training.")
            break

    print("Training complete!")

    torch.save(model.state_dict(), "ReLU/cifar10_relu_model.pth")
    print("Model saved to ReLU/cifar10_relu_model.pth")

    print("\nFinal evaluation on test set:")
    final_metrics = test(test_loader, model, loss_fn)
    print(f"Final Test Accuracy: {(100*final_metrics['accuracy']):>0.1f}%")


def get_batch_examples(num_examples, test_loader, model, loss_fn):
    print(f"Capturing {num_examples} batches from the test set...")
    captured_data = test(test_loader, model, loss_fn,
                         capture_batches=num_examples)
    with open('ReLU/data_relu.json', 'w') as f:
        def tensor_converter(o):
            if isinstance(o, torch.Tensor):
                return o.tolist()
        json.dump(captured_data, f, default=tensor_converter)


def generate_model_visualization(model, input_size):
    print("Generating model visualization...")
    model_graph = draw_graph(model, input_size=input_size, expand_nested=True)
    model_graph.visual_graph.render(filename='ReLU/model_relu', format='png')


def main():
    # Create ReLU output directory if it doesn't exist
    os.makedirs('ReLU', exist_ok=True)

    print("Loading CIFAR-10 dataset...")

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root="../data",
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root="../data",
        train=False,
        transform=test_transform,
        download=True,
    )

    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model = ReLUNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_and_evaluate_model(epochs=1, train_loader=train_loader, test_loader=test_loader,
                             model=model, loss_fn=loss_fn, optimizer=optimizer, desired_accuracy=.7)

    get_batch_examples(1, test_loader, model, loss_fn)

    generate_model_visualization(model, input_size=(1, 3, 32, 32))


if __name__ == "__main__":
    main()
