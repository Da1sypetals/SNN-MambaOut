# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DATASET_PATH = "~/train_data/fmnist"

# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.FashionMNIST(
    root=DATASET_PATH, train=True, download=True, transform=transform
)
valset = torchvision.datasets.FashionMNIST(
    root=DATASET_PATH, train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
valloader = DataLoader(valset, batch_size=128, shuffle=False)


# import polynomials as layer
# import v1_test as layer
from act import SepLU, sReLU
import sys


act = sys.argv[1]
act_layer = None

if act == "gelu":
    act_layer = nn.GELU()
elif act == "srelu":
    act_layer = sReLU(0.3)
elif act == "seplu":
    act_layer = SepLU(0.3)
else:
    print(f"Unknown activation: {act}")
    exit(1)

model = nn.Sequential(
    nn.Linear(784, 256),
    act_layer,
    nn.Linear(256, 10),
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

num_epoch = 3

# Define loss
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epoch):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28)
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            # print('forward\n')
            loss = criterion(output, labels.to(device))
            loss.backward()
            # print('backward\n')
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()

            _loss = loss.item()
            _acc = accuracy.item()
            _lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{_loss: .3f}", accuracy=f"{_acc: .3f}", lr=f"{_lr: .6f}"
            )
            # print(_loss)

    # Validation
    # model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28)
            images = images.to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
    # print(f'tanh_scale = {model.layer1.tanh_scale.view(-1)[:10]}')
    # print(f'tanh_bias = {model.layer1.tanh_bias.view(-1)[:10]}')
