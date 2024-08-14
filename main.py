import struct
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from fastai.vision.all import *
import matplotlib.pyplot as plt
import wandb
import gc

# %matplotlib inline

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load and parse
def read_gnt_file(file_path):
    with open(file_path, "rb") as f:
        while f.readable():
            try:
                sample_size = struct.unpack("<I", f.read(4))[0]
                tag_code = f.read(2).decode("gb2312")
                width = struct.unpack("<H", f.read(2))[0]
                height = struct.unpack("<H", f.read(2))[0]
                bitmap = np.frombuffer(f.read(width * height), dtype=np.uint8).reshape(
                    (height, width)
                )

                yield tag_code, bitmap
            except Exception as e:
                break


def calculate_padding(height, width, target_height=64, target_width=64):
    scale = min(target_height / height, target_width / width)
    scaled_height = int(height * scale)
    scaled_width = int(width * scale)

    padding_top = (target_height - scaled_height) // 2
    padding_bottom = target_height - scaled_height - padding_top
    padding_left = (target_width - scaled_width) // 2
    padding_right = target_width - scaled_width - padding_left

    return padding_left, padding_right, padding_top, padding_bottom


def create_transform(height, width, target_size=(64, 64)):
    target_height, target_width = target_size
    padding_left, padding_top, padding_right, padding_bottom = calculate_padding(
        height, width, target_height, target_width
    )

    transform = transforms.Compose(
        [
            # transforms.Pad(
            #    (padding_left, padding_top, padding_right, padding_bottom), fill=255
            # ),
            transforms.Resize((64, 64)),
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    return transform


def preprocess_image(image):
    height, width = image.shape
    image = Image.fromarray(image)
    transform = create_transform(height, width, (64, 64))
    image = transform(image)

    return image


class HWDBDataset(Dataset):
    def __init__(self, gnt_files):
        self.data = []
        self.labels = []
        for file_path in gnt_files:
            for tag_code, image in read_gnt_file(file_path):
                processed_image = preprocess_image(image)
                self.data.append((processed_image))
                self.labels.append(tag_code)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label


def get_dataloader(gnt_files, shuffle=True, config=None):
    batch_size = config["batch_size"]
    dataset = HWDBDataset(gnt_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class Network(nn.Module):
    def __init__(self, num_classes, config=None):
        """
        Divide network into 5 blocks of layers
        4 blocks of Conv + BatchNorm + ReLU + Pooling
        1 block of fully connected layers
        """
        super(Network, self).__init__()

        self.config = config

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.bn8 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(256 * 1 * 1, 1024)
        self.dropout1 = nn.Dropout(p=self.config["dropout"])
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(p=self.config["dropout"])
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy}%")

        wandb.log(
            {
                "Epoch": epoch + 1,
                "Loss": running_loss / len(train_loader),
                "Accuracy": val_accuracy,
            }
        )


def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        train_files = list(Path("./data/train").rglob("*.gnt"))
        test_files = list(Path("./data/test").rglob("*.gnt"))
        train_dataset = HWDBDataset(train_files)
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )
        test_dataset = HWDBDataset(test_files)
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )
        num_classes = len(train_dataset.label_encoder.classes_)
        dls = DataLoaders(train_loader, test_loader)
        model = Network(num_classes=num_classes, config=config).to(device)
        wandb.watch(model, log="all")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        train(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs=config["epochs"],
        )
        torch.save(model.state_dict(), "model.pth")

        torch.mps.empty_cache()
        gc.collect()


if __name__ == "__main__":

    # wandb.init(
    #    project="Chinese-Handwriting-Recognition",
    #    config={
    #        "learning_rate": 0.0001,
    #        "architecture": "CNN",
    #        "dataset": "HWDB",
    #        "epochs": 10,
    #        "batch_size": 32,
    #    },
    # )

    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "Accuracy",
            "goal": "maximize",
        },
        "parameters": {
            "learning_rate": {"values": [0.0001, 0.001, 0.01]},
            "batch_size": {"values": [32, 64]},
            "epochs": {"values": [10]},
            "dropout": {"values": [0.1, 0.3, 0.5, 0.7]},
            "architecture": {"values": ["CNN"]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="Chinese-Handwriting-Recognition")
    wandb.agent(sweep_id, function=train_model, count=10)

    wandb.finish()

    # for i in range(10):
    #    image, label = train_dataset[i]
    #    print(f"Image size (HxW): {image.size(1)}x{image.size(2)}")
    #    image_pil = transforms.ToPILImage()(image.squeeze(0))
    #    image_pil.show()
