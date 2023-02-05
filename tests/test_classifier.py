import pytest
import torch
import torchvision
import matplotlib.pyplot as plt
import torchmetrics

from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


from models.autoencoder import MyAutoencoder
from models.classifier import MyClassifier
from src.utils import plot_reconstructed, grid_plot, vis_confusion


@pytest.fixture
def classes():
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    classes = {label: i for i, label in enumerate(classes)}
    
    return classes

@pytest.fixture
def data():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Prepare train/val data
    cifar_train = CIFAR10('data/', train=False, download=True, transform=transform)

    val_size = 2000
    train_size= len(cifar_train) - val_size
    torch.manual_seed(42)
    cifar_train, cifar_val = torch.utils.data.random_split(cifar_train, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=1000, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=cifar_val, batch_size=1000)

    return train_dataloader, val_dataloader

@pytest.fixture
def autoencoder():
    autoencoder_weights = 'outputs/autoencoder_model.pth'

    autoencoder = MyAutoencoder()
    autoencoder.load_state_dict(torch.load(autoencoder_weights))

    return autoencoder

def test_classifier(classes, data, autoencoder):
    train_dataloader, val_dataloader = data

    clf = nn.Sequential(nn.Linear(256, 10), nn.ReLU())

    net = MyClassifier(autoencoder, clf, classes, lr=1e-3)
    
    logger = TensorBoardLogger('../outputs', name='tests', version='testing')
    trainer = Trainer(max_epochs=1, logger=logger)

    trainer.fit(net, train_dataloader, val_dataloader)