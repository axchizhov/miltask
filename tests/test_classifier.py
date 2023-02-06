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

from src.models.classifier import MyClassifier


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


def test_classifier_forward(data):
    _, val_dataloader = data

    imgs = next(iter(val_dataloader))[0][:2]

    net = MyClassifier()

    y_pred = net(imgs)
    assert list(y_pred.shape) == [2, 10]


# def test_classifier_train(data):
#     train_dataloader, val_dataloader = data

#     net = MyClassifier()
    
#     logger = TensorBoardLogger('run_test', name='tests', version='testing')
#     trainer = Trainer(max_epochs=1, logger=logger)

#     trainer.fit(net, train_dataloader, val_dataloader)

#     # todo: remove run_test folder
