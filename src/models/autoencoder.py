import torch
import torchvision
import torch.nn.functional as F

from torch import nn


class MyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)

        return decoded
