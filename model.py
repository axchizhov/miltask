import torch
import torchvision

from torch import nn



class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(12, 4, 3, stride=2, padding=1)
            )
        
        self.code = None

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 12, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)

        return decoded