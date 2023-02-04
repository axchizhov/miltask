import torch
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from itertools import chain


def plot_reconstructed(images: torch.Tensor, reconstructed: torch.Tensor):
    images = images.permute(0, 2, 3, 1).detach().numpy()
    reconstructed = reconstructed.permute(0, 2, 3, 1).detach().numpy()

    figure = plt.figure(figsize=(9,2))

    for i, img in enumerate(images[:9]):
        plt.subplot(2, 9, i+1)
        plt.imshow(img)

    for i, img in enumerate(reconstructed[:9]):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(img)

    plt.show()

    return figure


def grid_plot(images: torch.Tensor, reconstructed: torch.Tensor):
    chained = list(chain(images[:8], reconstructed[:8]))

    grid = make_grid(chained)

    return grid

