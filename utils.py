import torch
import matplotlib.pyplot as plt

def plot_reconstructed(images: torch.Tensor, reconstructed: torch.Tensor):
    images = images.permute(0, 2, 3, 1).detach().numpy()
    reconstructed = reconstructed.permute(0, 2, 3, 1).detach().numpy()

    plt.figure(figsize=(9,2))

    for i, img in enumerate(images[:9]):
        plt.subplot(2, 9, i+1)
        plt.imshow(img)

    for i, img in enumerate(reconstructed[:9]):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(img)

    plt.show()