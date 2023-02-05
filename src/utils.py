import torch
import matplotlib.pyplot as plt
import numpy as np

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





import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def vis_confusion(writer, tag, step, matrix: torch.Tensor, class_dict):
    """
    Visualization of confusion matrix

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped array of size class x class.
            Should specify cross-class accuracies/confusion in percent
            values (range 0-1).
        class_dict (dict): Dictionary specifying class names as keys and
            corresponding integer labels/targets as values.
    """

    all_categories = sorted(class_dict, key=class_dict.get)

    # Normalize by dividing every row by its sum
    matrix = matrix.numpy().astype(float)
    for i in range(len(class_dict)):
        matrix[i] = matrix[i] / matrix[i].sum()

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Show the matrix and define a discretized color bar
    cax = ax.matshow(matrix)
    fig.colorbar(cax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Set up axes. Rotate the x ticks by 90 degrees.
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Turn off the grid for this plot. Enforce a tight layout to reduce white margins
    ax.grid(False)
    plt.tight_layout()

    writer.add_figure(f"Confusion matrix: {tag}", fig, step)


if __name__ == '__main__':
    labels = torch.Tensor([7, 3, 4, 6, 5, 9, 0, 8, 4, 9, 3, 4, 2, 9, 4, 4])
    y_pred = torch.Tensor([7, 6, 6, 6, 4, 8, 0, 8, 6, 9, 1, 0, 3, 9, 1, 4])

    import torchmetrics
    from utils import vis_confusion 

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    classes = {label: i for i, label in enumerate(classes)}


    matrix = torchmetrics.ConfusionMatrix('multiclass', num_classes=10)

    matrix = matrix(y_pred, labels).numpy()

    
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter()

    vis_confusion(tb, 1, matrix, classes)

