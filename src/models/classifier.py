import torch
import torch.nn.functional as F
import torchmetrics

from pytorch_lightning import LightningModule
from torch import nn

from src.models.autoencoder import MyAutoencoder
from src.utils import vis_confusion


AUTOENCODER_WEIGHTS = 'models/autoencoder_without_love.pth'
CLASSIFIER_WEIGHTS = 'models/clf_model.pth'
LR = 1e-3

CLASSESS = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


class MyClassifier(LightningModule):
    def __init__(self):
        super().__init__()

        self._autoencoder = MyAutoencoder()
        self._autoencoder.load_state_dict(torch.load(AUTOENCODER_WEIGHTS))

        self._autoencoder.requires_grad_(False)
        self.encoder = self._autoencoder.encoder

        self.clf = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 10)
            )

        self.classes = CLASSESS
        self.lr = LR

        # Quality metrics
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=len(self.classes))
        self.conf_matrix = torchmetrics.ConfusionMatrix('multiclass', num_classes=len(self.classes))

    def forward(self, x):
        # Freeze the encoder
        self.encoder.eval()
        with torch.no_grad():
            encoded = self.encoder(x).flatten(1)
        
        x = self.clf(encoded)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)

        # Logs
        self.logger.experiment.add_scalars('Loss', 
                                           {'train loss': loss}, 
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('Accuracy', 
                                           {'train accuracy': self.accuracy(y_pred, y)}, 
                                           global_step=self.global_step)

        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y)


        # Logs
        self.logger.experiment.add_scalars('Loss', 
                                           {'val loss': loss}, 
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('Accuracy', 
                                           {'val accuracy': self.accuracy(y_pred, y)}, 
                                           global_step=self.global_step)
        matrix = self.conf_matrix(y_pred, y)
        vis_confusion(self.logger.experiment, 'val', self.global_step, matrix, self.classes)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)

        # Logs
        self.logger.experiment.add_scalars('Accuracy', 
                                           {'test accuracy': self.accuracy(y_pred, y)}, 
                                           global_step=self.global_step)
        matrix = self.conf_matrix(y_pred, y)
        vis_confusion(self.logger.experiment, 'test', self.global_step, matrix, self.classes)
