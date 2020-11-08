""" Base Model Class: A Lighning Module
This class implements all the logic code and will be the one to be fit by a Trainer.
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from vit import ViT
from typing import Tuple, Dict


class LightningModel(pl.LightningModule):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.net       = ViT.from_config(config)
        self.criterion = CrossEntropyLoss()
        self.accuracy  = pl.metrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self) -> Dict:
        optimizer = SGD(self.net.parameters(),
                        lr           = 0.001, 
                        momentum     = 0.9,
                        nesterov     = True,
                        weight_decay = 5e-4)
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode     = 'min',
                                      factor   = 0.2,
                                      patience = 5,
                                      verbose  = True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        inputs, targets = batch
        outputs = self(inputs)
        loss    = self.criterion(outputs, targets)
        acc     = self.accuracy(outputs,  targets)
        self.log('Loss/Train', loss)
        self.log('Accuracy/Train', acc, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        inputs, targets = batch
        outputs = self(inputs)
        loss    = self.criterion(outputs, targets)
        acc     = self.accuracy(outputs,  targets)
        self.log('Loss/Validation', loss)
        self.log('Accuracy/Validation', acc, logger=True)
        return {'val_loss': loss, 'acc': acc}

    def test_step(self, batch: torch.Tensor, batch_idx) ->  torch.Tensor:
        """ Not implemented. """