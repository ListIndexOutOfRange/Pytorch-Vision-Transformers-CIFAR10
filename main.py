""" Main Python file to start training using config.py """

from argparse import ArgumentParser
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import config as cfg
from model import LightningModel


def init_trainer():
    """ Init a Lightning Trainer using from_argparse_args
    Thus every CLI command (--gpus, distributed_backend, ...) become available.
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()
    lr_logger      = LearningRateMonitor()
    early_stopping = EarlyStopping(monitor   = 'val_loss',
                                   mode      = 'min',
                                   min_delta = 0.001,
                                   patience  = 10,
                                   verbose   = True)
    return Trainer.from_argparse_args(args, callbacks = [lr_logger, early_stopping])


def run_training(config):
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data) """
    data    = CIFAR10DataModule(config.rootdir)
    model   = LightningModel(config)
    trainer = init_trainer()
    trainer.fit(model, data)


if __name__ == '__main__':
    run_training(cfg.TrainConfig()) 

