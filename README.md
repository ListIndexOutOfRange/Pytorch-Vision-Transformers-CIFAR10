<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                               MAIN TITLE                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# SPYGLASS     

Pytorch implementation of Vision transformer from [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
Trainable on CIFAR10 only (for now).

Based on [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                          TABLE OF CONTENTS                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# SUMMARY

- [Last Commit Changes Log](#last-commit-changes-log)
- [Installation](#installation)
- [Usage](#usage)
- [Fastai integration](#fastai-integration)

# Last Commit Changes Log

- Initial Commit


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              INSTALLATION                                          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Installation
[(Back to top)](#summary)

Clone repo:

```git clone https://github.com/the-dharma-bum/Pytorch-Vision-Transformers-CIFAR10```

Install dependancies by running: 

``` pip install -r requirements.txt ```


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                 USAGE                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Usage
[(Back to top)](#summary)

All hyperparameters can be configured in config.py
One can setup a training config inside this file and then simply run:

```python main.py ```

This command accepts a huge number of parameters. Run 

```python main.py -h ```

to see them all, or refer to [documentation de Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Some usefull parameters:

- ```--gpus n```: launch training on n gpus
- ```--distributed_backend ddp``` : use DistributedDataParallel as multi gpus training backend.
- ```--fast_dev_run True``` : launch a training loop (train, eval, test) on a single batch. Use it to debug.

If fast_dev_run doesn't suit your debugging need (for instance if you wanna see what's happening between two epochs), 
you can use:

- ```--limit_train_batches i --limit_val_batches j --max_epochs k```
     
     i,j,k being of course three integers of your choice.




# Fastai Integration
[(Back to top)](#summary)

It's very easy to integrate a Lighning code into the Fastai training environnement.

One must define

- a model (see model.py):
```python
import config as cfg
from model import LightningModel

model = LightningModel(cfg.TrainConfig())
```

- a datamodule:
```python
from pl_bolts.datamodules import CIFAR10DataModule

dm = CIFAR10DataModule(args, kwargs)
```

Using this datamodule, two fastai DataLoaders can be defined like this:
```python
from fastai.vision.all import DataLoaders

data = Dataloaders(dm.train_dataloader(), dm.val_dataloader()).cuda()
```

Then a Learner can be defined and used like a standart Fastai code, for instance:
```python
learn = Learner(data, model, loss_func=F.cross_entropy, opt_func=Adam, metrics=accuracy)
learn.fit_one_cycle(1, 0.001)
```

This makes every fastai training fonctionalites availables (callbacks, transforms, visualizations ...).