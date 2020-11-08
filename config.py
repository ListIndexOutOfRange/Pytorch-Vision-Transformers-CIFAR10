from dataclasses import dataclass


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         CONFIG                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class TrainConfig:
    rootdir:               str = './data/'
    train_batch_size:      int = 32
    val_batch_size:        int = 32
    num_workers:           int = 4
    patch_size:            int = 4
    dim:                   int = 1024
    depth:                 int = 6
    heads:                 int = 8
    mlp_dim:               int = 2048
    dropout_rate:          int = 0.1
    emb_dropout_rate:      int = 0.1

