""" Adapted from https://github.com/lucidrains/vit_pytorch. """

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple


MIN_NUM_PATCHES = 16 # minimal patches number for attention to be effective.




# +---------------------------------------------------------------------------------------------+ #
# |                                             BASIC BLOCKS                                    | #
# +---------------------------------------------------------------------------------------------+ #

class Residual(nn.Module):

    """ A simple residual connection. """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(x, **kwargs) + x


class PreNorm(nn.Module):

    """ A simple layer prenormalization.
    That is, Layer normalization right before applying layer.
    """

    def __init__(self, dim: int, layer: nn.Module):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layer(self.norm(x), **kwargs) 


class FeedForward(nn.Module):

    """ A simple feed forward network with one GELU and two Linears. """

    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)




# +---------------------------------------------------------------------------------------------+ #
# |                                        MULTI HEAD ATTENTION                                 | #
# +---------------------------------------------------------------------------------------------+ #

#TODO: check the use of Pytorch MultiheadAttention class instead 
 
class MultiHeadAttention(nn.Module):

    """ Implementation of the Multi Head Attention mechanism as decribed in the paper
    'Attention Is All You Need', by Vaswani et al, 2017. 
    """

    def __init__(self, dim: int, heads: int, dropout_rate: float):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5 # (1/sqrt(dim)): scaling factor
        self.to_qkv  = nn.Linear(dim, dim*3, bias=False)
        self.concat  = Rearrange('b h n d -> b n (h d)')
        self.linear  = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def get_query_key_value(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        return map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

    def scaled_dot_product(self, a, b):
        return torch.einsum('bhid, bhjd -> bhij', a, b) * self.scale

    @staticmethod
    def matmul(a, b):
        return torch.einsum('bhij, bhjd -> bhid', a, b)

    @staticmethod
    def apply_mask(x, mask):
        mask = F.pad(mask.flatten(1), (1,0), value=True)
        mask = mask[:, None, :] * mask[:, :, None]
        x.masked_fill_(~mask, float('-inf'))

    def forward(self, x, mask=None):
        q, k, v = self.get_query_key_value(x)
        dots = self.scaled_dot_product(q, k)
        if mask is not None:
            self.apply_mask(dots, mask)
        attention = dots.softmax(dim=-1)
        out = self.matmul(attention, v)
        out = self.concat(out)
        out = self.linear(out)
        out = self.dropout(out)
        return out
        #return self.dropout(self.linear(self.concat(self.matmul(attention, v))))




# +---------------------------------------------------------------------------------------------+ #
# |                                            TRANSFORMER                                      | #
# +---------------------------------------------------------------------------------------------+ #

class Transformer(nn.Module):

    """ Implementation of the Transformer Encoder as decribed in the paper
    'An Image is Worth 16x16 words: Transformers for image recognition at scale',
     by Dosovitskiy et al, 2020. 
    """

    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Residual(PreNorm(dim, MultiHeadAttention(dim, heads, dropout_rate))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate)))
                ])
            )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for Attention, FeedForward in self.layers:
            x = FeedForward(Attention(x, mask=mask))
        return x




# +---------------------------------------------------------------------------------------------+ #
# |                                        VISION TRANSFORMER                                   | #
# +---------------------------------------------------------------------------------------------+ #

class ViT(nn.Module):

    def __init__(self, image_size: int, patch_size: int, num_classes: int,
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int=3,
                 dropout_rate: float=0., emb_dropout_rate: float=0.):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.check_patch_size(image_size, patch_size, num_patches)
        patch_dim = channels * patch_size ** 2
        self.image_to_patch = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.patch_to_embedding   = nn.Linear(patch_dim, dim)
        self.class_token          = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout              = nn.Dropout(emb_dropout_rate)
        self.transformer          = Transformer(dim, depth, heads, mlp_dim, dropout_rate)
        self.to_class_token       = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, num_classes)
        )

    @staticmethod
    def check_patch_size(image_size, patch_size, num_patches):
        assert image_size % patch_size == 0, "the patch size must be a divisor of the image size."
        num_patches_error = f'your number of patches ({num_patches}) is too small.'
        assert num_patches > MIN_NUM_PATCHES, num_patches_error

    def forward(self, image: torch.Tensor, mask: torch.Tensor =None) -> torch.Tensor:
        x = self.image_to_patch(image)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        class_tokens = repeat(self.class_token, '() n d -> b n d', b=b)
        x = torch.cat((class_tokens, x), dim=1)
        x += self.positional_embedding[:, :(n+1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_class_token(x[:, 0])
        return self.mlp_head(x)

    @classmethod
    def from_config(cls, config):
        return cls(image_size       = 32,
                   channels         = 3,
                   num_classes      = 10,
                   dim              = 1024,
                   patch_size       = config.patch_size,
                   depth            = config.depth,
                   heads            = config.heads,
                   mlp_dim          = config.mlp_dim,
                   dropout_rate     = config.dropout_rate,
                   emb_dropout_rate = config.emb_dropout_rate)