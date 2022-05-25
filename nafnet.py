from einops import rearrange
from typing import List
from layers import *
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class NAFNet(nn.Module):
    n_filters: int = 16
    n_enc_blocks: List = 1, 1, 1, 28
    n_middle_blocks: int = 1
    n_dec_blocks: List = 1, 1, 1, 1
    dropout_rate: float = .1
    train: bool = False

    @nn.compact
    def __call__(self, x):
        n_stages = len(self.n_enc_blocks)

        features = nn.Conv(self.n_filters,
                           kernel_size=(3, 3),
                           padding='SAME'
                           )(x)
        enc_skip = []
        for i, n_blocks in enumerate(self.n_enc_blocks):
            for _ in range(n_blocks):
                features = NAFBlock(self.n_filters * (2 ** i),
                                    self.dropout_rate
                                    )(features, deterministic=not self.train)
            enc_skip.append(features)
            features = nn.Conv(self.n_filters * (2 ** (i + 1)),
                               kernel_size=(2, 2),
                               strides=(2, 2),
                               padding='VALID'
                               )(features)
        enc_skip = enc_skip[::-1]

        for _ in range(self.n_middle_blocks):
            features = NAFBlock(self.n_filters * (2 ** n_stages),
                                self.dropout_rate
                                )(features, deterministic=not self.train)
        for i, n_blocks in enumerate(self.n_dec_blocks):
            features = nn.Conv(self.n_filters * (2 ** (n_stages - i)) * 2,
                               kernel_size=(1, 1),
                               padding='VALID'
                               )(features)
            features = PixelShuffle(2)(features)
            features = features + enc_skip[i]
            for _ in range(n_blocks):
                features = NAFBlock(self.n_filters * (2 ** (n_stages - (i + 1))),
                                    self.dropout_rate
                                    )(features, deterministic=not self.train)

        x_res = nn.Conv(3,
                        kernel_size=(3, 3),
                        padding='SAME'
                        )(features)
        x = x + x_res
        return x



