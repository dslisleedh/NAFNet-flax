from typing import Sequence
from layers import *
import jax
import flax.linen as nn


class NAFSSR(nn.Module):
    n_filters: int = 48
    stochastic_depth_rate: float = .1
    n_blocks: int = 16
    upscale_rate: int = 4
    fusion_from: int = -1
    fusion_to: int = 1000
    is_training: bool = True

    @nn.compact
    def __call__(self, inputs: Sequence):
        B, H, W, C = inputs[0].shape

        features = [
            nn.Conv(self.n_filters,
                    (3, 3),
                    (1, 1),
                    padding='SAME'
                    )(f) for f in inputs
        ]

        for i in range(self.n_blocks):
            features_res = NAFBlockSR(self.n_filters,
                                      fusion=(self.fusion_from <= i <= self.fusion_to)
                                      )(features)
            features_res = DropPath(1. - self.stochastic_depth_rate)(features_res, not self.is_training)
            features = [f + f_r for f, f_r in zip(features, features_res)]

        recons = [
            PixelShuffle(self.upscale_rate)(
                nn.Conv(3 * (self.upscale_rate ** 2),
                        (3, 3),
                        (1, 1),
                        padding='SAME'
                        )(f)
            ) for f in features
        ]
        recons = [
            jax.image.resize(i,
                             (B, H * self.upscale_rate, W * self.upscale_rate, C),
                             method='bilinear'
                             ) + r for i, r in zip(inputs, recons)
        ]
        return recons
