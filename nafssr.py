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
    train_size: List = None, 40, 100, 3
    base_rate: float = 1.5

    def setup(self):
        self.intro = nn.Conv(
            self.n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME'
        )

        kh, kw = int(self.train_size[1] * self.base_rate), int(self.train_size[2] * self.base_rate)
        self.middles = nn.Sequential([
            NAFBlockSR(
                self.n_filters,
                kh, kw,
                (self.fusion_from <= i <= self.fusion_to),
                1. - self.stochastic_depth_rate
            ) for i in range(self.n_blocks)
        ])

        self.end = nn.Sequential([
            nn.Conv(
                3 * (self.upscale_rate ** 2),
                (3, 3),
                (1, 1),
                padding='SAME'
            ),
            PixelShuffle(self.upscale_rate)
        ])

    def __call__(self, inputs: Sequence, training: bool = True):
        B, H, W, C = inputs[0].shape

        features = [
            self.intro(i) for i in inputs
        ]
        features = self.middles(features)
        recons = [
            self.end(f) for f in features
        ]
        recons = [
            jax.image.resize(i,
                             (B, H * self.upscale_rate, W * self.upscale_rate, C),
                             method='bilinear'
                             ) + r for i, r in zip(inputs, recons)
        ]
        return recons
