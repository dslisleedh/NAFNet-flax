from einops import rearrange
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List


'''
class PlainBlock(nn.Module):
    n_filters: int
    dw_expansion_rate: int = 2
    ffn_expansion_rate: int = 2

    @nn.compact
    def __call__(self, x):
        dw_filters = self.n_filters * self.dw_expansion_rate
        ffn_filters = self.n_filters * self.ffn_expansion_rate

        spatial = nn.Conv(dw_filters,
                          kernel_size=(1, 1),
                          padding='VALID'
                          )(x)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(3, 3),
                          padding='SAME',
                          feature_group_count=dw_filters
                          )(spatial)
        spatial = nn.relu(spatial)
        spatial = nn.Conv(self.n_filters,
                          kernel_size=(1, 1),
                          padding='VALID'
                          )(spatial)
        x = spatial + x

        channel = nn.Dense(ffn_filters)(x)
        channel = nn.relu(channel)
        channel = nn.Dense(self.n_filters)(channel)
        x = channel + x
        return x


class BaselineBlock(nn.Module):
    n_filters: int
    dw_expansion_rate: int = 2
    ffn_expansion_rate: int = 2
    r: int = 4

    @nn.compact
    def __call__(self, x):
        dw_filters = self.n_filters * self.dw_expansion_rate
        ffn_filters = self.n_filters * self.ffn_expansion_rate

        spatial = nn.LayerNorm()(x)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(1, 1),
                          padding='VALID',
                          )(spatial)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(3, 3),
                          padding='SAME',
                          feature_group_count=dw_filters
                          )(spatial)
        spatial = nn.gelu(spatial)
        spatial_gap = jnp.mean(spatial,
                               axis=(1, 2),
                               keepdims=True
                               )
        spatial_attention = nn.relu(nn.Dense(ffn_filters // self.r)(spatial_gap))
        spatial_attention = nn.sigmoid(nn.Dense(ffn_filters)(spatial_attention))
        spatial = spatial * spatial_attention
        spatial = nn.Conv(self.filters,
                          kernel_size=(1, 1),
                          padding='VALID'
                          )(spatial)
        x = x + spatial

        channel = nn.LayerNorm()(x)
        channel = nn.Dense(ffn_filters)(channel)
        channel = nn.gelu(channel)
        channel = nn.Dense(self.filters)(channel)
        x = x + channel
        return x
'''


class PixelShuffle(nn.Module):
    upsample_rate: int

    @nn.compact
    def __call__(self, x):
        return rearrange(x, ('b h w (hc wc c) -> b (h hc) (w wc) c'),
                         hc=self.upsample_rate, wc=self.upsample_rate
                         )


class NAFBlock(nn.Module):
    n_filters: int
    dropout_rate: float
    kh: int
    kw: int
    dw_expansion_rate: int = 2
    ffn_expansion_rate: int = 2

    @nn.compact
    def __call__(self, x, deterministic=False):
        dw_filters = self.n_filters * self.dw_expansion_rate
        ffn_filters = self.n_filters * self.ffn_expansion_rate
        beta = self.param('beta',
                          nn.initializers.zeros,
                          (1, 1, 1, self.n_filters)
                          )
        gamma = self.param('gamma',
                           nn.initializers.zeros,
                           (1, 1, 1, self.n_filters)
                           )

        spatial = nn.LayerNorm()(x)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(1, 1),
                          padding='VALID'
                          )(spatial)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(3, 3),
                          padding='SAME',
                          feature_group_count=dw_filters
                          )(spatial)
        spatial_gate1, spatial_gate2 = jnp.split(spatial,                                  # simple gate
                                                 indices_or_sections=2,
                                                 axis=-1
                                                 )
        spatial = spatial_gate1 * spatial_gate2
        if deterministic:
            b, h, w, c = x.shape                                                           # TLSC (https://arxiv.org/pdf/2112.04491v2.pdf)
            s = spatial.cumsum(2).cumsum(1)
            s = jnp.pad(s,
                        [[0, 0], [1, 0], [1, 0], [0, 0]]
                        )
            kh, kw = min(h, self.kh), min(w, self.kw)
            s1, s2, s3, s4 = s[:, :-kh, :-kw, :],\
                             s[:, :-kh, kw:, :],\
                             s[:, kh:, :-kw, :],\
                             s[:, kh:, kw:, :]
            spatial_gap = (s4 + s1 - s2 - s3) / (kh * kw)
            if (kh != h) and (kw != w):
                _, h_s, w_s, _ = spatial_gap.shape
                h_pad, w_pad = [(h - h_s) // 2, (h - h_s + 1) // 2], [(w - w_s) // 2, (w - w_s + 1) // 2]
                spatial_gap = jnp.pad(spatial_gap,
                                      [[0, 0], h_pad, w_pad, [0, 0]],
                                      mode='edge'
                                      )
        else:
            spatial_gap = jnp.mean(spatial,                                                # simple attention
                                   axis=(1, 2),
                                   keepdims=True
                                   )
        spatial_attention = nn.Dense(self.n_filters)(spatial_gap)
        spatial = spatial * spatial_attention
        spatial = nn.Conv(self.n_filters,
                          kernel_size=(1, 1),
                          padding='VALID'
                          )(spatial)
        spatial = nn.Dropout(self.dropout_rate)(spatial, deterministic=deterministic)
        x = x + beta * spatial

        channel = nn.LayerNorm()(x)
        channel = nn.Dense(ffn_filters)(channel)
        channel_gate1, channel_gate2 = jnp.split(channel,                                  # simple gate
                                                 indices_or_sections=2,
                                                 axis=-1
                                                 )
        channel = channel_gate1 * channel_gate2
        channel = nn.Dense(self.n_filters)(channel)
        channel = nn.Dropout(self.dropout_rate)(channel, deterministic=deterministic)
        x = x + gamma * channel
        return x


# ------------------------------------------- NAFSSR -------------------------------------------


class SCAM(nn.Module):
    n_filters: int
    scale: float

    @nn.compact
    def __call__(self, feats):
        x_l, x_r = feats
        beta = self.param('beta',
                          nn.initializers.zeros,
                          (1, 1, 1, self.n_filters)
                          )
        gamma = self.param('gamma',
                           nn.initializers.zeros,
                           (1, 1, 1, self.n_filters)
                           )

        q_l = nn.Dense(self.n_filters)(nn.LayerNorm()(x_l))
        q_r_t = nn.Dense(self.n_filters)(nn.LayerNorm()(x_r)).transpose(0, 1, 3, 2)

        v_l = nn.Dense(self.n_filters)(x_l)
        v_r = nn.Dense(self.n_filters)(x_r)

        attention = jnp.matmul(q_l, q_r_t) * self.scale
        f_r2l = jnp.matmul(nn.softmax(attention, axis=-1), v_r) * beta
        f_l2r = jnp.matmul(nn.softmax(attention.transpose(0, 1, 3, 2), axis=-1), v_l) * gamma
        return [x_l + f_r2l, x_r + f_l2r]


class NAFBlockSR(nn.Module):
    n_filters: int
    kh: int
    kw: int
    fusion: bool
    survival_prob: float

    def setup(self):
        self.block = NAFBlock(self.n_filters, 0., self.kh, self.kw)
        self.scam = SCAM(self.n_filters, self.n_filters ** -.5)

    def forward(self, feats):
        feats = [f + self.block(f) for f in feats]
        if self.fusion:
            feats = self.scam(feats)
        return feats

    def __call__(self, feats, deterministic):
        if self.survival_prob == 0.:
            return feats
        elif (self.survival_prob == 1.) or deterministic:
            return self.forward(feats)

        rng = self.make_rng('droppath')
        survival_state = jax.random.bernoulli(rng, self.survival_prob, shape=(1,))[0]
        if survival_state:
            feats_ = self.forward(feats)
            return [f + (f_ - f) / self.survival_prob for f, f_ in zip(feats, feats_)]
        else:
            return feats
