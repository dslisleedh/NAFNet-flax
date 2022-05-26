from einops import rearrange
from typing import List
from layers import *
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


'''
Incomplete
'''


class NAFSSR(nn.Module):
    n_filters: int
    stochastic_depth_rate

