# Adapted from https://github.com/lucidrains/egnn-pytorch

import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None

def safe_div(num, den, eps=1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res 

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def huber_reconstruction_loss(coors, coors_hat):
    orig_pair_distance_matrices = torch.cdist(coors, coors)
    pred_pair_distance_matrices = torch.cdist(coors_hat, coors_hat)
    huber_loss = nn.HuberLoss()

    return huber_loss(orig_pair_distance_matrices, pred_pair_distance_matrices)
