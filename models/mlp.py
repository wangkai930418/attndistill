import torch
import torch.nn as nn
from PIL import Image

import numpy as np
import torch
from torch.autograd import Function


def get_mlp_student(hidden_dims, n_layers):

    layers = []
    layers.append(nn.LayerNorm(hidden_dims[0], eps=1e-6))
    for i in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.GELU())

    layers.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
    mlp = nn.Sequential(*layers)
    return mlp

