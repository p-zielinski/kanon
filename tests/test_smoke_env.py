import torch
from torch import nn


def test_torch_import():
    layer = nn.Linear(8, 8)
    assert tuple(layer.weight.shape) == (8, 8)
