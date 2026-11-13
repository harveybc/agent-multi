"""Retain the matching native head without changing the second twin's RNG seed."""
from copy import deepcopy

from torch import nn


def native_from_ensemble(ensemble):
    if len(ensemble.twins)!=2 or any(len(twin)!=3 for twin in ensemble.twins):
        raise ValueError('NATIVE_CAPACITY_SOURCE')
    native=deepcopy(ensemble)
    native.twins=nn.ModuleList([nn.ModuleList([twin[0]]) for twin in native.twins])
    return native
