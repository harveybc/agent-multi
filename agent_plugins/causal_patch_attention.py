"""Small channel-independent causal patch adapter, inspired by PatchTST.

This is a sequence-preserving SAC adapter, not the full forecasting model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class CausalPatchAttention(nn.Module):
    def __init__(self, length, patch_len=3, embedding=8, heads=2, feedforward=16,
                 gate_init=.1):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (length,patch_len,embedding,heads,feedforward))
                or patch_len > length or embedding % 2 or embedding % heads
                or not math.isfinite(gate_init)):
            raise ValueError('CAUSAL_PATCH_CONFIG')
        self.length, self.patch_len = length, patch_len
        self.patch_projection = nn.Linear(patch_len, embedding)
        self.encoder = nn.TransformerEncoderLayer(embedding, heads, feedforward,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.output = nn.Linear(embedding, 1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        position = torch.arange(length, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0,embedding,2,dtype=torch.float32)*(-math.log(10000.)/embedding))
        pe = torch.zeros(length,embedding)
        pe[:,0::2] = torch.sin(position*frequency)
        pe[:,1::2] = torch.cos(position*frequency)
        self.register_buffer('positions',pe.unsqueeze(0))
        self.register_buffer('causal_mask',torch.ones(length,length,dtype=torch.bool).triu(1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.length:
            raise ValueError('CAUSAL_PATCH_SHAPE')
        batch, channels, length = x.shape
        # Each token is a trailing patch ending at its own timestamp. Padding
        # is an internal convolution boundary, never an imputed market row.
        patches = F.pad(x,(self.patch_len-1,0)).unfold(-1,self.patch_len,1)
        tokens = self.patch_projection(patches.reshape(batch*channels,length,self.patch_len))
        tokens = tokens + self.positions.to(dtype=tokens.dtype)
        # TorchRL vmaps twin critics; CUDA's fused bias kernel cannot align
        # the 13-token mask in that path. Scope the portable backend here.
        with sdpa_kernel(SDPBackend.MATH):
            encoded = self.encoder(tokens,src_mask=self.causal_mask)
        correction = self.output(encoded).reshape(batch,channels,length)
        return x + self.gate*correction
