"""Small grouped context encoder, aligned only to preceding latent slots."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agent_plugins.temporal_features import CausalConv1d


class LongContext(nn.Module):
    def __init__(self,recipe,config,core_width,block_class):
        super().__init__()
        self.branches=nn.ModuleList()
        width=config['branch_width']
        g=recipe['representation']['resolved']
        for i,(indices,kind) in enumerate(zip(g['indices'],g['kinds'])):
            self.register_buffer(f'indices_{i}',torch.tensor(indices,dtype=torch.long))
            kernel=3 if kind=='conv' else 1
            self.branches.append(nn.Sequential(CausalConv1d(len(indices),width,1),
                *[block_class(width,2**j,kernel) for j in range(config['branch_blocks'])]))
        self.projection=CausalConv1d(len(self.branches)*width,core_width,1)
        nn.init.zeros_(self.projection.conv.weight)
        if self.projection.conv.bias is not None: nn.init.zeros_(self.projection.conv.bias)
        w=recipe['tensor']['window']
        short=-np.arange(w['observations']-1,-1,-1)*w['stride_rows']
        long=-np.arange(config['observations']-1,-1,-1)*config['stride_rows']
        alignment=np.searchsorted(long,short,side='right')-1
        if (alignment<0).any(): raise ValueError('CONTEXT_SPAN')
        self.register_buffer('alignment',torch.tensor(alignment,dtype=torch.long))

    def forward(self,x):
        available=torch.isfinite(x).all(dim=-1).all(dim=-1)
        # Padding is internal to a disabled branch, never an imputed source value.
        safe=torch.where(available[:,None,None],x,torch.zeros_like(x)).transpose(1,2)
        grouped=[branch(safe.index_select(1,getattr(self,f'indices_{i}')))
                 for i,branch in enumerate(self.branches)]
        latent=self.projection(F.silu(torch.cat(grouped,dim=1)))
        aligned=latent.index_select(2,self.alignment)
        return aligned*available[:,None,None]
