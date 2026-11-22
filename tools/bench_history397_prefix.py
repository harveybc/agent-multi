"""Disposable actor-only cache benchmark on different real windows."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tools import pilot391 as p
from tools.account397 import assemble
from agent_plugins.discrete_curve_sac import tensor_observation


def main(dest):
    p.configure(7)
    torch.backends.cudnn.allow_tf32=False
    root=Path('/var/tmp/pilot397')
    bundle=root/'model/61/bundle'
    m,_=p.load_bundle(bundle,'cuda')
    train,_=assemble(Path('/var/tmp/sac70-warm-v3-20261108'),bundle,root/'history397-61-r2')
    enc=m.actor_net.encoder
    assert enc.context is None
    valid=np.flatnonzero(train.inputs.valid)
    ix=valid[np.linspace(0,len(valid)-1,384,dtype=int)]
    obs=[dict(features=train.inputs.windows[i],mask=np.ones(3,bool)) for i in ix]
    with torch.no_grad(), m.loss.actor_network_params.to_module(m.actor):
        assert not any(v.requires_grad for part in (enc.branches,enc.fusion) for v in part.parameters())
        torch.cuda.synchronize()
        t=time.perf_counter()
        cache=[]
        for start in range(0,len(obs),128):
            x=torch.as_tensor(train.inputs.windows[ix[start:start+128]],device=m.device).transpose(1,2)
            grouped=[b(x.index_select(1,getattr(enc,f'indices_{i}'))) for i,b in enumerate(enc.branches)]
            cache.append(enc.fusion_activation(enc.fusion(torch.cat(grouped,1))+enc.fixed_fusion_offset))
        cache=torch.cat(cache)
        torch.cuda.synchronize()
        setup=time.perf_counter()-t
    def cached(_,core_first_norm=None):
        x=enc.post_fusion(cache[index:index+1])
        if core_first_norm is None: return enc.core(x)
        for i,b in enumerate(enc.core): x=b(x,first_norm=core_first_norm if i==0 else None)
        return x
    runs=[]
    from contextlib import nullcontext
    for fast in (False,True,True,False):
        torch.manual_seed(57)
        logits=[]
        actions=[]
        with torch.no_grad(), (patch.object(enc,'sequence',cached) if fast else nullcontext()):
            torch.cuda.synchronize()
            t=time.perf_counter()
            for index,o in enumerate(obs):
                dist=m.distribution(tensor_observation(o,m.device))
                logits.append(dist.logits.detach())
                actions.append(int(dist.sample()[0]))
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
        logits=torch.cat(logits)
        if not runs: ref,draws=logits,actions
        runs.append(dict(cached=fast,seconds=elapsed,max_normalized_logit_delta=float((logits-ref).abs().max()),
                         mismatches=sum(a!=b for a,b in zip(actions,draws)),
                         argmax_mismatches=int((logits.argmax(1)!=ref.argmax(1)).sum())))
    Path(dest).write_text(json.dumps(dict(windows=len(ix),setup_seconds=setup,
        bytes_per_window=cache[0].nbytes,runs=runs),indent=2))


if __name__=='__main__':
    main(sys.argv[1])
