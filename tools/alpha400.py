"""Single native update intervention on 8y-s19, using an actual saved TRAIN minibatch."""
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from tools import pilot391 as p
from tools.account397 import assemble


def run():
    root=Path('/var/tmp/sac399-series')
    folder=Path('/var/tmp/diag400/8y-s19/recovery/epoch_0015')
    bundle=root/'model/61/bundle'
    p.configure(7)
    train,_=assemble(Path('/var/tmp/sac70-warm-v3-20261108'),bundle,root/'history397-61-r2')
    rows=np.flatnonzero(train.inputs.valid)
    replay=torch.load(folder/'learning.pt',map_location='cpu',weights_only=False)['replay']['_storage']
    td=TensorDict().load_state_dict(replay['_storage'],strict=False)
    td=td[np.random.default_rng(57).choice(replay['_len'],128,replace=False)].clone()
    for k in ('features',('next','features')):
        td[k]=torch.tensor(train.inputs.windows[rows[td[k].flatten().numpy()]])
    out={}
    for enabled in (False,True):
        m,_=p.load_bundle(bundle,'cpu')
        m.load(folder/'model.pt')
        a=m.loss.log_alpha
        assert not a.requires_grad and any(a is q for g in m.optimizer.param_groups for q in g['params'])
        a.requires_grad_(enabled)
        before=a.detach().exp().item()
        torch.manual_seed(17)
        m.update(td.clone())
        out[str(enabled)]=dict(before=before,after=a.detach().exp().item(),
                               clipped_grad=None if a.grad is None else a.grad.item())
    if out['False']['before']!=out['False']['after'] or not out['True']['after']>out['True']['before']:
        raise AssertionError(out)
    (folder.parent.parent/'ALPHA.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__': run()
