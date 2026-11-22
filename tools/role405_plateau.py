"""Validation-best rollback for independent role heads, not a SAC rollback."""
import copy
import torch

POLICY=dict(initial_lr=3e-4, factor=.2, floor=4.8e-7, patience=20,
            grace_epochs=20, es_first_count=100, es_patience=80, threshold=1e-8)

def snapshot(optim):
    return copy.deepcopy(optim.state_dict())

def observe(epoch, improved, misses, lr, stop):
    if improved:return 0,None
    if epoch<=POLICY['grace_epochs'] or stop or lr<=POLICY['floor']*(1+1e-12):return misses,None
    misses+=1
    if misses<POLICY['patience']:return misses,None
    return 0,max(POLICY['floor'],round(lr*POLICY['factor'],15))

def restore(model,optim,weights,moments,rate):
    assert len(optim.param_groups)==1 and rate<optim.param_groups[0]['lr']
    model.load_state_dict(weights)
    # AdamW may retain references to the tensors passed to load_state_dict.
    optim.load_state_dict(copy.deepcopy(moments))
    optim.param_groups[0]['lr']=rate
    for k,v in model.state_dict().items():assert torch.equal(v.cpu(),weights[k])
    state=optim.state_dict()['state']
    for p,items in moments['state'].items():
        for k,v in items.items():
            assert torch.equal(state[p][k].cpu(),v.cpu())
