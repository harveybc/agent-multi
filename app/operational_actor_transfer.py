"""Gates and exact categorical actor objective for a bounded critic transfer."""
import numpy as np
import torch
from app.operational_opening_critic import validate_labels


def check_lineage(source, support, control, corrected):
    if (source!=control['source_sha256'] or source!=corrected['source_sha256']
            or support!=control['support_sha256'] or control['identity']!=corrected['control']):
        raise ValueError('ACTOR_TRANSFER_LINEAGE')


def fit_indices(records):
    validate_labels(records)
    result=np.array([i for i,r in enumerate(records) if r['role']=='fit' and not r['forbidden']
        and all(s['actions'][a]['first_effect']['entries']>0 for s in r['suffixes'] for a in (1,2))],dtype=int)
    if not len(result):raise ValueError('ACTOR_TRANSFER_EMPTY_FIT')
    return result


def actor_loss(logits, q, alpha):
    if (logits.ndim!=2 or logits.shape[-1]!=3 or logits.shape!=q.shape or not len(logits)
            or alpha.numel()!=1 or not torch.isfinite(alpha).all() or alpha.item()<0
            or not torch.isfinite(logits).all() or not torch.isfinite(q).all()):
        raise ValueError('ACTOR_TRANSFER_OBJECTIVE')
    lp=logits.log_softmax(-1)
    return (lp.exp()*(alpha.detach()*lp-q.detach())).sum(-1).mean()
