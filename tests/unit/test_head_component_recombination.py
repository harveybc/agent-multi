import copy

import numpy as np
import pytest
import torch
from torch import nn

from app.head_component_recombination import combine_heads, fit_preservation_gate
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest


def heads(seed):
    return RoleHeads(nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)),1,seed)


def test_components_include_bias_and_never_alias_donors():
    a,b=heads(19).state_dict(),heads(43).state_dict()
    before=(tree_digest(a),tree_digest(b));c=combine_heads(a,b)
    for name,v in c.items():
        donor=a if name.split('.')[3]=='0' else b
        assert torch.equal(v,donor[name]) and v.data_ptr()!=donor[name].data_ptr()
    next(iter(c.values())).zero_()
    assert before==(tree_digest(a),tree_digest(b))


def test_self_combination_and_bilinear_interaction():
    a,b=heads(19),heads(43);z=torch.randn(2,13,208)
    aa=heads(1);aa.load_state_dict(combine_heads(a.state_dict(),a.state_dict()))
    ab=heads(1);ab.load_state_dict(combine_heads(a.state_dict(),b.state_dict()))
    ba=heads(1);ba.load_state_dict(combine_heads(b.state_dict(),a.state_dict()))
    with torch.no_grad():
        assert torch.equal(aa(z),a(z))
        h=lambda m:torch.stack([t[0][1](t[0][0](z[i])) for i,t in enumerate(m.twins)])
        w=lambda m:torch.stack([t[0][2].weight for t in m.twins])
        predicted=torch.einsum('tni,tai->tna',h(b)-h(a),w(b)-w(a))
        actual=(b(z)-ba(z)-ab(z)+a(z)).mean(2)
        assert torch.allclose(actual,predicted,atol=2e-7,rtol=1e-5)


def test_bad_component_contract_is_rejected():
    a=heads(19).state_dict()
    for mutation in ('extra','missing','shape','dtype','nan'):
        b=copy.deepcopy(a);key=next(iter(b))
        if mutation=='extra':b['extra']=torch.ones(1)
        if mutation=='missing':del b[key]
        if mutation=='shape':b[key]=b[key][:-1]
        if mutation=='dtype':b[key]=b[key].double()
        if mutation=='nan':b[key].flatten()[0]=float('nan')
        with pytest.raises(ValueError):combine_heads(a,b)


def test_gate_uses_only_fit_and_tolerance():
    baseline=dict(fit=.1,expanded_fit=.3)
    assert fit_preservation_gate(baseline,dict(fit=.1+1e-9,expanded_fit=.2))
    assert not fit_preservation_gate(baseline,dict(fit=.11,expanded_fit=.1))
    assert not fit_preservation_gate(baseline,dict(fit=.1,expanded_fit=.3-1e-9))
    assert fit_preservation_gate(baseline,dict(fit=.1,expanded_fit=.2,audit=1000))
    with pytest.raises(ValueError):fit_preservation_gate(baseline,dict(fit=np.nan,expanded_fit=.2))
