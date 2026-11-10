import copy
import pytest
from app.operational_opening_critic import validate_labels,BOUNDARY,END


def records():
    return [dict(role=role,timestamp=end-100,terminal=end,window_start=start,position=0,forbidden=True,
        suffixes=[dict(actions=[dict(soft_return=.1)]*3) for _ in range(3)])
        for role,start,end in [('fit',0,BOUNDARY),('monitor',BOUNDARY,END)]]


def test_fit_return_tail_never_reaches_monitor():
    r=records();assert validate_labels(r)
    r[0]['terminal']=END
    with pytest.raises(ValueError,match='SUPPORT'):validate_labels(r)


def test_equivalent_actions_and_real_position_gates():
    r=records();r[0]['position']=1
    with pytest.raises(ValueError):validate_labels(r)
    r=copy.deepcopy(records());r[0]['suffixes'][0]['actions']=[dict(soft_return=.1),dict(soft_return=.2),dict(soft_return=.1)]
    with pytest.raises(ValueError,match='EQUIVALENT'):validate_labels(r)


@pytest.mark.parametrize('fail',[False,True])
def test_reference_inference_restores_flags_even_on_failure(monkeypatch,fail):
    import torch
    from types import SimpleNamespace
    from app.operational_opening_critic import emit_actor_reference
    from app import three_signal_learning
    layer=torch.nn.Linear(2,3);layer.weight.requires_grad_(False)
    previous={n:p.requires_grad for n,p in layer.named_parameters()}
    model=SimpleNamespace(loss=layer)
    target={n:True for n in previous}
    before={n:p.detach().clone() for n,p in layer.named_parameters()}
    def emitted(m,inputs,indices):
        assert all(p.requires_grad for p in m.loss.parameters())
        if fail:raise RuntimeError('probe')
        return 'exact'
    monkeypatch.setattr(three_signal_learning,'emit',emitted)
    if fail:
        with pytest.raises(RuntimeError,match='probe'):emit_actor_reference(model,None,None,target)
    else:assert emit_actor_reference(model,None,None,target)=='exact'
    assert previous=={n:p.requires_grad for n,p in layer.named_parameters()}
    assert all(torch.equal(v,dict(layer.named_parameters())[n]) for n,v in before.items())
