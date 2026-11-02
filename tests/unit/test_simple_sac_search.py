import copy
import numpy as np
import pytest
import torch

from app.simple_sac_search import GENES, BASELINE, decode, baseline_genome, combined, make_model, assert_consumed
from agent_plugins.three_signal_sac import ThreeLogits, ThreeSignalSAC
from app.three_signal_learning import Patience
from tests.unit.test_double_tcn_probe import probe_recipe


def test_baseline_roundtrip():
    assert decode(baseline_genome()) == BASELINE


@pytest.mark.parametrize('i', range(len(GENES)))
def test_each_gene_endpoints_reach_real_consumers(i):
    lo, hi = baseline_genome(), baseline_genome()
    lo[i], hi[i] = 0., 1.
    a, b = decode(lo), decode(hi)
    assert [k for k in a if a[k] != b[k]] == [GENES[i][0]]
    for c in (a, b):
        model = make_model(probe_recipe(), c, 7, 'cpu')
        assert_consumed(model, c, c)
        x = torch.zeros(2, 13, 2)
        assert model.surfaces(x).shape == (2,3)
        assert model.q_surfaces(x).shape == (2,2,3)


def test_no_silent_default_substitution():
    config = dict(BASELINE, learning_rate=1e-5)
    model = ThreeSignalSAC(probe_recipe())
    with pytest.raises(ValueError, match='NOT_CONSUMED'):
        assert_consumed(model, config, config)


def test_default_is_explicit_baseline_without_dropout():
    torch.manual_seed(7); old = ThreeLogits(probe_recipe())
    torch.manual_seed(7); new = ThreeLogits(probe_recipe(), {k:BASELINE[k] for k in
        ('branch_width','core_width','branch_blocks','core_blocks','head_width')})
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(x,new.state_dict()[k]) for k,x in old.state_dict().items())
    assert not any(isinstance(m,(torch.nn.Dropout,torch.nn.BatchNorm1d)) for m in new.modules())


def test_checkpoint_effective_config_checked(tmp_path):
    a = make_model(probe_recipe(), BASELINE, 7, 'cpu')
    a.save(tmp_path/'c.pt')
    b = make_model(probe_recipe(), dict(BASELINE, gamma=.95), 7, 'cpu')
    with pytest.raises(ValueError, match='EFFECTIVE_CONFIG'):
        b.load(tmp_path/'c.pt')
    a.load(tmp_path/'c.pt')


def test_combined_patience_not_train_only_or_validation_veto():
    p = Patience(0, 10, 1e-5)
    assert p.observe(combined(1.,1.), True, 1)['improved']
    assert p.observe(combined(3.,.5), True, 2)['improved']
    for i in range(9):
        assert not p.observe(combined(3.1,.2), True, i+3)['stop']
    assert p.observe(combined(3.1,.2), True, 12)['stop']
    assert p.best == 1.75
    p.observe(combined(4.,0.), True, 13)
    assert p.bad == 0


def test_largest_candidate_is_causal():
    config = decode([1.]*len(GENES))
    model = make_model(probe_recipe(), config, 7, 'cpu')
    x = torch.randn(2,13,2); y = x.clone(); y[:,7:] += 100
    for net in (model.actor_net,model.q_net):
        assert torch.equal(net.encoder.sequence({'features':x})[:,:,:7],
                           net.encoder.sequence({'features':y})[:,:,:7])


@pytest.mark.parametrize('x', [-.001, 1.001, float('nan')])
def test_invalid_genotype_rejected(x):
    g = baseline_genome(); g[0] = x
    with pytest.raises(ValueError, match='GENOME_DOMAIN'): decode(g)


def test_evolution_resume_identical_and_bounds():
    import json
    import random
    from tools.run_simple_sac_deap import individual_type, next_population, tuples
    cls=individual_type(); pop=[cls(baseline_genome()) for _ in range(20)]
    for i,x in enumerate(pop): x.fitness.values=(float(i),)
    settings=dict(elite=2,population=20,tournament=3,crossover=.7,mutation=.4)
    random.seed(42); state=json.loads(json.dumps(random.getstate()))
    a=next_population(pop,settings)
    random.setstate(tuples(state)); b=next_population(pop,settings)
    assert a==b and len(a)==20
    assert all(0<=g<=1 for x in a for g in x)
    assert a[0]==pop[-1] and a[1]==pop[-2]
    assert any(decode(x)!=BASELINE for x in a)


def test_receipt_integrity_and_identity(tmp_path):
    from tools.run_simple_sac_deap import finish, verified
    out=tmp_path/'result'; out.mkdir()
    (out/'weights').write_bytes(b'weights')
    finish(out,{'identity':'id'},BASELINE,7,eligible=False,status='test')
    assert verified(out/'RESULT.json','id',BASELINE,7)['status']=='test'
    with pytest.raises(ValueError,match='IDENTITY'):
        verified(out/'RESULT.json','id',BASELINE,19)
    (out/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError,match='ARTIFACT'):
        verified(out/'RESULT.json','id',BASELINE,7)
