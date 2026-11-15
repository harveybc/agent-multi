import copy

import pytest
import torch

from app.architecture_gene_contract import set_at, value_at
from app.modular_gene_contract import make_contract, decode_modular, baseline_genome
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_architecture_gene_contract import SPACE, baseline
from tests.unit.test_modular_temporal_architecture import grouped_recipe


def test_all_gene_boundaries_and_choices_consumed_or_explicitly_inactive():
    r=grouped_recipe();space,base=make_contract(SPACE,baseline(),r)
    raw=baseline_genome(space,base)
    assert not set(space['shadowed_shared_paths'])&{g['path'] for g in space['genes']}
    for i,g in enumerate(space['genes']):
        points=[0.,1.] if g['kind']!='choice' else [(j+.5)/len(g['values']) for j in range(len(g['values']))]
        for x in points:
            trial=list(raw);trial[i]=x
            out=decode_modular(space,trial,base,r)
            assert len(out['active_gene_mask'])==len(raw)
            if out['active_gene_mask'][i]:
                assert value_at(out['requested'],g['path'])==value_at(out['effective'],g['path'])
            else:
                assert any(row['path']==g['path'] for row in out['inactive_genes'])


def test_default_decode_preserves_model_parameters_and_logits():
    torch.set_num_threads(1)
    r=grouped_recipe();old=baseline();space,base=make_contract(SPACE,old,r)
    decoded=decode_modular(space,baseline_genome(space,base),base,r)
    a=ThreeSignalSAC(r,architecture=old['architecture'])
    b=ThreeSignalSAC(r,architecture=decoded['effective']['architecture'])
    assert list(a.loss.state_dict())==list(b.loss.state_dict())
    assert all(torch.equal(v,b.loss.state_dict()[k]) for k,v in a.loss.state_dict().items())
    x=torch.randn(2,13,2)
    with torch.no_grad():assert torch.equal(a.surfaces(x),b.surfaces(x))


def test_pinned_group_order_and_invalid_list_paths():
    r=grouped_recipe();s,b=make_contract(SPACE,baseline(),r);raw=baseline_genome(s,b)
    bad=copy.deepcopy(r);bad['representation']['resolved']['names'].reverse()
    with pytest.raises(ValueError,match='RECIPE_IDENTITY'):decode_modular(s,raw,b,bad)
    for index in ('-1','02','x','99'):
        with pytest.raises(ValueError,match='GENE_LIST_INDEX'):set_at(b,'architecture.group_configs.'+index+'.width',4)


def test_pointwise_mutation_records_inactive_temporal_genes():
    r=grouped_recipe();s,b=make_contract(SPACE,baseline(),r);raw=baseline_genome(s,b)
    for i,g in enumerate(s['genes']):
        if g['path'].startswith('architecture.group_configs.0.'):
            if g['path'].endswith(('kind','kernel','dilation_base')):raw[i]=1.
    d=decode_modular(s,raw,b,r)
    assert d['requested']['architecture']['group_configs'][0]['kernel']==5
    assert d['effective']['architecture']['group_configs'][0]['kernel']==1
    assert sum(not x for x in d['active_gene_mask'])>=2
