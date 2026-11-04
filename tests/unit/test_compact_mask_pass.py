import pytest
import torch

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, subset_recipe
from app.frozen_actor_compaction import compact_actor, effective_actor
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe
from tools.run_discrete_curve_probe import write
from tools.run_compact_mask_pass import authenticated_compaction
from tools.run_refit_mask_pass import dataset_for, score_metrics
from feature_selector.plugins import resolve
from types import SimpleNamespace


def parent_fixture(path, **changes):
    spec=dict(checkpoint_sha256='fixture'); spec['identity']=digest(spec)
    result=dict(identity=spec['identity'],compact_inference_qualified=True,full_reload_exact=True,
        source_unchanged=True,test_read=False,training=False) | changes
    config=dict(lineage=dict(diagnostic_identity=spec['identity'],source_checkpoint_sha256='fixture'))
    for name,value in [('SPEC.private.json',spec),('RESULT.json',result),('bundle/CONFIG.json',config)]:
        write(path/name,value)
    for name in ('MANIFEST.json','actor.pt','PROBE.npz','RELOAD.json'): write(path/'bundle'/name,{})
    write(path/'COMPLETE.json',dict(identity=spec['identity'],
        files={str(p.relative_to(path)):file_sha(p) for p in path.rglob('*') if p.is_file()}))


@pytest.mark.parametrize('changes',[dict(test_read=True),dict(training=True),dict(source_unchanged=False),
                                  dict(compact_inference_qualified=False),dict(full_reload_exact=False)])
def test_unqualified_parent_rejected(tmp_path,changes):
    parent_fixture(tmp_path,**changes)
    with pytest.raises(ValueError,match='QUALIFICATION'): authenticated_compaction(tmp_path)


def test_modified_parent_and_missing_bundle_file_rejected(tmp_path):
    parent_fixture(tmp_path)
    assert authenticated_compaction(tmp_path)[1]['full_reload_exact']
    write(tmp_path/'bundle/actor.pt',{'corrupt':True})
    with pytest.raises(ValueError,match='BUNDLE_HASH'): authenticated_compaction(tmp_path)


def test_second_round_export_uses_original_graph_and_keeps_previous_masks():
    torch.set_num_threads(1)
    from app.rl_subset import seal_recipe
    from app.frozen_actor_compaction import ConstantResponse
    r=probe_recipe()
    r['list']=dict(n=3,columns=['x','y','z'])
    r['tensor']['columns'].append(dict(store_column='z',tensor_name='z',consumer_ops=[]))
    r['scaler']['features'].append('z'); r['scaler']['mean'].append(0.); r['scaler']['std'].append(1.)
    r['representation']['resolved']=dict(names=['empty','partial'],indices=[[0],[1,2]],units=[2,2],kinds=['conv','conv'])
    r=seal_recipe(r); actor=effective_actor(ThreeSignalSAC(r))
    first,_=compact_actor(actor,r,['y','z'])
    final,_=compact_actor(actor,r,['z'])
    x=torch.randn(5,13,1); masked=torch.cat([torch.zeros_like(x),x],dim=-1)
    with torch.inference_mode():
        torch.testing.assert_close(first.surface(masked),final.surface(x),atol=1e-5,rtol=1e-5)
    assert isinstance(first.encoder.branches[0],ConstantResponse)
    assert torch.equal(first.encoder.branches[0].response,final.encoder.branches[0].response)
    assert subset_recipe(r,['z'])['list']['columns']==['z']


def test_actual_selector_deletes_tie_and_does_not_reinsert_it():
    from tests.unit.test_rl_subset import split_recipe
    from app.frozen_recipe_inputs import FrozenRecipeInputs
    from tests.unit.test_rl_subset import frame
    r=split_recipe()
    r['split']['periods']['VALIDATION']=['2024-01-01','2025-01-01']
    from app.rl_subset import seal_recipe
    r=seal_recipe(r)
    inputs=FrozenRecipeInputs.from_frame(r,frame())
    dataset=dataset_for(SimpleNamespace(recipe=r,inputs=inputs))
    selector,_=resolve('selectors','two_pass_greedy')
    calls=[]
    def evaluate(columns):
        calls.append(list(columns))
        return score_metrics({p:dict(return_per_week=1.,orders_per_week=3.) for p in ('TRAIN','VALIDATION')})
    result=selector.select(dataset,evaluate,dict(remove_on_tie=True,tolerance=0.,min_features=1))
    assert result['counts']==dict(initial=2,after_deletion=1,final=1)
    assert result['selected']==['y']
    assert calls==[['x','y'],['y']]
    assert [t['accepted'] for t in result['trace']]==[True,False]
