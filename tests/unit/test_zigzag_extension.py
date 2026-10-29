import copy

import pytest

from tools.extend_zigzag_opportunity import paired
from tools.zigzag_opportunity_deap import search


def test_300_generation_extension_keeps_entire_40_generation_prefix():
    def evaluate(g):
        return dict(eligible=g[0]<20,fitness=-abs(g[0]-13),params=g)
    a=search(evaluate,7,30,40)
    b=search(evaluate,7,30,300)
    r=paired(a,b)
    assert r['prefix_generations_verified']==40 and r['delta']>=0
    assert len(b['history'])==300


def test_20_by_300_replays_the_20_by_30_prefix():
    def evaluate(g):
        return dict(eligible=g[0]<20,fitness=-abs(g[0]-13),params=g)
    a=search(evaluate,19,20,30)
    b=search(evaluate,19,20,300)
    r=paired(a,b)
    assert r['prefix_generations_verified']==30
    assert r['delta']>=0


@pytest.mark.parametrize('mutation,reason',[
    ('seed','PAIRED_SEARCH_SETTINGS'),('population','PAIRED_SEARCH_SETTINGS'),
    ('generations','EXTENSION_BUDGET'),('history','GENERATION_PREFIX_MISMATCH'),
    ('fitness','ELITISM_REGRESSION')])
def test_pairing_rejects_unmatched_or_regressed_result(mutation,reason):
    a=dict(seed=7,population=30,generations=40,history=list(range(40)),
           best=dict(eligible=True,fitness=1,params=[3,5,1]))
    b=copy.deepcopy(a); b.update(generations=300,history=list(range(300)))
    if mutation=='history':
        b['history'][0]=-1
    elif mutation=='fitness':
        b['best']['fitness']=0
    else:
        b[mutation]=1
    with pytest.raises(ValueError,match=reason):
        paired(a,b)
