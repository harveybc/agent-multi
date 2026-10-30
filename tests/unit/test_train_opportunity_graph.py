from types import SimpleNamespace

import numpy as np
import pytest

from app.train_opportunity_graph import expand_edges
from app.zigzag_opportunities import HOUR, OpportunityEvaluator


def sample():
    o=np.array([10.,11.,12.,11.,9.,8.])
    tape=SimpleNamespace(open=o,high=o+.5,low=o-.5,times=np.arange(6)*HOUR,
        chunks=[(0,6)],allowed=np.array([True,False,False,False,True,False]),
        fill_cost=np.full(6,.1),financing=np.zeros((2,6)),limit=np.full(6,5))
    pivots=[(1,-1,10.5),(2,1,12.5),(4,-1,8.5)]
    z=SimpleNamespace(Params=lambda *x:x,hindsight_pivots=lambda *x:pivots)
    old=OpportunityEvaluator(tape,z,3,future_pivots=3).evaluate((3,5,1),details=True)['records']
    return tape,pivots,old


def test_all_edges_both_sides_and_censored_tail():
    tape,pivots,old=sample()
    graph,summary=expand_edges(tape,pivots,old)
    assert summary['retained_directional_edges']==6
    assert summary['nonwinning_edges']==4 and summary['nonpositive_edges']>0
    assert graph['candidate_end'][0].tolist()==[1,2,4]
    assert graph['complete'].tolist()==[True,False]
    assert np.isnan(graph['net'][:,1]).all()
    assert summary['original_winner_max_error']==0


def test_no_cross_gap_or_edited_source():
    tape,pivots,old=sample()
    tape.chunks=[(0,3),(3,6)]
    with pytest.raises(ValueError,match='SOURCE_EVENT_MISMATCH'):
        expand_edges(tape,pivots,old)
    tape,pivots,old=sample()
    old['score'][0,0]+=1
    with pytest.raises(ValueError,match='SOURCE_VALUE_CHANGED'):
        expand_edges(tape,pivots,old)


def test_duplicate_pivots_refused():
    tape,pivots,old=sample()
    with pytest.raises(ValueError,match='EVENT_ORDER'):
        expand_edges(tape,[pivots[0]]+pivots,old)


def test_pinned_blob_current_hash_and_path_guard(tmp_path):
    import hashlib
    from tools.build_train_opportunity_graph import pinned_blob
    p=tmp_path/'indicator.py'
    p.write_bytes(b'example source\n')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    data,revision=pinned_blob(tmp_path,'indicator.py',digest)
    assert data==p.read_bytes() and revision=='CURRENT_BYTES_MATCH'
    with pytest.raises(ValueError,match='UNSAFE_SOURCE_NAME'):
        pinned_blob(tmp_path,'../indicator.py',digest)
