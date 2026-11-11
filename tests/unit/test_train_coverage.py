import copy

import pytest

from app.all_state_refinement import START, BOUNDARY
from app.train_coverage import expand_coverage


def fixtures():
    trace=[dict(timestamp=START+(i+30)*3600*10**9,window_start=START,row=i,
                role='fit',year=2020,position=0,allowed=True,logits=[0.,1.,-1.]) for i in range(20)]
    monitor=dict(timestamp=BOUNDARY+30*3600*10**9,window_start=BOUNDARY,row=99,
                 role='monitor',year=2023,position=1)
    return trace,[trace[0],trace[10],monitor]


def test_nested_fourfold_and_monitor_unchanged():
    trace,anchors=fixtures();selected,audit=expand_coverage(trace,anchors)
    assert len(selected)==9
    assert selected[-1]==anchors[-1]
    assert {r['timestamp'] for r in anchors}<={r['timestamp'] for r in selected}
    assert audit['narrow_fit']==2 and audit['broad_fit']==8
    assert audit['shortfall']==0


def test_deterministic_outcome_blind_and_trace_order_invariant():
    trace,anchors=fixtures();a=expand_coverage(trace,anchors)[0]
    b=copy.deepcopy(trace)
    for r in b:r['outcomes']=[{'reward_return':1e12}]
    got=expand_coverage(list(reversed(b)),anchors)[0]
    assert [r['timestamp'] for r in a]==[r['timestamp'] for r in got]
    assert all('outcomes' not in r for r in got if r['role']=='fit')


def test_shortage_reported_without_replacement():
    trace,anchors=fixtures();trace=[trace[0],trace[10],trace[11]]
    selected,audit=expand_coverage(trace,anchors)
    assert audit['broad_fit']==3 and audit['shortfall']==5
    assert len({r['timestamp'] for r in selected})==len(selected)


@pytest.mark.parametrize('mutation',['duplicate','monitor_trace','future','missing','nan','wrong_year'])
def test_bad_support_rejected(mutation):
    trace,anchors=fixtures()
    if mutation=='duplicate':trace.append(trace[0])
    if mutation=='monitor_trace':trace[1]['role']='monitor'
    if mutation=='future':trace[1]['timestamp']=BOUNDARY
    if mutation=='missing':anchors[0]=dict(anchors[0],timestamp=START+1)
    if mutation=='nan':trace[1]['logits']=[0.,float('nan'),1.]
    if mutation=='wrong_year':trace[1]['year']=2021
    with pytest.raises(ValueError):expand_coverage(trace,anchors)
