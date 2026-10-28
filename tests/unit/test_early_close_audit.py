import pytest

from app.early_close_audit import audit_episode


def case(direction=1):
    return dict(mode='ideal', trades=[dict(direction=direction, entry_bar_abs=1,
        exit_bar_abs=4, close_reason='AGENT_CLOSE', net_usd=2.)], decisions=[dict(
        bar=3, command=3 if direction==1 else 4, expected=dict(close_net=2.,
        hold_net=-3., reference=dict(reason='SL')), event=dict(close='AGENT_CLOSE'))])


@pytest.mark.parametrize('direction', [1, -1])
def test_actual_two_sided_close_capture(direction):
    result=audit_episode(case(direction))
    assert not result['failures']
    assert result['counts']['captured']==result['counts']['opportunities']==1
    assert result['state']=='CAPTURE_VERIFIED'


@pytest.mark.parametrize('mutation', ['hold','wrong_side','sl','delay','net','reference'])
def test_mutations_fail(mutation):
    row=case();d=row['decisions'][0];t=row['trades'][0]
    if mutation=='hold':d['command']=0
    if mutation=='wrong_side':d['command']=4
    if mutation=='sl':t['close_reason']='SL';d['event']['close']='SL'
    if mutation=='delay':t['exit_bar_abs']=5
    if mutation=='net':t['net_usd']=3.
    if mutation=='reference':d['expected']['hold_net']=4.
    assert audit_episode(row)['failures']


def test_no_trades_or_opportunities_is_not_close_success():
    r=audit_episode(dict(mode='ideal',trades=[],decisions=[]))
    assert r['state']=='NO_EARLY_CLOSE_WITNESS'
    assert r['counts']['cells_without_trades']==1


def test_profitable_hold_is_not_penalized_and_equality_does_not_require_close():
    row=case();row['trades'][0]['close_reason']='TP'
    d=row['decisions'][0];d['command']=0;d['event']['close']=None
    for value in (2.,4.):
        d['expected']['hold_net']=value
        r=audit_episode(row)
        assert not r['failures'] and r['state']=='NO_EARLY_CLOSE_WITNESS'
