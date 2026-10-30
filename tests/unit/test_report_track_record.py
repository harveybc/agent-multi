from dataclasses import replace
import copy

import numpy as np
import pytest

from app.close_hold_controller import PathReport
from app.report_track_record import ReportTrackRecord


def fixture_report(at=10):
    paths=np.full((32, 3, 4), 100.)
    paths[:, :, 1]=101
    paths[:, :, 2]=99
    return PathReport(at, at, at, 1, at+3, paths, np.zeros((32,3)),
                      np.full(32,1/32), 'causal', np.zeros(32))


def test_only_matured_targets_enter_history_and_interval():
    ledger=ReportTrackRecord('producer-v1', horizons=(1,2,3), minimum=2)
    r=fixture_report()
    ledger.issue(r, now=10, expected_deadline=13)
    r.paths[:]=900
    assert ledger.diagnostic(1)['count']==0
    with pytest.raises(ValueError, match='OBSERVATION_CLOCK'):
        ledger.observe(11,101,now=10)
    ledger.observe(11,101,now=11)
    assert ledger.diagnostic(1)['mae']==1
    assert ledger.diagnostic(2)['count']==0
    assert ledger.interval(100,1,now=11) is None
    ledger.issue(fixture_report(11),now=11,expected_deadline=14)
    ledger.observe(12,102,now=12)
    assert ledger.diagnostic(1)['count']==2
    assert ledger.interval(100,1,now=12)==[98.,102.]


def test_restart_snapshot_and_producer_identity():
    ledger=ReportTrackRecord('producer-v1',horizons=(1,2,3))
    ledger.issue(fixture_report(),now=10,expected_deadline=13)
    state=ledger.state()
    other=ReportTrackRecord.restore(state,expected_producer='producer-v1')
    for at in (11,12,13):
        ledger.observe(at,100+at/10,now=at)
        other.observe(at,100+at/10,now=at)
    assert other.state()==ledger.state()
    with pytest.raises(ValueError,match='PRODUCER'):
        ReportTrackRecord.restore(state,expected_producer='changed')
    altered=copy.deepcopy(state);altered['data']['clock']=99
    with pytest.raises(ValueError,match='SEAL'):
        ReportTrackRecord.restore(altered,expected_producer='producer-v1')


def test_duplicate_future_and_synthetic_issues_refused():
    ledger=ReportTrackRecord('x')
    with pytest.raises(ValueError,match='STALE_OR_FUTURE'):
        ledger.issue(fixture_report(11),now=10,expected_deadline=14)
    with pytest.raises(ValueError,match='CAUSAL_REPORT'):
        ledger.issue(replace(fixture_report(),provenance='synthetic_ideal'),now=10,expected_deadline=13)
    ledger.issue(fixture_report(),now=10,expected_deadline=13)
    with pytest.raises(ValueError,match='DUPLICATE'):
        ledger.issue(fixture_report(),now=10,expected_deadline=13)


def test_missing_outcome_is_not_filled_with_later_price():
    ledger=ReportTrackRecord('x',horizons=(1,2,3))
    ledger.issue(fixture_report(),now=10,expected_deadline=13)
    ledger.observe(12,110,now=12)
    assert ledger.diagnostic(1)['count']==0
    assert ledger.diagnostic(2)['mae']==10
    assert ledger.state()['data']['missing_targets']==1
