import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.frozen_policy_diagnostics import close_stages, DiagnosticWorld


def test_penalty_stages_and_strict_threshold():
    values=np.array([[2.,0.,1.],[4.,0.,1.]])
    w=np.array([.5,.5])
    r=close_stages(values,w,1.,1.,.75,.5)
    assert r['raw']==2 and r['after_uncertainty']==1 and r['final']==.25
    assert r['raw_pass'] and r['uncertainty_pass'] and not r['history_pass']
    r=close_stages(values,w,1.,0.,0.,2.)
    assert not r['raw_pass'] and not r['history_pass']


def test_nested_source_manifest_and_tampering(tmp_path):
    from app.zigzag_opportunities import sha
    from tools.diagnose_frozen_history_policy import verify_sources
    root=tmp_path/'agent';native=tmp_path/'native';root.mkdir();native.mkdir()
    (root/'a.py').write_text('a');(native/'n.py').write_text('n')
    files={'a.py':sha(root/'a.py')}
    manifest=dict(sources=files,native_sources=dict(agent_multi=files,
                   heuristic_strategy={'n.py':sha(native/'n.py')}))
    verify_sources(manifest,root,native)
    (native/'n.py').write_text('changed')
    with pytest.raises(ValueError,match='DIAGNOSTIC_SOURCE_CHANGED'):
        verify_sources(manifest,root,native)


@pytest.mark.parametrize('history',[0.,1.])
def test_observation_does_not_change_account(history):
    root=os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root: pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    from app.causal_history_account import HistoryWorld,base_genes,specifications
    engine(Path(root))
    risk=json.loads((Path(__file__).resolve().parents[2]/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
    spec=specifications('confirmation',smoke=True)[0]
    g=base_genes(history,history)
    actual=DiagnosticWorld(spec,risk,'median')
    observed=actual.run(g,details=True)
    reference=HistoryWorld(spec,risk,'median').run(g,details=True)
    assert observed==reference
    assert actual.diagnostic['flat_ticks']>0
    assert actual.diagnostic['raw_bids']==8*actual.diagnostic['flat_ticks']
    from collections import Counter
    other=base_genes(1-history,1-history)
    actual.diagnostic=Counter()
    assert actual.run(other,details=True)==HistoryWorld(spec,risk,'median').run(other,details=True)
