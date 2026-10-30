import json
import pytest

from tools.collect_causal_history_lanes import authenticate
from tools.run_causal_history_queue import freeze_manifest
from tools.run_crossed_account_queue import save
from app.zigzag_opportunities import sha


def test_collector_authenticates_artifacts_and_refuses_tampering(tmp_path):
    identity=freeze_manifest(tmp_path/'MANIFEST.json',{'mode':'median'})
    save(tmp_path/'TESTS.json',identity,dict(passed=200,skipped=0))
    save(tmp_path/'COMPLETE.json',identity,dict(mode='median',status='COMPLETE',candidates=0,
        search_gate={'eligible':False},confirmation={},receipt_tail=identity,
        artifacts={'TESTS.json':sha(tmp_path/'TESTS.json')}))
    assert authenticate(tmp_path)['tests']['passed']==200
    (tmp_path/'TESTS.json').write_text(json.dumps({'tampered':True}))
    with pytest.raises(ValueError,match='COLLECTION_ARTIFACT'):
        authenticate(tmp_path)
