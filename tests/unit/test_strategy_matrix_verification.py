import hashlib
import json

import pytest

from app.strategy_matrix_verification import digest,verify


def fixture(root):
    axes={'horizon':[1,4]}
    identity={'design':{'axes':axes,'constants':{},'expected_cells':2},'expected_cells':2,'limit':None}
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    put('IDENTITY.json',identity)
    for h in (1,4):
        cell={'horizon':h}
        put('cells/'+digest(cell)+'.json',dict(cell=cell,state='PASS',identity_sha256=digest(identity),
            comparisons=[dict(horizon=h,state='FLAT',chosen=0)],prescribed=[{}, {}, {}],unreachable=[]))
    summary=dict(state='PASS_DECLARED_GRID',cells=2,comparisons={'FLAT':2},chosen={'0':2},
        prescribed_checks=6,unreachable=0,failures=0)
    put('SUMMARY.json',summary)
    artifacts={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.json')}
    put('CLOSE.json',dict(summary=summary,artifacts=artifacts))


def test_independent_cartesian_reconstruction(tmp_path):
    fixture(tmp_path);r=verify(tmp_path)
    assert r['cells']==2 and r['axis_counts']=={'horizon':{'1':1,'4':1}}


def test_removed_cell_never_counts_as_pass(tmp_path):
    fixture(tmp_path)
    next((tmp_path/'cells').glob('*.json')).unlink()
    with pytest.raises(ValueError,match='ARTIFACT_HASH'):verify(tmp_path)


def test_forged_zero_failures_does_not_hide_cell_failure(tmp_path):
    fixture(tmp_path);p=next((tmp_path/'cells').glob('*.json'))
    r=json.loads(p.read_text());r['state']='FAIL';p.write_text(json.dumps(r))
    close=json.loads((tmp_path/'CLOSE.json').read_text())
    close['artifacts'][str(p.relative_to(tmp_path))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'CLOSE.json').write_text(json.dumps(close))
    with pytest.raises(ValueError,match='CELL_BINDING'):verify(tmp_path)
