import gzip
import json

import pytest

from app.rl_subset import digest
from app.frozen_recipe_inputs import file_sha
from tools.audit_early_close_grid import tick
from tools.recipe_sac_queue import write,read


def source(root):
    identity={'source':'synthetic_fixture'}
    write(root/'IDENTITY.json',identity)
    case=dict(mode='ideal',trades=[],decisions=[])
    batch=dict(rows=[dict(case_id=0,state='PASS',result=case)])
    path=root/'batches'/'0000000.json.gz';path.parent.mkdir()
    path.write_bytes(gzip.compress(json.dumps(batch).encode()))
    receipt=path.with_suffix('')
    write(receipt,dict(identity=digest(identity),sha256=file_sha(path),count=1))
    write(root/'CLOSE.json',dict(identity=digest(identity),failures=0,completed=1,
                               receipts={receipt.name:file_sha(receipt)}))
    return path


def test_no_opportunity_is_explicit_and_close_is_immutable(tmp_path):
    root=tmp_path/'source';output=tmp_path/'audit';source(root)
    assert tick(root,output)
    closed=(output/'CLOSE.json').read_bytes()
    result=read(output/'CLOSE.json')
    assert result['capture_rate'] is None and not result['two_sided_close_witness']
    assert result['counts']['cells_without_close_opportunity']==1
    assert tick(root,output) and (output/'CLOSE.json').read_bytes()==closed


def test_terminal_data_corruption_rejected_even_with_cached_audit(tmp_path):
    root=tmp_path/'source';output=tmp_path/'audit';data=source(root)
    tick(root,output);data.write_bytes(b'altered')
    with pytest.raises(ValueError,match='TERMINAL_DATA'):tick(root,output)
