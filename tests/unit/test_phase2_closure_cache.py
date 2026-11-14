import pytest

from app.rl_subset import digest
from tools.run_discrete_curve_probe import write
from tools.run_phase2_closure import old_cache


def test_cache_requires_original_context_columns_and_result_digest(tmp_path):
    spec = dict(identity='sealed-source', device='cuda')
    context = dict(spec)
    columns = ['a', 'b']
    result = dict(score=1., eligible=True)
    identity = digest(dict(context=context, columns=columns))
    path = tmp_path/'cache'/(identity+'.json')
    row = dict(identity=identity, columns=columns, result=result, sha256=digest(result))
    write(path, row)
    assert old_cache(tmp_path, spec)[tuple(columns)][0] == result
    with pytest.raises(ValueError, match='CACHE_IDENTITY'):
        old_cache(tmp_path, dict(spec, device='cpu'))
    with pytest.raises(ValueError, match='CACHE_IDENTITY'):
        old_cache(tmp_path, dict(spec, identity='other-source'))
    write(path, dict(row, columns=['b', 'a']))
    with pytest.raises(ValueError, match='CACHE_IDENTITY'):
        old_cache(tmp_path, spec)
    write(path, dict(row, result=dict(score=2., eligible=True)))
    with pytest.raises(ValueError, match='CACHE_IDENTITY'):
        old_cache(tmp_path, spec)
