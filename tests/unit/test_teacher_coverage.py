import numpy as np
import pytest

from app.teacher_coverage import nested_support


def test_nested_exact_deterministic_and_quarter_balanced():
    base=np.array([3,9]);pool=np.arange(100,300);q=np.repeat(['Q1','Q2','Q3','Q4'],50)
    a=nested_support(base,pool,q);b=nested_support(base,pool,q)
    assert len(a[4])==8 and len(a[16])==32
    assert set(base)<=set(a[4])<=set(a[16])<=set(base)|set(pool)
    assert all(np.array_equal(a[f],b[f]) for f in a)
    for rows in a.values():
        counts=[sum(int(x) in set(rows) for x in pool[q==k]) for k in sorted(set(q))]
        assert max(counts)-min(counts)<=1


def test_excluded_rows_never_enter_and_inputs_not_mutated():
    base=np.array([1,2]);pool=np.arange(20,100);q=np.repeat([0,1],40)
    old=pool.copy();excluded={3,4,5,6}
    assert not set(nested_support(base,pool,q)[16])&excluded
    assert np.array_equal(old,pool)


@pytest.mark.parametrize('base,pool,quarters,factors',[
    ([1,1],list(range(100)),[0]*100,(4,16)),
    ([1],[1,2],[0,0],(2,)),
    ([1],[2],[0],(4,)),
    ([1],list(range(20,40)),[0]*20,(16,4)),
])
def test_invalid_support_rejected(base,pool,quarters,factors):
    with pytest.raises(ValueError):nested_support(base,pool,quarters,factors)
