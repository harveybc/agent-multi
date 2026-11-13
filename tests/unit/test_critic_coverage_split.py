import pytest
from app.critic_coverage_split import split_new_panel


def panel():
    return [dict(timestamp=year*1000+j+offset,year=year,kind=kind)
        for year in range(2020,2024) for kind,n,offset in [('flat',12,0),('long',4,20),('short',4,30),('restricted',2,40)]
        for j in range(n)]


def test_partition_fixed_disjoint_and_keeps_2023_out_of_fit():
    a=panel();fit,held=split_new_panel(a,[])
    assert (len(fit),len(held))==(33,55)
    assert not set(fit)&set(held)
    assert set(fit)|set(held)==set(range(88))
    assert all(a[i]['year']<2023 for i in fit)
    assert split_new_panel(a,[])==(fit,held)
    for row in a:row['target']=999999
    assert split_new_panel(a,[])==(fit,held)


def test_bad_origins_rejected():
    a=panel()
    with pytest.raises(ValueError,match='OVERLAP'):split_new_panel(a,[a[0]['timestamp']])
    with pytest.raises(ValueError,match='OVERLAP'):split_new_panel(a+[a[0]],[])
    a[0]['year']=2024
    with pytest.raises(ValueError,match='YEAR'):split_new_panel(a,[])
