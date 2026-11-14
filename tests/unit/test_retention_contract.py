from copy import deepcopy
import pytest
from app.retention_contract import validate_resume,validate_retention_suffix


def fixture():
    trace=[dict(offset=i,accepted=True,searches_used=min(i,4)) for i in range(1,17)]
    return dict(identity='parent',source_sha256='source',offset=16,accepted=16,
        searches_used=4,trace=trace,optimizer={'state':{0:{'step':26}}})


def test_resume_valid_without_mutation():
    cp=fixture();old=deepcopy(cp)
    validate_resume(cp,'parent','source',10)
    assert cp==old


@pytest.mark.parametrize('key,value',[('offset',15),('searches_used',0),('accepted',15),('identity','other'),('source_sha256','other')])
def test_bad_resume_refused(key,value):
    cp=fixture();cp[key]=value
    with pytest.raises(ValueError):validate_resume(cp,'parent','source',10)


def test_optimizer_reset_refused():
    cp=fixture();cp['optimizer']['state'][0]['step']=0
    with pytest.raises(ValueError):validate_resume(cp,'parent','source',10)
    cp['optimizer']['state']={}
    with pytest.raises(ValueError):validate_resume(cp,'parent','source',10)


def test_no_search_suffix():
    rows=[dict(offset=i,arm='ordinary',searches_used=4) for i in range(17,33)]
    validate_retention_suffix(rows,16,32)
    for key,value in [('arm','targeted'),('searches_used',5),('target_search',{})]:
        bad=deepcopy(rows);bad[0][key]=value
        with pytest.raises(ValueError):validate_retention_suffix(bad,16,32)
    with pytest.raises(ValueError):validate_retention_suffix(rows[1:],16,32)
