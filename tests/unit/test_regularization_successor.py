from types import SimpleNamespace

import pytest

from tools.run_regularization_successor import wait_for_success


def test_waits_until_successful_predecessor_exit():
    rows=iter(['active','deactivating','inactive'])
    stages=[]; sleeps=[]
    def run(command,**kwargs):
        assert command[:3]==['systemctl','--user','show']
        return SimpleNamespace(stdout='LoadState=loaded\nActiveState='+next(rows)+
                               '\nResult=success\nExecMainStatus=0\n')
    wait_for_success('example.service',stages.append,run,sleeps.append)
    assert stages==['WAITING_FOR_PREDECESSOR']*2 and sleeps==[30,30]


@pytest.mark.parametrize('row',[
    'LoadState=not-found\nActiveState=inactive\nResult=success\nExecMainStatus=0',
    'LoadState=loaded\nActiveState=failed\nResult=exit-code\nExecMainStatus=1',
    'LoadState=loaded\nActiveState=inactive\nResult=success\nExecMainStatus=1',
    'LoadState=loaded\nActiveState=inactive\nResult=oom-kill\nExecMainStatus=0'])
def test_never_launches_after_missing_or_failed_predecessor(row):
    with pytest.raises(ValueError,match='PREDECESSOR_SERVICE'):
        wait_for_success('example.service',lambda x:None,
                         lambda *a,**kw:SimpleNamespace(stdout=row),
                         lambda seconds:pytest.fail('must not wait on failed predecessor'))
