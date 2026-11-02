import json
import pytest
from tools.prepare_mask_selection_handoff import prepare,digest,sha


def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v))


def fixture(root,best=True):
    s={'design':{'seeds':[7,19]}}; s['identity']=digest(s); put(root/'SPEC.json',s)
    b={'config':{'batch_size':32},'eligible':True,'score':1.5}
    put(root/'CLOSE.json',{'identity':s['identity'],'best':b if best else None})
    put(root/'generations'/'0000.json',{'identity':s['identity'],'results':[b]})
    for seed,score in [(7,1.),(19,2.)]:
        key=digest({'config':b['config'],'seed':seed})
        out=root/'collected'/'worker'/'runs'/key/'attempt1'
        put(out/'BEST.json',{'step':100,'score':score})
        for name in ['best.pt','EFFECTIVE.json','HISTORY.json','TRAINING.json']: put(out/name,{})
        r={'identity':s['identity'],'config':b['config'],'seed':seed,'eligible':True,
           'score':score,'selected_step':100,'metrics':{},
           'artifacts':{p.name:sha(p) for p in out.iterdir()}}
        put(out/'RESULT.json',r)
    return out


def test_wait_does_not_create_selection(tmp_path):
    assert prepare(tmp_path)['state']=='WAIT_DEAP_CLOSE'
    assert not list(tmp_path.iterdir())


def test_complete_and_idempotent(tmp_path):
    fixture(tmp_path)
    r=prepare(tmp_path)
    assert r['state']=='READY_FOR_BRIEFING' and not r['phase2_launched']
    assert [x['seed'] for x in r['checkpoints']]==[7,19]
    before=(tmp_path/'MASK_SELECTION_HANDOFF.json').read_bytes()
    assert prepare(tmp_path)==r
    assert (tmp_path/'MASK_SELECTION_HANDOFF.json').read_bytes()==before


def test_no_winner_is_not_substituted(tmp_path):
    fixture(tmp_path,False)
    assert prepare(tmp_path)['state']=='NO_ELIGIBLE_CHAMPION'


@pytest.mark.parametrize('attack',['weights','missing_seed','score','generation','identity'])
def test_tampering_refused(tmp_path,attack):
    out=fixture(tmp_path)
    if attack=='weights': (out/'best.pt').write_text('altered')
    elif attack=='missing_seed': (out/'RESULT.json').unlink()
    elif attack=='score':
        c=json.loads((tmp_path/'CLOSE.json').read_text()); c['best']['score']=999
        put(tmp_path/'CLOSE.json',c)
    elif attack=='generation': (tmp_path/'generations'/'0000.json').unlink()
    else:
        c=json.loads((tmp_path/'CLOSE.json').read_text()); c['identity']='wrong'
        put(tmp_path/'CLOSE.json',c)
    with pytest.raises(ValueError,match='HANDOFF'): prepare(tmp_path)
    assert not (tmp_path/'MASK_SELECTION_HANDOFF.json').exists()
