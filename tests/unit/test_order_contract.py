from types import SimpleNamespace

import numpy as np
import pytest

from env_plugins.recipe_native import RecipeNativeEnv


def decoder(mode, position=0, threshold=1/3):
    from env_plugins.order_contract import OrderContractEnv
    env = object.__new__(OrderContractEnv)
    env.contract = {'mode': mode, 'threshold': threshold}
    env.order_counts = {}
    env.engine = SimpleNamespace(pos=SimpleNamespace(direction=position) if position else None,
                                cfg=SimpleNamespace(tp_bounds=(.25,3.),sl_bounds=(.5,3.)))
    return env


@pytest.mark.parametrize('position', [-1,0,1])
@pytest.mark.parametrize('value,command,required,executed', [
    (-.8,'CLOSE_SHORT',-1,3),(-.4,'OPEN_SHORT',0,2),(0.,'HOLD',None,0),
    (.4,'OPEN_LONG',0,1),(.8,'CLOSE_LONG',1,3)])
def test_explicit_commands_truth_table(position,value,command,required,executed):
    env=decoder('commands5',position)
    expected=executed if required is None or position==required else 0
    result=env._decode(np.array([value,0.,0.]))
    assert result==(expected,1.625,1.75)
    assert env.last_order['command']==command
    assert env.last_order['invalid']==(required is not None and position!=required)


@pytest.mark.parametrize('value,command', [(-1.,'CLOSE_SHORT'),(-.6,'OPEN_SHORT'),
    (-.2,'HOLD'),(.2,'HOLD'),(.6,'OPEN_LONG'),(1.,'CLOSE_LONG')])
def test_boundaries(value,command):
    for dtype in (np.float32,np.float64):
        env=decoder('commands5')
        env._decode(np.array([value,0.,0.],dtype))
        assert env.last_order['command']==command


def test_target_equals_original_exactly():
    rng=np.random.default_rng(7)
    for pos in (-1,0,1):
        env=decoder('target',pos)
        for x in [np.array([v,0.,0.]) for v in (-1.,-1/3,0.,1/3,1.)]+list(rng.uniform(-1,1,(100,3))):
            assert env._decode(x)==RecipeNativeEnv._decode(env,x)


def test_threshold_controls_flat_and_entry_not_hold():
    env=decoder('target',1,.25)
    assert env._decode(np.array([.2,0.,0.]))[0]==3
    env=decoder('target',1,.15)
    assert env._decode(np.array([.2,0.,0.]))[0]==0
    assert decoder('commands5',1)._decode(np.array([0.,0.,0.]))[0]==0


@pytest.mark.parametrize('bad', [[np.nan,0,0],[2,0,0],[0,0]])
def test_bad_action_rejected(bad):
    with pytest.raises(ValueError,match='ACTION'):
        decoder('commands5')._decode(np.array(bad))
