"""Explicit, versioned order decoding; market observations are never augmented."""
import numpy as np

from app.frozen_recipe_inputs import require
from env_plugins.recipe_native import RecipeNativeEnv


CONTRACTS = {
    'TARGET': {'mode': 'target', 'threshold': 1/3},
    'COMMANDS5': {'mode': 'commands5', 'threshold': None},
}


class OrderContractEnv(RecipeNativeEnv):
    def __init__(self, engine, inputs, gamma=.99, contract=None):
        self.contract = dict(contract or CONTRACTS['TARGET'])
        require(self.contract['mode'] in ('target', 'commands5'), 'ORDER_MODE')
        if self.contract['mode'] == 'target':
            require(0 < self.contract['threshold'] < 1, 'ORDER_THRESHOLD')
        self.order_counts = {}
        super().__init__(engine, inputs, gamma)

    def reset(self, **kwargs):
        self.order_counts = {}
        return super().reset(**kwargs)

    def _decode(self, x):
        require(np.shape(x) == (3,) and np.isfinite(x).all()
                and (np.abs(x) <= 1.000001).all(), 'ACTION')
        position = self.engine.pos.direction if self.engine.pos is not None else 0
        invalid = False
        if self.contract['mode'] == 'target':
            threshold = self.contract['threshold']
            target = 1 if x[0] > threshold else (-1 if x[0] < -threshold else 0)
            command = {0: 'TARGET_FLAT', 1: 'TARGET_LONG', -1: 'TARGET_SHORT'}[target]
            action = 0 if target == position else (3 if target == 0 else (1 if target > 0 else 2))
        else:
            # Boundaries use the action dtype, including its representable float32 edges.
            cast = np.asarray(x).dtype.type
            if x[0] < cast(-.6):
                command, required, action = 'CLOSE_SHORT', -1, 3
            elif x[0] < cast(-.2):
                command, required, action = 'OPEN_SHORT', 0, 2
            elif x[0] <= cast(.2):
                command, required, action = 'HOLD', None, 0
            elif x[0] <= cast(.6):
                command, required, action = 'OPEN_LONG', 0, 1
            else:
                command, required, action = 'CLOSE_LONG', 1, 3
            invalid = required is not None and position != required
            if invalid:
                action = 0
        self.last_order = dict(command=command, position=position, engine_action=action, invalid=invalid)
        key = f'{position}:{command}:{"INVALID_HOLD" if invalid else "APPLICABLE"}'
        self.order_counts[key] = self.order_counts.get(key, 0) + 1
        tp_lo, tp_hi = self.engine.cfg.tp_bounds
        sl_lo, sl_hi = self.engine.cfg.sl_bounds
        return (action, tp_lo + (float(x[1]) + 1) / 2 * (tp_hi - tp_lo),
                sl_lo + (float(x[2]) + 1) / 2 * (sl_hi - sl_lo))

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info['order_contract'] = dict(self.last_order)
        return obs, reward, done, truncated, info
