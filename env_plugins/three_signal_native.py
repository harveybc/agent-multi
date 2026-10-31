"""Market-only three-signal policy with deterministic native position control."""
import numpy as np
from env_plugins.discrete_curve_native import CurveNativeEnv, HOUR


def command_for(signal, position):
    if type(signal) not in (int, np.int64) or signal not in (0,1,2) or position not in (-1,0,1):
        raise ValueError('SIGNAL_CONTRACT')
    if not position:
        return signal
    return 3 if signal == (2 if position == 1 else 1) else 0


class ThreeSignalEnv(CurveNativeEnv):
    def observation(self):
        self.last_observation = dict(features=self.inputs.observation(int(self.rows[self.bar]))['features'],
                                     mask=np.ones(3, bool))
        return self.last_observation

    def step(self, action):
        if self.finished:
            raise ValueError('EPISODE_FINISHED')
        e = self.engine
        command = command_for(action, 0 if e.pos is None else e.pos.direction)
        self.last_command = command
        plan = self.entry_plan(1 if command == 1 else -1) if command in (1,2) else None
        reward, duration = 0., 0
        while True:
            previous = float(e.equity_at(self.bar))
            e.step(self.bar, command, entry_plan=plan)
            self.bar += 1
            equity = float(e.equity_at(self.bar))
            reward += self.gamma ** duration * 100. * (equity-previous) / e.cfg.capital
            duration += 1
            self.equity_trace.append((int(e.v[self.bar])+HOUR, equity))
            if self.bar >= e.last:
                if e.pos is not None or not np.isclose(equity, e.balance):
                    raise ValueError('TERMINAL_SETTLEMENT_MISSING')
                self.finished = True
                return self.last_observation, reward, True, duration
            if self.valid[self.bar]:
                return self.observation(), reward, False, duration
            command, plan = 0, None
            self.auto_holds += 1
