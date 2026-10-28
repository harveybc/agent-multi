"""Causal scalar rewards. None of these transforms market observations."""
import math


class Reward:
    def __init__(self, mode, capital, penalty=0., eta=.01, variance_floor=1e-6, warmup=32):
        if mode not in ('equity','log','drawdown','dsr','mixed') or capital <= 0:
            raise ValueError('REWARD_CONTRACT')
        if not (0 < eta <= 1 and variance_floor > 0 and penalty >= 0):
            raise ValueError('REWARD_PARAMETERS')
        self.mode, self.capital, self.penalty = mode, float(capital), penalty
        self.eta, self.floor, self.warmup = eta, variance_floor, warmup
        self.peak, self.mdd, self.a, self.b, self.n = float(capital), 0., 0., 0., 0

    def step(self, before, after):
        if not (math.isfinite(before) and math.isfinite(after) and before > 0 and after > 0):
            raise ValueError('REWARD_NONPOSITIVE_EQUITY')
        u = after/before-1
        g = math.log(after/before)
        self.peak = max(self.peak, after)
        mdd = max(self.mdd, 1-after/self.peak)
        increment = mdd-self.mdd
        self.mdd = mdd
        da, db = u-self.a, u*u-self.b
        dsr = (self.b*da-.5*self.a*db) / max(self.b-self.a*self.a, self.floor)**1.5
        self.a += self.eta*da
        self.b += self.eta*db
        self.n += 1
        if self.n <= self.warmup:
            dsr = 0.
        value = {'equity':100*(after-before)/self.capital, 'log':100*g,
                 'drawdown':100*(g-self.penalty*increment), 'dsr':dsr,
                 'mixed':.5*100*g+.5*dsr}[self.mode]
        if not math.isfinite(value):
            raise ValueError('REWARD_NONFINITE')
        return value
