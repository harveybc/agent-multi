"""Bounded, ordered CPU workers; each owns a fixed subset of causal worlds."""
import multiprocessing as mp
import math
from pathlib import Path

import numpy as np


def activity_eligible(mean_orders):
    if not math.isfinite(mean_orders) or mean_orders<0:
        raise ValueError('INVALID_ACTIVITY')
    return bool(mean_orders>2.)


def worker(channel,specs,cfg,native,mode):
    try:
        from app.zigzag_opportunities import engine
        engine(Path(native))
        from app.causal_history_account import HistoryWorld
        worlds=[HistoryWorld(s,cfg,mode) for s in specs]
        channel.send(('ready',len(worlds)))
        while True:
            message=channel.recv()
            if message is None: break
            genes,enabled,verify,details,voluntary=message
            channel.send(('ok',[w.run(genes,enabled,verify,details,
                voluntary_close=voluntary) for w in worlds]))
    except Exception as exc:
        channel.send(('error',(type(exc).__name__,str(exc))))
    finally:
        channel.close()


class HistoryPool:
    def __init__(self,specs,cfg,native,mode,workers=2):
        from app.crossed_account_workers import ShardPool
        self.receive=ShardPool.receive
        self.channels=[];self.processes=[]
        if not specs or not 1<=workers<=4: raise ValueError('HISTORY_WORKERS')
        ctx=mp.get_context('spawn')
        try:
            for indices in np.array_split(np.arange(len(specs)),min(workers,len(specs))):
                parent,child=ctx.Pipe()
                p=ctx.Process(target=worker,args=(child,[specs[int(i)] for i in indices],cfg,str(native),mode),daemon=True)
                p.start();child.close();self.channels.append(parent);self.processes.append(p)
            for c in self.channels:
                if self.receive(c)[0]!='ready': raise ValueError('HISTORY_NOT_READY')
        except BaseException:
            self.close();raise

    def evaluate(self,genes,enabled=None,verify=False,details=False,voluntary=True):
        for c in self.channels: c.send((genes,enabled,verify,details,voluntary))
        rows=[]
        for c in self.channels:
            kind,values=self.receive(c)
            if kind!='ok': raise ValueError('HISTORY_RESPONSE')
            rows.extend(values)
        mean=float(np.mean([r['orders'] for r in rows]))
        return dict(objectives=[float(np.mean([r['return_pct'] for r in rows])),
            -max(r['drawdown_pct'] for r in rows)],mean_orders=mean,eligible=activity_eligible(mean),worlds=rows)

    def close(self):
        from app.crossed_account_workers import ShardPool
        ShardPool.close(self)
