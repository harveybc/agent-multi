"""Bounded persistent CPU shards for one candidate at a time."""
import multiprocessing as mp
import numpy as np


def _worker(connection,specs,config,native):
    try:
        from pathlib import Path
        from app.zigzag_opportunities import engine
        engine(Path(native))
        from app.crossed_account_panel import CrossedWorld
        from app.dynamic_account_panel import evaluate
        worlds=[CrossedWorld(s,config) for s in specs]
        connection.send(('ready',len(worlds)))
        while True:
            request=connection.recv()
            if request is None: break
            genes,enabled,verify,details=request
            connection.send(('ok',evaluate(worlds,genes,enabled,verify,details)['worlds']))
    except Exception as exc:
        connection.send(('error',(type(exc).__name__,str(exc))))
    finally:
        connection.close()


class ShardPool:
    """Fixed shard ownership keeps bounded caches local and aggregation ordered."""
    def __init__(self,specs,config,native,workers=4):
        if not specs or not 1<=workers<=4: raise ValueError('WORKER_BUDGET')
        context=mp.get_context('spawn');self.channels=[];self.processes=[]
        try:
            for indices in np.array_split(np.arange(len(specs)),min(workers,len(specs))):
                parent,child=context.Pipe()
                process=context.Process(target=_worker,args=(child,[specs[int(i)] for i in indices],config,str(native)),daemon=True)
                process.start();child.close();self.channels.append(parent);self.processes.append(process)
            for channel in self.channels:
                kind,_=self.receive(channel)
                if kind!='ready': raise ValueError('WORKER_NOT_READY')
        except BaseException:
            self.close();raise

    @staticmethod
    def receive(channel):
        if not channel.poll(600): raise TimeoutError('SCENARIO_WORKER_STALLED')
        kind,value=channel.recv()
        if kind=='error': raise RuntimeError(('SCENARIO_WORKER_FAILED',value))
        return kind,value

    def evaluate(self,genes,enabled=None,verify=False,details=False):
        for channel in self.channels: channel.send((genes,enabled,verify,details))
        rows=[]
        for channel in self.channels:
            kind,result=self.receive(channel)
            if kind!='ok': raise ValueError('WORKER_RESPONSE')
            rows.extend(result)
        mean_orders=float(np.mean([r['orders'] for r in rows]))
        return dict(objectives=[float(np.mean([r['return_pct'] for r in rows])),
                               -max(r['drawdown_pct'] for r in rows)],
                    eligible=mean_orders>=3,mean_orders=mean_orders,worlds=rows)

    def close(self):
        for channel in self.channels:
            try: channel.send(None)
            except (OSError,EOFError): pass
        for process in self.processes:
            process.join(10)
            if process.is_alive(): process.terminate();process.join(10)
        for channel in self.channels: channel.close()
        self.channels=[];self.processes=[]
