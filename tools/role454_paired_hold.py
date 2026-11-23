"""Fork real filled SELL positions; compare learned CLOSE with broker-only exits."""
import copy
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import cloudpickle
import numpy as np
from app.frozen_recipe_inputs import file_sha
from app.recipe_runtime import attach_engine
from tools.role404_data import HS
from tools.role453_sell_audit import csv_rows


def tally(rows):
    delta=np.array([r['delta_usd'] for r in rows]);tol=1e-8
    return dict(n=len(rows),helped=int((delta>tol).sum()),hurt=int((delta < -tol).sum()),
                tied=int((abs(delta)<=tol).sum()),early=sum(r['early_net'] for r in rows),
                hold=sum(r['hold_net'] for r in rows),delta=float(delta.sum()),
                median=float(np.median(delta)),protected=float(delta[delta>tol].sum()),
                sacrificed=float(-delta[delta < -tol].sum()))


def main():
    root=Path('/var/tmp/role448-temporal-4plus1');src=Path('/var/tmp/role453-sell-audit')
    out=Path('/var/tmp/role454-paired-hold');out.mkdir(exist_ok=True)
    archive=json.loads((src/'DONE.json').read_text())['policies']['g3r3']
    assert file_sha(root/'43.sim.gz')==json.loads((root/'43.json').read_text())['sim']
    attach_engine(root/'engine')
    with gzip.open(root/'43.sim.gz','rb') as f:panel,ref=cloudpickle.load(f)[2024]
    assert panel.account(panel.raw)==ref
    with (src/'isolated.csv').open() as f:sigs=[r for r in csv.DictReader(f) if int(r['new_h'])]
    with (src/'trades.csv').open() as f:ledger=[r for r in csv.DictReader(f) if r['policy']=='g3r3']
    horizon={int(r['bar']):HS.index(int(r['new_h'])) for r in sigs}
    assert len(horizon)==len(sigs)==archive['signals'] and set(horizon.values())=={HS.index(12)}
    scores=np.zeros_like(panel.raw)
    for b,j in horizon.items():scores[panel.ixbar[b],1,j]=1.
    actual=panel.account(scores);assert actual==archive['account']
    e=panel.e;reference=list(e.trades);assert len(reference)==len(ledger)==676
    for t,r in zip(reference,ledger):
        assert (t['entry_bar_abs']-1,str(t['entry_time']),str(t['exit_time']),t['exit_reason'],
                t['gross_usd'],t['net_usd'])==(int(r['bar']),r['entry'],r['exit'],r['reason'],float(r['gross']),float(r['net']))
    e.reset();counter={};pairs=[]
    # Fork after every surviving fill, not just those known to close early.
    # Market arrays are read-only during execution; mutable position/account state is copied.
    for b in range(e.first,e.last):
        cmd=0;plan=None
        if e.pos is None:
            if b in horizon and e.entry_gate(b)[0]:
                j=horizon[b];env=panel.envs[j];env.bar=b;cmd=2;plan=env.entry_plan(-1)
        else:cmd=panel.close(e,b,j)
        balance=e.balance
        info=e.step(b,cmd,entry_plan=plan)
        if info['entry'] and e.pos is not None:
            memo={id(v):v for v in vars(e).values() if isinstance(v,np.ndarray)}
            # Past append-only logs are not inputs to the exit rule.
            memo.update({id(e.trades):[],id(e.decisions):[]})
            alt=copy.deepcopy(e,memo)
            while alt.pos is not None:
                assert alt.next_decision<alt.last
                alt.step(alt.next_decision,0)
            t,=alt.trades
            assert t['exit_reason'] in ('SL','TP','TIME') and t['entry_bar_abs']==b+1
            assert abs(alt.balance-balance-t['net_usd'])<1e-7
            counter[b+1]=t
    assert e.trades==reference
    for early in reference:
        t=counter.get(early['entry_bar_abs'],early)
        if early['exit_reason']!='EARLY':
            assert t==early
            continue
        assert early['entry_bar_abs'] in counter
        for key in ('entry_bar_abs','entry_time','entry_price','units','direction','take_profit','stop_loss','entry_cost_usd'):
            assert early[key]==t[key]
        assert early['exit_time']<=t['exit_time']
        pairs.append(dict(entry=str(t['entry_time']),entry_bar=t['entry_bar_abs'],units=t['units'],
                  price=t['entry_price'],sl=t['stop_loss'],tp=t['take_profit'],
                  early_exit=str(early['exit_time']),hold_exit=str(t['exit_time']),hold_reason=t['exit_reason'],
                  early_net=early['net_usd'],hold_net=t['net_usd'],delta_usd=early['net_usd']-t['net_usd']))
    assert len(pairs)==archive['exits']['EARLY']==559
    # Separate account-level counterfactual: later entries and sizes can change.
    panel.close=lambda engine,b,j:0
    alternative=panel.account(scores)
    assert all(t['exit_reason'] in ('SL','TP','TIME') for t in panel.e.trades)
    assert abs(alternative['net_usd']-sum(t['net_usd'] for t in panel.e.trades))<1e-6
    result=dict(seed=43,policy='g3r3 crossed diagnostic',factual=actual,broker_only=alternative,
                broker_only_exits=dict(Counter(t['exit_reason'] for t in panel.e.trades)),
                pairs=tally(pairs),by_hold_reason={k:tally([p for p in pairs if p['hold_reason']==k])
                for k in sorted({p['hold_reason'] for p in pairs})},parity_trades=len(reference))
    csv_rows(out/'paired.csv',pairs)
    tmp=out/'result.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(out/'DONE.json')


if __name__=='__main__':main()
