"""Fixed semester-2 weights, seed 43: separate admission from ranking in 2024."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from scipy.stats import spearmanr
from role448_worker import Sell, attach_engine, bids, emit, file_sha, heads
from tools.role404_data import HS
from tools.role407_openings import decisions


def csv_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    root=Path('/var/tmp/role448-temporal-4plus1')
    run=Path('/var/tmp/role450-full-policy/run-43')
    out=Path('/var/tmp/role453-sell-audit');out.mkdir(exist_ok=True)
    read=lambda path:json.loads(path.read_text())
    manifest=root/'43.json';spec=read(manifest)
    for name,key in [('43.npz','inputs'),('43.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(root/name)==spec[key]
    assert torch.cuda.get_device_name()==read(run/'PREFLIGHT.json')['device']
    attach_engine(root/'engine')
    with gzip.open(root/'43.sim.gz','rb') as f:panel,ref=cloudpickle.load(f)[2024]
    assert panel.account(panel.raw)==ref
    with np.load(root/'43.npz') as z:
        x=torch.from_numpy(z['x8']).cuda();ix=z['i8'];ts=z['t8']
    net=Sell(read(root/'recipe.json'),read(root/'arch.json')).cuda().eval()
    eligible={};rank={};cuts={};admitted={};scores={};labels={}
    for s in (2,3):
        ck=torch.load(run/f's{s-1}-validation.pt',map_location='cpu',weights_only=False)
        assert ck['sig']==file_sha(manifest)
        if s==2:
            net.load_state_dict(ck['champ']['weights'])
            p=torch.cat([heads(net(b))[-1] for b in x.split(512)]).cpu().numpy()
        table=np.sort(ck['champ']['reference'],axis=0)
        q=np.maximum(0,np.quantile(table,.9,axis=0))
        eligible[s]=p>q;cuts[s]=q.tolist();admitted[s]=eligible[s].sum(axis=0).tolist()
        rank[s]=np.column_stack([np.searchsorted(table[:,j],p[:,j],'right')/len(table) for j in range(4)])
        assert np.array_equal(np.where(eligible[s],rank[s],0),bids(p,table))
    expected=read(Path('/var/tmp/role452-cross/43.json'))['rows'][1]
    assert expected['semester']==3
    lookup={int(b):i for i,b in enumerate(panel.bars[ix])}
    ledger=[];summary={}
    for gate,r in [(2,2),(3,3),(3,2),(2,3)]:
        tag=f'g{gate}r{r}';a=np.zeros_like(panel.raw);a[ix,1,:]=np.where(eligible[gate],rank[r],0)
        cmd,h=decisions(a);jj=np.where(cmd[ix]==2,h[ix],-1);labels[tag]=jj
        ac=panel.account(a);tr=panel.e.trades
        if gate==r:assert ac==expected['accounts'][f'weights_2_table_{r}']
        assert len(tr)==len({t['entry_bar_abs'] for t in tr})==ac['trades']
        assert abs(ac['net_usd']-sum(t['net_usd'] for t in tr))<1e-6
        scores[tag]=a[ix,1,:].max(axis=1)
        summary[tag]=dict(account=ac,signals=int((jj>=0).sum()),
                         routed=[int((jj==j).sum()) for j in range(4)],
                         exits=dict(Counter(t['exit_reason'] for t in tr)))
        for t in tr:
            bar=int(t['entry_bar_abs'])-1;i=lookup[bar];j=int(jj[i]);assert j>=0
            ledger.append(dict(policy=tag,bar=bar,decision_ns=int(ts[i]),horizon=HS[j],
                       raw=float(p[i,j]),rank=float(scores[tag][i]),entry=str(t['entry_time']),
                       exit=str(t['exit_time']),reason=t['exit_reason'],gross=t['gross_usd'],net=t['net_usd']))
    trades={tag:{row['bar'] for row in ledger if row['policy']==tag} for tag in summary}
    panel.e.reset();cap=panel.e.cfg.capital
    isolated=[];groups={}
    for i,(old,new) in enumerate(zip(labels['g2r2'],labels['g3r3'])):
        if max(old,new)<0:continue
        group=('added' if old<0 else 'removed' if new<0 else 'same' if old==new else 'rerouted')
        record=dict(bar=int(panel.bars[ix[i]]),decision_ns=int(ts[i]),group=group)
        for side,j in [('old',old),('new',new)]:
            horizon,raw,ranked,net,filled=0,None,None,None,False
            if j>=0:
                horizon=HS[j];raw=float(p[i,j]);ranked=float(scores['g2r2' if side=='old' else 'g3r3'][i])
                fill,cash,_=panel.isolated(record['bar'],2,int(j),cap)
                filled=bool(fill);net=cash-cap if filled else None
            record.update({f'{side}_h':horizon,f'{side}_raw':raw,f'{side}_rank':ranked,
                           f'{side}_net':net,f'{side}_filled':filled})
        isolated.append(record)
    for group in sorted({z['group'] for z in isolated}):
        zz=[z for z in isolated if z['group']==group];stats={}
        for side in ('old','new'):
            filled=[z for z in zz if z[side+'_filled']]
            if not filled:continue
            v=np.array([z[side+'_net'] for z in filled]);ranks=[z[side+'_rank'] for z in filled]
            corr=spearmanr(ranks,v).statistic if len(set(ranks))>1 and len(set(v))>1 else None
            stats[side]=dict(fills=len(v),wins=int((v>0).sum()),mean_usd=float(v.mean()),rank_net_spearman=corr)
        groups[group]=dict(signals=len(zz),isolated=stats)
    common=trades['g2r2']&trades['g3r3']
    fills={}
    for tag in ('g2r2','g3r3'):
        fills[tag]={}
        for subset,keys in [('common',common),('exclusive',trades[tag]-common)]:
            rows=[z for z in ledger if z['policy']==tag and z['bar'] in keys]
            fills[tag][subset]=dict(n=len(rows),net=sum(z['net'] for z in rows),
                           by_h={h:dict(n=sum(z['horizon']==h for z in rows),
                                        net=sum(z['net'] for z in rows if z['horizon']==h)) for h in HS})
    csv_rows(out/'trades.csv',ledger);csv_rows(out/'isolated.csv',isolated)
    emit(out/'DONE.json',dict(decisions=len(ix),horizons=HS,effective_cut=cuts,
           passed_by_horizon=admitted,policies=summary,signals=groups,fills=fills))


if __name__=='__main__':main()
