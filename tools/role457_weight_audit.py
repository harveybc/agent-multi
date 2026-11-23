"""Read-only parameter ledger and within-lineage weight-space audit, 448/450/451."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from role448_worker import Sell,bids,heads,file_sha
from role405_plateau import POLICY

@torch.inference_mode()
def main():
    root=Path('/var/tmp/role448-temporal-4plus1')
    out=Path('/var/tmp/role457-audit');out.mkdir(exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    ck=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    torch.set_num_threads(1)
    recipe=read(root/'recipe.json');arch=read(root/'arch.json');net=Sell(recipe,arch).eval()
    assert recipe['list']['columns']==recipe['scaler']['features']==[c['tensor_name'] for c in recipe['tensor']['columns']]
    ledger=[];changes=[]
    for seed in (7,19,43):
        spec=read(root/f'{seed}.json')
        assert file_sha(root/f'{seed}.npz')==spec['inputs']
        with np.load(root/f'{seed}.npz') as z:
            shapes=[list(z[f'x{i}'].shape) for i in range(9)]
            assert all(sh[1:]==[13,61] and sh[0]==spec['blocks'][str(i)]['decisions'] for i,sh in enumerate(shapes))
            witness=torch.from_numpy(z['x0'][::8].copy())
        for camp,path in [(448,root),(450,Path('/var/tmp/role450-full-policy')),(451,Path('/var/tmp/role451-activity'))]:
            d=path/f'run-{seed}'
            if not d.exists():continue
            for f in sorted(d.glob('s*-*.pt')):
                s=int(f.stem.split('-')[0][1:]);arm=f.stem.split('-')[1]
                a=ck(f);assert a['sig']==file_sha(root/f'{seed}.json')
                h=a['history'];best=a['best'];ch=a['champ'];v=h[best]['val']
                g=ch['optimizer']['param_groups'][0]
                src='reused' if (camp==450 and s==0) or (camp==451 and s==4) else 'trained'
                n=spec['blocks'][str(s)]['labelled'];b=(n+511)//512
                ledger.append(dict(campaign=camp,seed=seed,semester=s+1,arm=arm,source=src,
                    train_labelled=n,features=shapes[s][2],window=shapes[s][1],val_windows=shapes[8][0],
                    batches=b,batch_min=n//b,batch_max=(n+b-1)//b,epochs_recorded=len(h)-1,best_epoch=best,
                    stop=read(f.with_suffix('.json'))['stop'],lr_start=h[0]['lr'],lr_champ=g['lr'],
                    lr_last=h[-1]['lr'],wd=g['weight_decay'],rollbacks=sum(bool(r.get('rollback')) for r in h),
                    val_return=v['ret'],val_trades=v['trades'],val_sharpe=v['sharpe'],checkpoint_sha=file_sha(f)))
                if camp!=451 or s<5:continue
                prev=ck(d/f's{s-1}-validation.pt')['champ']
                wa,wb=prev['weights'],ch['weights'];diff={}
                for group in ['encoder.branches','encoder.fusion','encoder.post_fusion','encoder.core','head']:
                    names=[k for k,_ in net.named_parameters() if k.startswith(group+'.')]
                    a0=torch.cat([wa[k].double().flatten() for k in names])
                    a1=torch.cat([wb[k].double().flatten() for k in names])
                    diff[group]=float((a1-a0).norm()/a0.norm())
                def surface(w):
                    net.load_state_dict(w);return heads(net(witness))[-1].numpy()
                p,q=surface(wa),surface(wb)
                flips=lambda t:int(np.count_nonzero((bids(p,prev['reference'])>0)!=(bids(t,prev['reference'])>0)))
                swaps={}
                for group,delta in diff.items():
                    mixed={k:(wb[k] if k.startswith(group+'.') else v) for k,v in wa.items()}
                    t=surface(mixed) if delta else p
                    swaps[group]=dict(mean_abs=float(abs(p-t).mean()),flips=flips(t))
                changes.append(dict(semester=s+1,relative_l2=diff,same_state=all(torch.equal(wa[k],wb[k]) for k in wa),
                      n_windows=len(witness),score_max_change=float(abs(p-q).max()),
                      mean_abs_score_change=float(abs(p-q).mean()),
                      admission_flips_fixed_table=flips(q),single_module_swaps=swaps))
    with (out/'checkpoint-ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=ledger[0],lineterminator='\n');w.writeheader();w.writerows(ledger)
    detail=dict(checkpoints=len(ledger),arch=arch,feature_names=recipe['list']['columns'],recipe_regularization=recipe['regularization'],
        actual_dropout=sorted({m.p for m in net.modules() if isinstance(m,torch.nn.Dropout)}),
        params=sum(p.numel() for p in net.parameters()),scheduler=POLICY,changes=changes,
        input_manifests={s:read(root/f'{s}.json') for s in (7,19,43)},split=recipe['split'])
    (out/'AUDIT.json').write_text(json.dumps(detail,indent=2,allow_nan=False))


if __name__=='__main__':main()
