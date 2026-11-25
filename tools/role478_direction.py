"""Out-of-year directional value conditional on forecast strength, no parameter search."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role470_barriers import probe
from tools.role471_net_selector import outcomes, audit_book
from tools.role477_accounts import opportunity_panel

H=(1,2,3,4,5,6,24,48)
LIVE_H=(6,24,48)


def quartiles(signal, calibration):
    signal=np.asarray(signal)
    if not np.isfinite(signal).all():
        raise ValueError('nonfinite signal')
    side=np.sign(signal).astype(int); bins=np.zeros(len(side),int);cuts=[]
    for sign in (-1,1):
        take=side==sign
        ref=abs(signal[take & calibration])
        if len(ref)<4:raise ValueError('missing calibration side')
        q=np.quantile(ref,[.25,.5,.75])
        if not (np.diff(q)>0).all(): raise ValueError('degenerate calibration')
        bins[take]=1+np.searchsorted(q,abs(signal[take]),side='right')
        cuts.append(dict(side=sign,calibration_rows=len(ref),q25=q[0],q50=q[1],q75=q[2]))
    return side,bins,cuts


def load(root,year):
    ts=None;pred=[];hashes={}
    for task in ('short','long'):
        d=root/str(year)/task
        with (d/'pred.npz').open('rb') as f:
            hashes[task]=hashlib.file_digest(f,'sha256').hexdigest()
        done=json.loads((d/'done.json').read_text())
        if done['year']!=year or done['task']!=task or done['predictions']!=hashes[task]:
            raise ValueError('prediction identity')
        with np.load(d/'pred.npz') as ar:
            if ts is not None:np.testing.assert_array_equal(ts,ar['ts'])
            ts=ar['ts'];pred.append(ar['pred'])
    return ts,np.column_stack(pred),hashes


def preferences(v):
    k=v.argmax(1)
    b=v[:,:3].max(1);s=v[:,3:].max(1)
    return np.where((v.max(1)>0)&(b!=s),np.where(k<3,1,-1),0)


def one(a,year):
    ts,pred,hashes=load(a.root,year)
    idx,fit,basic,sig,data,costs,fees,_=opportunity_panel(ts,pred,year,a.hist,a.market,a.bundle)
    t,o,hi,lo,c,atr,_,_,_,dead=data;_,spread,roll=costs
    pnl,end=outcomes(idx,t,o,hi,lo,atr,dead,spread,roll,fees,1.,1.,H)
    ev=~fit; ii=idx[ev];tm=t[ii]
    assert t[end[fit].max()]<tm.min() and t[end[ev].max()]<pd.Timestamp(str(year+1),tz='UTC').value
    records=[]; thresholds=[];checks=0
    for k,h in enumerate(H):
        side,quart,cuts=quartiles(sig[:,k],fit)
        thresholds.extend(dict(year=year,h=h,**r) for r in cuts)
        buy=pnl[ev,k];sell=pnl[ev,k+len(H)]
        side=side[ev];q=quart[ev]
        ref=np.where(side==1,buy,sell);opp=np.where(side==1,sell,buy)
        records.append(pd.DataFrame(dict(year=year,ts=tm,h=h,side=side,q=q,
            signal_atr=sig[ev,k],buy=buy,sell=sell,chosen=np.where(side!=0,ref,0.),
            opposite=np.where(side!=0,opp,np.nan),delta=np.where(side!=0,ref-opp,np.nan))))
        for j in np.linspace(0,len(ii)-1,9,dtype=int):
            i=ii[j];width=atr[i-1]*np.sqrt(h)
            for sign,col in [(1,k),(-1,k+len(H))]:
                row=probe(i,sign,width,width,1.,h,data,costs)
                np.testing.assert_allclose(row['usd'],pnl[ev,col][j],rtol=0,atol=1e-8)
                assert row['end']==end[ev,col][j];checks+=1
    panel=pd.concat(records,ignore_index=True)
    src=a.root/str(year)
    saved=pd.read_csv(src/'book.csv'); stats=json.loads((src/'accounts.json').read_text())
    assert len(ii)==stats['context']['eval_origins']
    cols=[H.index(h)+off for off in (0,len(H)) for h in LIVE_H]
    p6=pnl[:,cols];e6=end[:,cols];attrib=[];decomp=[];entries=[]
    for arm,x in [('context',basic),('seasonal',np.column_stack([basic,sig,abs(sig)]))]:
        x0=x[fit];sd=x0.std(0);sd[sd<1e-10]=1.
        xx=(x-x0.mean(0))/sd;off=p6[fit].mean(0)
        weights=np.linalg.solve(xx[fit].T@xx[fit]+10*np.eye(x.shape[1]),xx[fit].T@(p6[fit]-off))
        scores=xx[ev]@weights+off
        variants=[('full',scores),('intercept_only',np.broadcast_to(off,scores.shape)),
                  ('centered_terms_only',scores-off)]
        for label,sl in [('clock',slice(0,5)),('volatility',slice(5,7)),('cost_over_atr',slice(7,9)),
                         ('forecast',slice(9,None))]:
            if xx[:,sl].shape[1]:
                variants.append(('without_'+label,scores-xx[ev,sl]@weights[sl]))
        for component,v in variants:
            prefer=preferences(v)
            decomp.append(dict(year=year,arm=arm,component=component,
                               buy=int(sum(prefer==1)),sell=int(sum(prefer==-1)),nop=int(sum(prefer==0)),
                               **{f'fit_mean_{s}_{h}':off[j] for j,(s,h) in
                                  enumerate((s,h) for s in ('buy','sell') for h in LIVE_H)}))
        book,ret=audit_book((ii,scores,p6[ev],e6[ev]),(1,1,0,0,15,1),t)
        orig=saved[saved.selector==arm].reset_index(drop=True);fresh=pd.DataFrame(book)
        for col in ('entry_ns','side','hours','exit_bar_open_ns'):
            np.testing.assert_array_equal(orig[col],fresh[col])
        for col in ('net_usd','q','edge_bp','balance'):
            np.testing.assert_allclose(orig[col],fresh[col],rtol=1e-11,atol=1e-8)
        np.testing.assert_allclose(stats[arm]['return_pct'],ret,atol=1e-8)
        # Of the proposals excluded by the single-slot rule, distinguish a
        # direction never proposed from one that never saw a free account.
        pref=preferences(scores)
        entry=fresh.entry_ns.to_numpy()
        exit_ns=fresh.exit_bar_open_ns.to_numpy()
        prior=np.searchsorted(entry,tm,side='left')-1
        busy=(prior>=0)&(tm<=exit_ns[prior.clip(min=0)])
        np.testing.assert_array_equal(tm[(pref!=0)&~busy],entry)
        for row in decomp:
            if row['arm']==arm and row['component']=='full':
                row.update(buy_busy=int(sum((pref==1)&busy)),sell_busy=int(sum((pref==-1)&busy)))
        joined=orig.merge(panel,left_on=['entry_ns','hours'],right_on=['ts','h'],suffixes=('_exec','_signal'),
                          validate='one_to_one')
        assert len(joined)==len(orig)
        sign=np.where(joined.side_exec=='buy',1,-1)
        same=sign==joined.side_signal
        verified=np.where(sign==1,joined.buy,joined.sell)*(joined.balance-joined.net_usd)*joined.q_exec/10000
        np.testing.assert_allclose(verified,joined.net_usd,atol=1e-8)
        joined['agrees']=same;entries.append(joined)
        for group,frame in [('all',joined),('agree',joined[same]),('disagree',joined[~same])]:
            attrib.append(dict(year=year,arm=arm,group=group,n=len(frame),
                buy=int(sum(frame.side_exec=='buy')),sell=int(sum(frame.side_exec=='sell')),
                sign_buy=int(sum(frame.side_signal==1)),sign_sell=int(sum(frame.side_signal==-1)),
                wins=int(sum(frame.net_usd>0)),net_usd=float(frame.net_usd.sum())))
    return panel,thresholds,attrib,decomp,pd.concat(entries),dict(year=year,hashes=hashes,
        engine_checks=checks,verified_fills=len(saved),neutral=int((panel.side==0).sum()))


def aggregate(panel):
    # Descriptive interval: preserve contemporaneous opportunities and nearby weeks.
    cuts=[];rng=np.random.default_rng(478)
    panel=panel[panel.side!=0].copy()
    panel['week']=pd.to_datetime(panel.ts,utc=True).dt.tz_localize(None).dt.to_period('W-SUN').astype('int64')
    counts=panel.groupby(['year','week','h','side','q']).agg(n=('delta','size'),
        chosen=('chosen','sum'),delta=('delta','sum'),wins=('chosen',lambda v:(v>0).sum()))
    counts=counts.unstack(['h','side','q']).fillna(0)
    cols=pd.MultiIndex.from_product([['n','chosen','delta','wins'],H,[-1,1],range(1,5)])
    counts=counts.reindex(columns=cols,fill_value=0)
    for year in [*sorted(panel.year.unique()),'all']:
        samp=counts if year=='all' else counts.loc[[year]]
        years=list(samp.index.get_level_values('year').unique())
        direct=np.zeros((4,len(H),2,4))
        boot=np.zeros((3000,4,len(H),2,4))
        for y in years:
            wk=samp.loc[y]
            wk=wk.reindex(range(wk.index.min(),wk.index.max()+1),fill_value=0)
            mat=wk.to_numpy();n=len(mat)
            starts=rng.integers(n,size=(len(boot),(n+3)//4))
            draw=((starts[:,:,None]+np.arange(4))%n).reshape(len(boot),-1)[:,:n]
            freq=np.zeros((len(boot),n),dtype=int)
            np.add.at(freq,(np.arange(len(boot))[:,None],draw),1)
            direct+=mat.sum(0).reshape(direct.shape)
            boot+=(freq@mat).reshape(boot.shape)
        for j,h in enumerate(H):
            for side in (-1,1,0):
                v=direct[:,j];b=boot[:,:,j]
                if side:
                    v=v[:,int(side>0)];b=b[:,:,int(side>0)]
                else:v=v.sum(1);b=b.sum(2)
                with np.errstate(invalid='ignore',divide='ignore'):
                    for name,k in [('net',1),('mirror_delta',2)]:
                        means=v[k]/v[0];bm=b[:,k]/b[:,0]
                        contrast=bm[:,3]-bm[:,0]
                        lo,hi=np.nanquantile(contrast,[.025,.975])
                        strong_lo,strong_hi=np.nanquantile(bm[:,3],[.025,.975])
                        cuts.append(dict(year=year,h=h,side=side,measure=name,
                            **{f'q{i+1}':means[i] for i in range(4)},
                            **{f'n{i+1}':int(v[0,i]) for i in range(4)},
                            q4_win_pct=100*v[3,3]/v[0,3],q4_lo=strong_lo,q4_hi=strong_hi,
                            q4_minus_q1=means[3]-means[0],gradient_lo=lo,gradient_hi=hi))
    return pd.DataFrame(cuts)


def matched(panel):
    # Diagnostic only: other market days in the same week and UTC hour.
    x=panel.loc[panel.side!=0,['year','ts','h','q','side','chosen']].copy()
    date=pd.to_datetime(x.ts,utc=True)
    x['week']=date.dt.tz_localize(None).dt.to_period('W-SUN').astype('int64')
    x['hour']=date.dt.hour
    keys=['year','h','side','week','hour']
    peers=x[x.q<4].groupby(keys).chosen.agg(peer='mean',peers='size')
    high=x[x.q==4].join(peers,on=keys)
    high['difference']=high.chosen-high.peer
    result=high.groupby(['year','h','side']).agg(strong=('chosen','size'),
        matched=('difference','count'),matched_q4_minus_weaker=('difference','mean')).reset_index()
    return result


def figure(stats,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True,sharex=True,sharey=True)
    for j,(metric,title) in enumerate([('net','Neto frente a NOP'),('mirror_delta','Ventaja frente al lado opuesto')]):
        mats=[stats[(stats.year=='all')&(stats.side==side)&(stats.measure==metric)].set_index('h')
              .loc[list(H),['q1','q2','q3','q4']].to_numpy() for side in (1,-1)]
        lim=max(abs(m).max() for m in mats)
        for i,m in enumerate(mats):
            ax=axes[i,j]
            im=ax.imshow(m,aspect='auto',vmin=-lim,vmax=lim,cmap='RdBu')
            ax.set_title(('BUY' if i==0 else 'SELL')+' | '+title)
            ax.set_xticks(range(4),['Q1','Q2','Q3','Q4'])
            ax.set_yticks(range(len(H)),[str(h)+' h' for h in H])
            for (row,col),v in np.ndenumerate(m):
                ax.text(col,row,f'{v:.1f}',ha='center',va='center',
                        color='white' if abs(v)>.6*lim else 'black')
        fig.colorbar(im,ax=axes[:,j],label='pb de cuenta por oportunidad')
    fig.suptitle('EURUSD 2020-2023 | Intensidad del cambio estacional previsto')
    fig.savefig(dest,dpi=135);plt.close(fig)


def main(a):
    packs=[one(a,y) for y in range(2020,2024)]
    panel=pd.concat([p[0] for p in packs],ignore_index=True)
    a.out.mkdir(parents=True,exist_ok=True)
    stats=aggregate(panel)
    stats.to_csv(a.out/'strength.csv',index=False)
    matched(panel).to_csv(a.out/'matched.csv',index=False)
    figure(stats,a.out/'strength.png')
    for i,name in [(1,'cuts'),(2,'entries'),(3,'preferences')]:
        pd.DataFrame([r for p in packs for r in p[i]]).to_csv(a.out/(name+'.csv'),index=False)
    (a.out/'audit.json').write_text(json.dumps([p[5] for p in packs],indent=2)+'\n')
    if a.detail:
        a.detail.mkdir(parents=True,exist_ok=True)
        panel.to_parquet(a.detail/'opportunities.parquet',index=False)
        pd.concat([p[4] for p in packs]).to_parquet(a.detail/'filled.parquet',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','hist','market','bundle','out'):p.add_argument(k,type=Path)
    p.add_argument('--detail',type=Path)
    main(p.parse_args())
