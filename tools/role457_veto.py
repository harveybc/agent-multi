"""2x2 ablation: label duration vs veto duration, expanding TRAIN folds."""
import copy
import numpy as np
from tools.role456_train_close import context,fit


def fork(e,b,j,native,rule):
    memo={id(v):v for v in vars(e).values() if isinstance(v,np.ndarray)}
    memo.update({id(e.trades):[],id(e.decisions):[]})
    alt=copy.deepcopy(e,memo)
    alt.step(b,0)
    while alt.pos is not None:
        k=alt.next_decision;assert k<alt.last
        cmd=native(alt,k,j)
        if cmd and rule is not None and rule(context(alt,k,j))<0:cmd=0
        alt.step(k,cmd)
    t,=alt.trades
    return t


class Trial:
    def __init__(self):
        self.past=[];self.one=None;self.long=None;self.targets={}

    def capture(self,e,b,j,native,x):
        one=fork(e,b,j,native,None)
        alt={}
        for name,fn in [('long',self.long),('one',self.one)]:
            if fn is not None and fn(x)<0:alt[name]=fork(e,b,j,native,fn)
        self.targets[e.pos.entry_bar]=(one,alt)

    def finish(self,s,panel,scores,pairs,long,features,row,end):
        native=panel.close
        if s:
            for target,fn in [('long',long),('one',self.one)]:
                for duration in ['lock','tick']:
                    lock=set();trigger=[0]
                    def gate(e,b,j):
                        if e.pos.entry_bar in lock:return 0
                        cmd=native(e,b,j)
                        if cmd and fn(context(e,b,j))<0:
                            trigger[0]+=1
                            if duration=='lock':lock.add(e.pos.entry_bar)
                            return 0
                        return cmd
                    panel.close=gate
                    try:account=panel.account(scores)
                    finally:del panel.close
                    differences=[]
                    for r in pairs:
                        bar=r['entry_bar']
                        if fn(features[bar])>=0:continue
                        alt=self.targets[bar][1]
                        pnl=r['hold_net'] if duration=='lock' else alt[target]['net_usd']
                        differences.append(pnl-r['early_net'])
                    row[target+'_'+duration]=dict(account=account,requests=trigger[0],pairs=len(differences),
                          saved=sum(max(0,v) for v in differences),lost=-sum(min(0,v) for v in differences))
                    if target=='long' and duration=='lock':
                        assert account==row['trial'] and len(differences)==row['paired_vetoes']
        for r in pairs:
            t=self.targets[r['entry_bar']][0]
            assert t['entry_bar_abs']==r['entry_bar'] and t['units']==r['units']
            assert t['exit_time'].value<end and t['exit_time'].value>=np.datetime64(r['early_exit'][:19],'ns').astype('int64')
            risk=r['units']*abs(r['price']-r['sl'])
            self.past.append((features[r['entry_bar']],(r['early_net']-t['net_usd'])/risk,
                                  r['delta_usd']/risk))
        self.one=fit([(x,y) for x,y,_ in self.past]);self.long=fit([(x,y) for x,_,y in self.past])
        self.targets.clear()
