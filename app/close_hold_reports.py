"""Prefix-only diagnostic reporter. Not a trained or selected forecasting model."""
import numpy as np

from app.close_hold_controller import PathReport


def prefix_report(completed_bars,decision_time,step,deadline,current_spread,*,terminal_spread):
    x=np.asarray(completed_bars,float)
    if (x.ndim!=2 or x.shape[1]!=4 or len(x)<2 or not np.isfinite(x).all()
            or step<=0 or deadline<=decision_time or (deadline-decision_time)%step
            or not np.isfinite(current_spread) or current_spread<0
            or not np.isfinite(terminal_spread) or terminal_spread<0):
        raise ValueError('CAUSAL_PREFIX_CONTRACT')
    close=x[-8:,3]
    changes=np.diff(close)
    drift=float(np.median(changes)); dispersion=float(np.std(changes))
    h=(deadline-decision_time)//step
    offsets=np.arange(h,dtype=float)
    means=close[-1]+drift*offsets
    centers=means[None,:]+np.array([-1.,0.,1.])[:,None]*dispersion*np.sqrt(offsets)[None,:]
    width=max(float(np.median(x[-8:,1]-x[-8:,2]))/2,dispersion*.25)
    paths=np.stack([centers,centers+width,centers-width,centers],axis=2)
    return PathReport(int(decision_time),int(decision_time),int(decision_time),int(step),int(deadline),
                      paths,np.full((3,h),current_spread),np.full(3,1/3),'causal',
                      np.full(3,terminal_spread))
