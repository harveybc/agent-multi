"""Figures and first descending zero crossings; no extrapolation or smoothing."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

STYLES=[('short','Sólo corto','#3673a6'),('long','Sólo largo','#b44b27'),('both','Ambos','#167468')]


def crossing(x,y):
    for i in range(len(x)-1):
        if y[i]>0 and y[i+1]<=0:
            return x[i],x[i+1],x[i]+(x[i+1]-x[i])*y[i]/(y[i]-y[i+1])
    return (np.nan,)*3


def save(fig,root,name):
    fig.savefig(root/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def figures(root):
    s=pd.read_csv(root/'sweep.csv.gz'); n=pd.read_csv(root/'noise.csv.gz')
    ctrl=pd.read_csv(root/'controls.csv'); thresholds=[]
    for metric,title,ylabel,scale in [
        ('return_pct','Beneficio neto frente al ruido de las predicciones estacionales','Beneficio neto (USD)',100),
        ('sharpe','Sharpe frente al ruido de las predicciones estacionales','Sharpe anualizado',1)]:
        fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
        fig.suptitle(title,fontsize=16)
        for (year,d),ax in zip(s.groupby('year'),axes):
            for key,label,color in STYLES:
                arm,xcol=(d[d.long==0],'short') if key=='short' else ((d[d.short==0],'long') if key=='long' else (d[d.short==d.long],'short'))
                pv=arm.pivot(index='seed',columns=xcol,values=metric)*scale
                x=pv.columns.to_numpy(); ys=pv.to_numpy(); y=ys.mean(axis=0)
                ax.plot(x,y,'o-',ms=3,color=color,label=label)
                lo,hi=np.quantile(ys,[.10,.90],axis=0)
                ax.fill_between(x,lo,hi,color=color,alpha=.13)
                a,b,pt=crossing(x,y)
                if np.isfinite(pt):ax.scatter([pt],[0],s=48,facecolor='white',edgecolor=color,zorder=4)
                if metric=='return_pct':
                    draws=np.random.default_rng(1463).integers(len(ys),size=(2000,len(ys)))
                    boot=np.array([crossing(x,z)[2] for z in ys[draws].mean(axis=1)])
                    finite=boot[np.isfinite(boot)]
                    thresholds.append([year,key,a,b,pt,*(np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)),len(finite)/len(boot)])
            ax.axhline(0,color='#273647',linewidth=.9);ax.axvline(1,color='#555',linestyle=':',label='MAE de persistencia')
            ax.set(title=str(year),xlabel='MAE esperado / MAE de persistencia',ylabel=ylabel)
            ax.grid(axis='y',alpha=.2)
        axes[0].legend(frameon=False,fontsize=8)
        fig.supxlabel('Media y percentiles 10–90 de 64 semillas · EURUSD horario real · cuenta inicial: 10.000 USD · costes incluidos',fontsize=9)
        save(fig,root,metric)
    th=pd.DataFrame(thresholds,columns=['year','arm','lo','hi','cross','mc025','mc975','bootstrap_crossed'])
    th.to_csv(root/'thresholds.csv',index=False)
    naive=n.drop_duplicates(['year','h'])[['year','h','naive','n']]
    out=th.merge(naive,on='year');out=out[(out.arm=='both')|((out.arm=='short')==(out.h<=5))]
    out['mae_at_cross']=out['naive']*out['cross'];out.to_csv(root/'absolute_thresholds.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True,sharey=True)
    norm=TwoSlopeNorm(vmin=s.groupby(['year','short','long']).return_pct.mean().min(),vcenter=0,vmax=100)
    for (year,d),ax in zip(s.groupby('year'),axes):
        tab=d.groupby(['short','long']).return_pct.mean().unstack()
        x=tab.columns.to_numpy();y=tab.index.to_numpy();v=tab.to_numpy()
        im=ax.pcolormesh(x,y,v,cmap='RdBu',norm=norm,shading='nearest')
        ax.contour(x,y,v,levels=[0],colors='black',linewidths=1.6)
        ax.axhline(1,ls=':',c='.3');ax.axvline(1,ls=':',c='.3')
        ax.set(title=str(year),xlabel='Ruido largo / naïve largo',ylabel='Ruido corto / naïve corto',xlim=(0,4),ylim=(0,4))
    fig.colorbar(im,ax=axes,label='Retorno neto medio (%)',shrink=.85)
    fig.suptitle('Respuesta conjunta al ruido · línea negra: retorno cero',fontsize=15)
    save(fig,root,'surface')
    fig,ax=plt.subplots(figsize=(8.5,4.3),constrained_layout=True)
    band=out[(out.year==2024)&(out.arm=='both')].sort_values('h')
    xs=np.arange(len(band));unit=1e4*band.naive.to_numpy()
    ax.plot(xs,unit,'o--',color='#54545f',label='Persistencia')
    ax.plot(xs,unit*band['cross'],'o-',color='#167468',label='Cruce conjunto gaussiano')
    ax.fill_between(xs,unit*band.mc025,unit*band.mc975,color='#167468',alpha=.18,label='IC Monte Carlo del cruce (95 %)')
    for mode,label,color in [('frozen','Congelado','#b44b27'),('tuned','Afinado','#3673a6')]:
        row=ctrl[(ctrl.year==2024)&(ctrl.control==mode)].iloc[0]
        ax.plot(xs,[1e4*row[f'mae_{h}'] for h in band.h],'.-',color=color,label=label)
    ax.set(xticks=xs,xticklabels=band.h,xlabel='Horizonte (horas)',ylabel='MAE estacional (pips)',
           title='2024 · Error observado y referencia de equilibrio por horizonte')
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    save(fig,root,'mae')


if __name__=='__main__':figures(Path(sys.argv[1]))
