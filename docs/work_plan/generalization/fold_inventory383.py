"""Read-only metadata census; a matching date/width never certifies lineage."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path

def order_hash(columns):
    return hashlib.sha256(json.dumps(columns, ensure_ascii=False, separators=(',',':')).encode()).hexdigest()

def end(period):
    if not isinstance(period,list) or len(period)!=2:
        return None
    try:
        start,stop=map(date.fromisoformat,period)
        return stop.isoformat() if start<stop else None
    except (ValueError,TypeError):
        return None

def audit(roots,expected):
    counts=Counter();matches=[];fail=[]
    for root in roots:
        if not root.is_dir():
            fail.append(str(root));continue
        for folder,_,files in os.walk(root,onerror=lambda e:fail.append(str(e.filename))):
            if 'RECIPE.json' not in files:continue
            counts['recipe_files']+=1
            try:
                d=json.loads((Path(folder)/'RECIPE.json').read_bytes())
                cols=d.get('list',{}).get('columns')
                if not isinstance(cols,list):counts['other_schema']+=1;continue
                if order_hash(cols)!=expected:counts['different_columns_or_order']+=1;continue
                sp=d.get('split',{});sc=d.get('scaler',{})
                row=dict(folder=folder,actor_train_end=end(sp.get('train') or sp.get('periods',{}).get('TRAIN')),
                         scaler_end=end(sc.get('fit_period')),selection_period=sp.get('selection_period'),
                         validation_end=end(sp.get('validation') or sp.get('periods',{}).get('VALIDATION')),
                         bundle=all(k in files for k in ('model.pt','MANIFEST.json','MODEL_CONFIG.json','PROBE.npz','RELOAD.json')))
                matches.append(row)
            except (ValueError,OSError,AttributeError,TypeError):
                counts['unreadable_recipe']+=1;fail.append(folder)
    return dict(counts=counts,matches=matches,inaccessible=fail)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('expected')
    p.add_argument('roots',nargs='+',type=Path)
    a=p.parse_args()
    print(json.dumps(audit([r.expanduser() for r in a.roots],a.expected)))
