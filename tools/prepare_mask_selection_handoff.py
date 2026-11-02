"""Authenticate DEAP closure and prepare a briefing; never launch selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(root):
    close=root/'CLOSE.json'
    if not close.exists(): return {'state':'WAIT_DEAP_CLOSE','phase2_launched':False}
    s=read(root/'SPEC.json'); c=read(close)
    if s['identity']!=digest({k:v for k,v in s.items() if k!='identity'}) or c['identity']!=s['identity']:
        raise ValueError('HANDOFF_IDENTITY')
    result=dict(schema='mask_selection_handoff.v1',campaign_identity=s['identity'],
        close_sha256=sha(close),spec_sha256=sha(root/'SPEC.json'),phase2_launched=False,
        requires_prelaunch_briefing=True,hyperparameters=c['best']['config'] if c['best'] else None,
        state='READY_FOR_BRIEFING' if c['best'] else 'NO_ELIGIBLE_CHAMPION',checkpoints=[])
    if c['best']:
        best=c['best']
        if not best.get('eligible'): raise ValueError('HANDOFF_INELIGIBLE')
        matched=False
        for p in sorted((root/'generations').glob('*.json')):
            g=read(p)
            if g['identity']!=s['identity']: raise ValueError('HANDOFF_GENERATION_IDENTITY')
            if best in g['results']: matched=True
        if not matched: raise ValueError('HANDOFF_CHAMPION_NOT_EVALUATED')
        scores=[]
        for seed in s['design']['seeds']:
            key=digest(dict(config=best['config'],seed=seed)); candidates=[]
            for p in sorted((root/'collected').glob('*/runs/'+key+'/attempt*/RESULT.json')):
                r=read(p)
                if r['identity']!=s['identity'] or r['config']!=best['config'] or r['seed']!=seed:
                    raise ValueError('HANDOFF_RECEIPT_IDENTITY')
                if not r['eligible']: continue
                if not {'best.pt','BEST.json','EFFECTIVE.json','HISTORY.json','TRAINING.json'}<=r['artifacts'].keys():
                    raise ValueError('HANDOFF_MISSING_ARTIFACTS')
                for name,expected in r['artifacts'].items():
                    path=(p.parent/name).resolve()
                    if not path.is_relative_to(p.parent.resolve()) or sha(path)!=expected:
                        raise ValueError('HANDOFF_ARTIFACT_CHANGED')
                checkpoint=read(p.parent/'BEST.json')
                if checkpoint['score']!=r['score'] or checkpoint['step']!=r['selected_step']:
                    raise ValueError('HANDOFF_CHECKPOINT_SCORE')
                candidates.append((p,r))
            if len(candidates)!=1: raise ValueError('HANDOFF_SEED_MISSING_OR_AMBIGUOUS')
            p,r=candidates[0]; scores.append(r['score'])
            result['checkpoints'].append(dict(seed=seed,path=str(p.parent/'best.pt'),
                sha256=r['artifacts']['best.pt'],receipt=str(p),receipt_sha256=sha(p),
                selected_step=r['selected_step'],score=r['score'],metrics=r['metrics']))
        if not all(math.isfinite(x) for x in scores) or not math.isclose(sum(scores)/len(scores),best['score'],abs_tol=1e-12):
            raise ValueError('HANDOFF_COMBINED_SCORE')
    destination=root/'MASK_SELECTION_HANDOFF.json'
    if destination.exists():
        if read(destination)!=result: raise ValueError('HANDOFF_EXISTING_CHANGED')
    else:
        temp=destination.with_suffix('.tmp')
        with temp.open('w') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        temp.replace(destination)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    r=prepare(p.parse_args().root)
    print(json.dumps({'state':r['state'],'phase2_launched':False}))
