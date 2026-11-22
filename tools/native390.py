"""Gated mechanical replay of the frozen policy; no optimization or validation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from app.frozen_actor_compaction import ConstantResponse, effective_actor
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import subset_recipe
from app.selected_input_market import market_for
from app.structural_refit import load_bundle
from app.three_signal_learning import environment, evaluate
from tools.audit_g3_producers import write
from tools.prepare_g3_joint import synthetic_market


def reduced_policy(model, cols):
    actor = effective_actor(model)
    enc = actor.encoder
    names = model.recipe['list']['columns']
    remap = {names.index(n):i for i,n in enumerate(cols)}
    # The ten removed columns occupy whole groups in this specific recipe.
    for i, idx in enumerate(model.recipe['representation']['resolved']['indices']):
        keep = [remap[j] for j in idx if j in remap]
        if not keep:
            enc.branches[i] = ConstantResponse(enc.branches[i](torch.zeros(1,len(idx),enc.shape[0])))
        elif len(keep)!=len(idx):
            raise ValueError('partial group not supported by this replay')
        setattr(enc,f'indices_{i}',torch.tensor(keep,dtype=torch.long))
    enc.shape = (enc.shape[0],len(cols))
    enc.transport_shape = enc.shape
    return actor


def run(a):
    read = lambda p: json.loads(p.read_text())
    rep = read(a.root/'RESULT.json')
    model,_ = load_bundle(a.bundle,'cpu')
    original = market_for(a.parent,read(a.parent/'SPEC.json'),model.recipe)
    risk = read(a.parent/'RISK.json')
    spec = read(a.spec)['producer_settings']
    cols = list(pd.read_parquet(a.root/'price7/FEATURES61.parquet').columns)
    parent = model.recipe['list']['columns']
    rows = {}
    with torch.inference_mode():
        for mode,width in [('price',61),('context',71)]:
            if not (rep[mode+'_screen'] and rep[mode+'_physical_screen']):
                rows[mode] = 'screen rejected';continue
            actor = reduced_policy(model,cols) if width==61 else effective_actor(model)
            rec = subset_recipe(model.recipe,cols) if width==61 else model.recipe
            market = copy.copy(original)
            market.recipe = rec
            scenarios = ['real',mode+'7',mode+'19']
            if width==61 and rep['context_screen'] and rep['context_physical_screen']:
                scenarios.extend(['context7','context19'])
            for scenario in scenarios:
                path = a.root/scenario
                if scenario=='real':
                    f = pd.read_parquet(a.root/(mode+'7')/'OHLC.parquet')
                    f = original.bars.loc[f.index]
                    z = pd.read_parquet(spec['store'],columns=['timestamp',*rec['list']['columns']],
                        filters=[('timestamp','>=',f.index[0]),('timestamp','<=',f.index[-1])]).set_index('timestamp')
                else:
                    f = pd.read_parquet(path/'OHLC.parquet')
                    z = pd.read_parquet(path/f'FEATURES{width}.parquet')
                m = synthetic_market(market,f,z)
                x = m.inputs
                logits = np.full((len(x.valid),3),np.nan,np.float32)
                ix = np.where(x.valid)[0]
                for j in range(0,len(ix),256):
                    s = ix[j:j+256]
                    logits[s] = actor.surface(torch.from_numpy(x.windows[s])).numpy()
                if width==61 and scenario=='real':
                    # Parity check against parent with excluded standardized channels neutral.
                    s = ix[::61]
                    padded = np.zeros((len(s),x.windows.shape[1],len(parent)),np.float32)
                    padded[:,:,[parent.index(c) for c in cols]] = x.windows[s]
                    ref = effective_actor(model).surface(torch.from_numpy(padded)).numpy()
                    np.testing.assert_allclose(logits[s],ref,rtol=1e-5,atol=1e-5)
                    assert np.array_equal(logits[s].argmax(1),ref.argmax(1))
                env = environment(m,risk,6,['2020-01-01','2023-01-01'],model.effective['gamma'])
                metrics = evaluate(env,logits,prepared=prepared_signals(logits))
                rows[f'{width}_{scenario}'] = {k:metrics[k] for k in
                    ('trades','return_pct','sharpe','cash_error','buy','sell','close_reasons')}
                if abs(metrics['cash_error'])>1e-6:raise ValueError('account mismatch')
    write(a.root/'NATIVE.json',rows)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for k in ('root','bundle','parent','spec'):
        parser.add_argument('--'+k,type=Path,required=True)
    run(parser.parse_args())
