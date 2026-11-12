"""Frozen-weight three-column COT replay; no timing certification or deployment."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.cot_identity_probe import COLUMNS, verified_weekly, replace_channels
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment, evaluate, emit
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write
from tools.run_refit_mask_pass import account_equal


def run(a):
    started = time.monotonic(); configure(311)
    reference = read(a.bundle.parent/'RESULT.json')
    verify_done(a.bundle.parent, reference['identity'])
    spec = read(a.parent/'SPEC.json'); verify_files(a.parent, spec['files'])
    doc = read(a.corrected); weekly = verified_weekly(doc)
    if weekly.index.max().year > 2024:
        raise ValueError('COT_TEST_ROWS')
    model, _ = load_bundle(a.bundle, 'cpu')
    market = market_for(a.parent, spec, model.recipe)
    periods = model.recipe['split']['periods']
    if periods != {'TRAIN':['2020-01-01','2024-01-01'], 'VALIDATION':['2024-01-01','2025-01-01']}:
        raise ValueError('COT_PERIODS')
    cfg = dict(schema='cot_identity_sensitivity.v1', scope='NON_GOVERNING_IDENTITY_ONLY',
        source_sha256=file_sha(a.bundle/'model.pt'), source_identity=reference['identity'],
        parent_identity=spec['identity'], corrected_file_sha256=file_sha(a.corrected),
        corrected_rows_sha256=doc['content_sha256'], support_sha256=support_hash(market.inputs),
        columns=list(COLUMNS), periods=periods, legacy_clock_hours=93, max_stale_days=14,
        publication_verified=False, source_archives_independently_checked=False,
        model_promoted=False, optimizer_steps=0, test_read=False,
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in
              ('app/cot_identity_probe.py','tools/diagnose_cot_identity.py')})
    cfg['identity'] = digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json') != cfg:
        raise ValueError('COT_OUTPUT_COLLISION')
    write(a.output/'CONFIG.json', cfg)
    if (a.output/'DONE.json').exists():
        verify_done(a.output, cfg['identity']); return
    # This fork cannot be exported as a recipe: its stored identity is deliberately
    # unchanged to keep the original model/scaler, not to admit a repaired store.
    repaired = copy.copy(market)
    repaired.inputs = replace_channels(market.inputs, weekly)
    if support_hash(repaired.inputs) != support_hash(market.inputs):
        raise ValueError('COT_SUPPORT')
    indices = [model.recipe['list']['columns'].index(c) for c in COLUMNS]
    others = [j for j in range(market.inputs.windows.shape[-1]) if j not in indices]
    if not np.array_equal(market.inputs.windows[:,:,others], repaired.inputs.windows[:,:,others], equal_nan=True):
        raise ValueError('COT_OTHER_CHANNELS')
    saved = {k:v.clone() for k,v in model.loss.state_dict().items()}
    risk = read(a.parent/'RISK.json'); results = {}; logits = {}
    for name, m in [('original', market), ('identity_repaired', repaired)]:
        print({'stage':'INFERENCE_AND_ACCOUNTS','arm':name}, flush=True)
        z, _ = emit(model, m.inputs); prepared = prepared_signals(z)
        logits[name] = z
        results[name] = {p:evaluate(environment(m, risk, spec['design']['horizon'], period,
            model.effective['gamma']), z, save=a.output/name/p, prepared=prepared)
            for p,period in periods.items()}
        if name == 'original':
            for p in periods: account_equal(results[name][p], reference['metrics'][p])
    z0, z1 = logits['original'], logits['identity_repaired']
    signals0, signals1 = prepared_signals(z0)[0], prepared_signals(z1)[0]
    report = dict(identity=cfg['identity'], scope=cfg['scope'], metrics=results,
        weekly_rows=len(weekly), arithmetic_verified=True, warmup_verification=weekly.attrs,
        original_accounts_exact=True,
        non_cot_channels_exact=True, support_unchanged=True, publication_verified=False,
        source_archives_independently_checked=False, model_promoted=False, test_read=False,
        optimizer_steps=0, periods={})
    for p in periods:
        rows = np.flatnonzero(market.inputs.valid & (market.inputs.period == p))
        delta = repaired.inputs.windows[rows][:,:,indices]-market.inputs.windows[rows][:,:,indices]
        matrix = np.zeros((3,3), dtype=int)
        np.add.at(matrix, (signals0[rows], signals1[rows]), 1)
        report['periods'][p] = dict(rows=len(rows), changed_signals=int((signals0[rows]!=signals1[rows]).sum()),
            transition_counts=matrix.tolist(), max_logit_delta=float(np.abs(z1[rows]-z0[rows]).max()),
            input_delta={c:dict(changed_windows=int(np.any(delta[:,:,j]!=0,axis=1).sum()),
                rms_scaled=float(np.sqrt(np.mean(delta[:,:,j].astype(float)**2))),
                max_abs_scaled=float(np.abs(delta[:,:,j]).max())) for j,c in enumerate(COLUMNS)})
    if any(not torch.equal(v, model.loss.state_dict()[k]) for k,v in saved.items()):
        raise ValueError('COT_WEIGHTS_CHANGED')
    if file_sha(a.bundle/'model.pt') != cfg['source_sha256'] or file_sha(a.corrected) != cfg['corrected_file_sha256']:
        raise ValueError('COT_FILES_CHANGED')
    np.savez_compressed(a.output/'LOGITS.npz', original=z0, identity_repaired=z1,
                        row_ts_ns=market.inputs.row_ts_ns, valid=market.inputs.valid)
    report['seconds'] = time.monotonic()-started
    write(a.output/'RESULT.json', report); seal(a.output, cfg['identity'])
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','corrected','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
