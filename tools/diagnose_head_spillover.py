"""TRAIN-only CPU dissection of the sealed307 head correction, without fitting."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.head_spillover import decompose_final_layer, spectrum, nearest_rows
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write


def collect(model, windows, rows):
    buffers = {'core': [], 'hidden': []}
    handles = []
    for name, module in [('core', model.actor_net.head[0]), ('hidden', model.actor_net.head[2])]:
        def capture(_module, inputs, name=name):
            buffers[name].append(inputs[0].detach().cpu().numpy().copy())
        handles.append(module.register_forward_pre_hook(capture))
    logits = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), 256):
                logits.append(model.surfaces(torch.tensor(windows[rows[start:start+256]])).numpy())
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(logits), {k: np.concatenate(v) for k, v in buffers.items()}


def run(a):
    started = time.monotonic(); configure(309)
    cfg307 = read(a.candidate/'CONFIG.json'); verify_done(a.candidate, cfg307['identity'])
    result307 = read(a.candidate/'RESULT.json')
    if not result307['train_qualified']:
        raise ValueError('SPILLOVER_REQUIRES_SEALED_TRAIN_CANDIDATE')
    source_receipt = read(a.bundle.parent/'RESULT.json')
    verify_done(a.bundle.parent, source_receipt['identity'])
    spec = read(a.parent/'SPEC.json'); verify_files(a.parent, spec['files'])
    sha = file_sha(a.bundle/'model.pt')
    if sha != cfg307['source_sha256']:
        raise ValueError('SPILLOVER_SOURCE')
    source, _ = load_bundle(a.bundle, 'cpu')
    candidate, _ = load_bundle(a.candidate/'bundle', 'cpu')
    if source.recipe != candidate.recipe or source.effective != candidate.effective:
        raise ValueError('SPILLOVER_RECIPE')
    market = market_for(a.parent, spec, source.recipe)
    if support_hash(market.inputs) != cfg307['support_sha256']:
        raise ValueError('SPILLOVER_SUPPORT')
    floor_cfg = read(a.floor/'CONFIG.json'); verify_done(a.floor, floor_cfg['identity'])
    if floor_cfg['identity'] != cfg307['floor']:
        raise ValueError('SPILLOVER_PANEL_LINEAGE')
    records = read(a.floor/'panel/RECORDS.json')
    anchor = records[cfg307['target'][0]]['row']
    rows = np.flatnonzero(market.inputs.valid & (market.inputs.period == 'TRAIN'))
    ix = np.flatnonzero(rows == anchor)
    if len(ix) != 1:
        raise ValueError('SPILLOVER_TRAIN_ANCHOR')
    anchor_index = int(ix[0])
    cfg = dict(schema='selected71_head_spillover.v1', source_sha256=sha,
               candidate_identity=cfg307['identity'], support_sha256=cfg307['support_sha256'],
               scope='TRAIN2020-2023', validation_inference=False, test_read=False,
               optimizer_steps=0, device='cpu', anchor_row=anchor,
               code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
                   'app/head_spillover.py', 'tools/diagnose_head_spillover.py')})
    cfg['identity'] = digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json') != cfg:
        raise ValueError('SPILLOVER_OUTPUT_COLLISION')
    write(a.output/'CONFIG.json', cfg)
    if (a.output/'DONE.json').exists():
        verify_done(a.output, cfg['identity']); return
    s0, s1 = source.loss.state_dict(), candidate.loss.state_dict()
    if set(s0) != set(s1):
        raise ValueError('SPILLOVER_STATE_KEYS')
    changed = [k for k in s0 if not torch.equal(s0[k], s1[k])]
    if not changed or any(not k.startswith('actor_') or '.head.' not in k for k in changed):
        raise ValueError('SPILLOVER_NONHEAD_CHANGE')
    saved0 = {k:v.clone() for k,v in s0.items()}
    saved1 = {k:v.clone() for k,v in s1.items()}
    print({'stage':'FORWARD_SOURCE', 'rows':len(rows)}, flush=True)
    z0, h0 = collect(source, market.inputs.windows, rows)
    print({'stage':'FORWARD_CANDIDATE'}, flush=True)
    z1, h1 = collect(candidate, market.inputs.windows, rows)
    if not np.array_equal(h0['core'], h1['core']):
        raise ValueError('SPILLOVER_ENCODER_NOT_EXACT')
    def parameter(state, suffix):
        names = [k for k in state if k.startswith('actor_') and k.endswith(suffix)]
        if len(names) != 1:raise ValueError('SPILLOVER_PARAMETER_KEY')
        return state[names[0]].numpy()
    parts = decompose_final_layer(h0['hidden'], h1['hidden'],
        parameter(s0, '.head.2.weight'), parameter(s1, '.head.2.weight'),
        parameter(s0, '.head.2.bias'), parameter(s1, '.head.2.bias'))
    delta = z1.astype(np.float64)-z0
    error = float(np.abs(sum(parts.values())-delta).max())
    tolerance = float(64*np.finfo(np.float32).eps*(1+float(np.abs(z0).max())+float(np.abs(z1).max())))
    if error > tolerance:raise ValueError('SPILLOVER_DECOMPOSITION')
    signals0, signals1 = prepared_signals(z0)[0], prepared_signals(z1)[0]
    flips = np.flatnonzero(signals0 != signals1)
    target = cfg307['target'][1]
    old = int(signals0[anchor_index])
    def detail(i):
        return dict(row=int(rows[i]), timestamp=int(market.inputs.row_ts_ns[rows[i]]),
            source_action=int(signals0[i]), candidate_action=int(signals1[i]),
            source_logits=z0[i].tolist(), candidate_logits=z1[i].tolist(),
            target_margin_delta=float(delta[i,target]-delta[i,old]),
            parts={k:float(v[i,target]-v[i,old]) for k,v in parts.items()})
    neighbours, distances = nearest_rows(h0['core'], anchor_index)
    attr = read(a.candidate/f"round_{len(result307['rounds']):02d}"/'ATTRIBUTION.json')
    gpu_changed_stamps = {e['timestamp'] for e in attr['events']}
    cpu_changed_stamps = set(market.inputs.row_ts_ns[rows[flips]].tolist())
    report = dict(identity=cfg['identity'], rows=len(rows), changed_state_keys=changed,
        encoder_activations_exact=True, spectra={k:spectrum(v) for k,v in h0.items()},
        source_signals={str(k):int((signals0==k).sum()) for k in range(3)},
        changed_signals=len(flips), changed_rows=[detail(int(i)) for i in flips],
        anchor=detail(anchor_index), neighbours=[dict(detail(int(i)),distance=float(d))
            for i,d in zip(neighbours,distances)],
        decomposition_max_error=error, decomposition_tolerance=tolerance,
        part_margin_rms={k:float(np.sqrt(np.mean((v[:,1:]-v[:,:1])**2))) for k,v in parts.items()},
        cpu_gpu_changed_timestamps_equal=cpu_changed_stamps==gpu_changed_stamps,
        cpu_only_changed_timestamps=sorted(cpu_changed_stamps-gpu_changed_stamps),
        gpu_only_changed_timestamps=sorted(gpu_changed_stamps-cpu_changed_stamps),
        parameter_counts={name:sum(p.numel() for k,p in source.loss.named_parameters() if k.startswith(name))
            for name in ('actor_', 'qvalue_')},
        validation_inference=False, test_read=False, optimizer_steps=0, model_promoted=False,
        limitation='Activation rank is descriptive, not a capacity bound; no new account replay.')
    for model, saved in ((source,saved0),(candidate,saved1)):
        if any(not torch.equal(v, model.loss.state_dict()[k]) for k,v in saved.items()):
            raise ValueError('SPILLOVER_STATE_MUTATED')
    if file_sha(a.bundle/'model.pt') != sha:raise ValueError('SPILLOVER_FILE_MUTATED')
    np.savez_compressed(a.output/'TRAIN_ARRAYS.npz',rows=rows,source_logits=z0,candidate_logits=z1,
                        core=h0['core'],hidden_source=h0['hidden'],hidden_candidate=h1['hidden'])
    report['seconds'] = time.monotonic()-started
    write(a.output/'RESULT.json', report); seal(a.output, cfg['identity'])
    print({k:report[k] for k in ('identity','rows','changed_signals','decomposition_max_error','seconds')},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','candidate','floor','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
