"""Fixed 2x2 actor-head intervention, full native accounts, no optimizer or promotion."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment, evaluate
from app.weight_space_diagnostics import transplant_actor_heads
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_refit_mask_pass import authenticated_parent, account_equal, score_metrics
from tools.run_structural_sac_refit import market_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'degraded', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('NEW_WEIGHT_INTERVENTION_ROOT_REQUIRED')
    configure(7)
    torch.backends.cudnn.allow_tf32 = False
    spec, _, archived, bundle = authenticated_parent(args.parent, 7)
    manifest = read(args.degraded.parent/'MANIFEST.json')
    if file_sha(args.degraded) != manifest['files']['model.pt']:
        raise ValueError('DEGRADED_WEIGHT_DIGEST')
    hashes = dict(reference=file_sha(bundle/'model.pt'), degraded=file_sha(args.degraded))
    reference = torch.load(bundle/'model.pt', map_location='cpu', weights_only=False)
    donor = torch.load(args.degraded, map_location='cpu', weights_only=False)
    for key, value in reference['loss'].items():
        if key.startswith('actor_') and '.head.' not in key:
            if not torch.equal(value, donor['loss'][key]):
                raise ValueError('ACTOR_CHANGE_OUTSIDE_HEAD:' + key)
    model, _ = load_bundle(bundle, args.device)
    market = market_for(args.parent, spec)
    if model.recipe != market.recipe:
        raise ValueError('WEIGHT_INTERVENTION_RECIPE')
    risk = read(args.parent/'RISK.json')
    # All combinations fixed before outcomes, including both negative controls.
    cases = {'reference': [], 'degraded_hidden': [0],
             'degraded_output': [2], 'degraded_actor': [0, 2]}
    args.output.mkdir(parents=True)
    write(args.output/'DESIGN.json', dict(cases=cases, source_hashes=hashes,
        source_sha256=file_sha(Path(__file__)), seed=7, test_read=False,
        scope='full paired TRAIN and VALIDATION; diagnostic only; no winner selection'))
    began = time.monotonic()
    results = {}
    for name, layers in cases.items():
        state, keys = transplant_actor_heads(reference['loss'], donor['loss'], layers)
        model.loss.load_state_dict(state, strict=True)
        logits, _ = emit(model, market.inputs)
        signals = prepared_signals(logits)
        metrics = {period:evaluate(environment(market, risk, spec['design']['horizon'],
            interval, spec['effective']['gamma']), logits, prepared=signals,
            diagnose=True, save=args.output/name/period)
            for period,interval in market.recipe['split']['periods'].items()}
        if name == 'reference':
            for period,m in metrics.items():
                account_equal(m, archived['metrics'][period])
        results[name] = dict(score_metrics(metrics), donor_keys=keys,
            signals=np.bincount(logits[market.inputs.valid].argmax(-1), minlength=3).tolist())
        write(args.output/'PROGRESS.json', dict(done=list(results), at=now(),
            seconds=time.monotonic()-began))
    model.load(bundle/'model.pt')
    if hashes != dict(reference=file_sha(bundle/'model.pt'), degraded=file_sha(args.degraded)):
        raise ValueError('SOURCE_WEIGHT_CHANGED')
    write(args.output/'RESULT.json', dict(results=results, source_hashes=hashes,
        original_support_preserved=True, source_unchanged=True, optimizer_steps=0,
        model_promoted=False, seconds=time.monotonic()-began))
    write(args.output/'DONE.json', dict(result_sha256=file_sha(args.output/'RESULT.json'),
        finished_at=now(), cases=len(results)))
    print({k: {p:m['return_pct'] for p,m in v['metrics'].items()} for k,v in results.items()})


if __name__ == '__main__':
    main()
