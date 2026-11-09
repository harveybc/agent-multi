"""Materialize a completed mask selection with exact retained weights, never train."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import project_inputs, support_hash
from app.selected_warm_start import transfer_selected, export_components
from app.structural_refit import build_model, export_bundle, load_bundle
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import authenticated_parent, account_equal
from tools.run_selected_warm_start import check_transfer
from tools.run_structural_sac_refit import market_for, selected_recipe_for_round, assess_full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'selection', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('NEW_WEIGHT_EXPORT_REQUIRED')
    configure(7)
    torch.backends.cudnn.allow_tf32 = False
    spec, out, _, bundle = authenticated_parent(args.parent, 7)
    ss, result, close = [read(args.selection/n) for n in
                        ('SPEC.private.json', 'RESULT.json', 'CLOSE.json')]
    if (file_sha(args.selection/'RESULT.json') != close['result_sha256']
            or file_sha(args.parent/'SPEC.json') != ss['parent_spec_sha256']
            or file_sha(out/'RESULT.json') != ss['parent_result_sha256']
            or file_sha(bundle/'model.pt') != ss['best_bundle']['model_sha256']):
        raise ValueError('WEIGHT_EXPORT_LINEAGE')
    source, _ = load_bundle(bundle, args.device)
    recipe = selected_recipe_for_round(source.recipe, ss, result, close)
    original_market = market_for(args.parent, spec)
    market = copy.copy(original_market)
    market.recipe = recipe
    market.inputs = project_inputs(original_market.inputs, recipe)
    if support_hash(market.inputs) != ss['dataset']['support_hash']:
        raise ValueError('WEIGHT_EXPORT_SUPPORT')
    model = build_model(recipe, source.effective, 7, args.device)
    copied = transfer_selected(source, model)
    parity = check_transfer(source, model, market)
    args.output.mkdir(parents=True)
    risk = read(args.parent/'RISK.json')
    metrics = assess_full(model, market, risk, spec['design'],
                          save=args.output/'native', diagnose=True)
    for period,m in metrics.items():
        account_equal(m, result['final']['metrics'][period])
    ix = np.flatnonzero(market.inputs.valid & (market.inputs.period == 'TRAIN'))
    probe = market.inputs.windows[ix[np.linspace(0, len(ix)-1, 32, dtype=int)]].copy()
    receipt = export_bundle(args.output/'bundle', model, probe,
        dict(epoch=0, identity=ss['identity'], kind='EXACT_SELECTED_WEIGHT_TRANSFER',
             trained=False, inference_only=True, parent_sha256=ss['best_bundle']['model_sha256']))
    export_components(model, args.output/'components')
    restored, _ = load_bundle(args.output/'bundle', args.device)
    reloaded = assess_full(restored, market, risk, spec['design'])
    for period,m in reloaded.items():
        account_equal(m, metrics[period])
    if file_sha(bundle/'model.pt') != ss['best_bundle']['model_sha256']:
        raise ValueError('SOURCE_CHANGED')
    write(args.output/'RESULT.json', dict(selection_identity=ss['identity'],
        selection_result_sha256=close['result_sha256'], bundle=receipt,
        copied=copied, parity=parity, metrics=metrics, native_accounts_exact=True,
        reload_accounts_exact=True, full_support_rows=int(market.inputs.valid.sum()),
        optimizer_steps=0, source_unchanged=True, model_promoted_to_paper=False))
    write(args.output/'DONE.json', dict(result_sha256=file_sha(args.output/'RESULT.json'),
        finished_at=now(), features=len(recipe['list']['columns']),
        groups=len(recipe['representation']['resolved']['names'])))
    print(receipt)


if __name__ == '__main__':
    main()
