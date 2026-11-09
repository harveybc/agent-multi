"""CPU-only effective module inspection; immutable source, no training or promotion."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_actor_compaction import effective_actor
from app.frozen_recipe_inputs import file_sha
from app.structural_refit import load_bundle
from app.weight_space_diagnostics import tensor_summary, state_delta, center_categorical_head


def inspect_network(net, features):
    observations = {}
    hooks = []
    names = ['encoder.fusion', 'encoder.post_fusion', 'head.0', 'head.1', 'head.2']
    names += [f'encoder.branches.{i}' for i in range(len(net.encoder.branches))]
    names += [f'encoder.core.{i}' for i in range(len(net.encoder.core))]
    modules = dict(net.named_modules())
    for name in names:
        def capture(module, inputs, output, key=name):
            observations[key] = output.detach().clone()
        hooks.append(modules[name].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            outputs = net.surface(features)
    finally:
        for hook in hooks:
            hook.remove()
    w, b = net.head[2].weight.detach(), net.head[2].bias.detach()
    hidden = observations['head.1']
    contributions = hidden.mean(0)[None, :] * w
    torch.testing.assert_close(contributions.sum(-1) + b, outputs.mean(0))
    return dict(
        weights={name: tensor_summary(p) for name, p in net.named_parameters()},
        activations={name: tensor_summary(x) for name, x in observations.items()},
        final_head=dict(weight=w.tolist(), bias=b.tolist(), hidden_mean=hidden.mean(0).tolist(),
            mean_contribution_by_hidden_unit=contributions.tolist(), mean_output=outputs.mean(0).tolist()),
    ), outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'degraded', 'anchor', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('NEW_WEIGHT_REPORT_REQUIRED')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    paths = dict(reference=args.bundle/'model.pt', degraded=args.degraded, anchor=args.anchor)
    hashes = {k: file_sha(p) for k,p in paths.items()}
    degraded_receipt = json.loads((args.degraded.parent/'MANIFEST.json').read_text())
    anchor_receipt = json.loads((args.anchor.parent/'DONE.json').read_text())
    if (hashes['degraded'] != degraded_receipt['files'][args.degraded.name]
            or hashes['anchor'] != anchor_receipt['files'][args.anchor.name]):
        raise ValueError('WEIGHT_SOURCE_RECEIPT')
    states = {k: torch.load(p, map_location='cpu', weights_only=False) for k,p in paths.items()}
    model, manifest = load_bundle(args.bundle, 'cpu')
    probe = np.load(args.bundle/'PROBE.npz')
    features = torch.tensor(probe['inputs'])
    report = dict(source_hashes=hashes, probe_sha256=file_sha(args.bundle/'PROBE.npz'),
        tool_sha256=file_sha(Path(__file__)),
        diagnostics_sha256=file_sha(Path(__file__).resolve().parents[1]/'app/weight_space_diagnostics.py'),
        rows=len(features), feature_count=features.shape[-1], device='cpu',
        scope='fixed TRAIN probe; descriptive weight/activation diagnosis; no fitting',
        comparisons={}, snapshots={}, common_mode_edits={})
    for label, state in states.items():
        model.load(paths[label])
        report['comparisons'][label] = state_delta(states['reference']['loss'], state['loss'])
        actor = effective_actor(model)
        actor_report, a = inspect_network(actor, features)
        with torch.inference_mode():
            native = model.surfaces(features)
            native_q = model.q_surfaces(features)
        torch.testing.assert_close(a, native, atol=1e-5, rtol=1e-5)
        critics = []
        for twin in range(2):
            with model.loss.qvalue_network_params[twin].to_module(model.loss.qvalue_network):
                qnet = copy.deepcopy(model.q_net).eval().requires_grad_(False)
            q_report, q = inspect_network(qnet, features)
            torch.testing.assert_close(q, native_q[twin], atol=1e-5, rtol=1e-5)
            critics.append(q_report)
        report['snapshots'][label] = dict(actor=actor_report, critics=critics,
            actor_copy_max_error=float((a-native).abs().max()),
            greedy_counts=torch.bincount(a.argmax(-1), minlength=3).tolist())
        # A direct, algebraically justified edit on a disposable actor only.
        edited = copy.deepcopy(actor)
        w,b = center_categorical_head(edited.head[2].weight, edited.head[2].bias)
        with torch.no_grad():
            edited.head[2].weight.copy_(w); edited.head[2].bias.copy_(b)
            logits = edited.surface(features)
        report['common_mode_edits'][label] = dict(
            changed_tensors=2, gradient_steps=0, promoted=False,
            max_probability_error=float((logits.softmax(-1)-a.softmax(-1)).abs().max()),
            action_disagreements=int((logits.argmax(-1)!=a.argmax(-1)).sum()),
            interpretation='softmax gauge removal, not a trading improvement')
    if hashes != {k: file_sha(p) for k,p in paths.items()}:
        raise ValueError('WEIGHT_SOURCE_CHANGED')
    report.update(seconds=time.monotonic()-start, source_unchanged=True, optimizer_steps=0,
                  model_promoted=False, representation_groups=model.recipe['representation']['resolved']['names'])
    args.output.mkdir(parents=True)
    target=args.output/'REPORT.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    (args.output/'DONE.json').write_text(json.dumps(dict(report_sha256=file_sha(target),
        source_hashes=hashes, seconds=report['seconds'], model_promoted=False), indent=2)+'\n')
    print(json.dumps(dict(seconds=report['seconds'],source_unchanged=True,
        edits=report['common_mode_edits'],snapshots=list(report['snapshots']))))


if __name__ == '__main__':
    main()
