"""Inference-only algebraic projection of a masked, functional SAC actor."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agent_plugins.three_signal_sac import ThreeLogits
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import subset_recipe
from app.structural_refit import verify_files, durable_files


def effective_actor(model):
    # TorchRL owns the learned parameters in the loss, not necessarily in the
    # standalone actor module outside this binding context.
    with model.loss.actor_network_params.to_module(model.actor):
        actor = copy.deepcopy(model.actor_net)
    return actor.eval().requires_grad_(False)


class ConstantResponse(nn.Module):
    def __init__(self, response):
        super().__init__()
        self.register_buffer('response', response.detach().clone())

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.response.shape[-1]:
            raise ValueError('CONSTANT_RESPONSE_SHAPE')
        return self.response.expand(x.shape[0], -1, -1)


class PackedActor(nn.Module):
    """Exact-width reference; intentionally retains the original architecture."""
    def __init__(self, actor, take, original_width):
        super().__init__()
        self.actor = copy.deepcopy(actor).eval().requires_grad_(False)
        self.register_buffer('take', torch.tensor(take, device=next(actor.parameters()).device))
        self.original_width = original_width

    def surface(self, x):
        if x.ndim != 3 or x.shape[-1] != len(self.take):
            raise ValueError('PACKED_INPUT_SHAPE')
        full = x.new_zeros(x.shape[0], x.shape[1], self.original_width)
        full.index_copy_(2, self.take, x)
        return self.actor.surface(full)


def compact_actor(actor, source_recipe, selected):
    projected = subset_recipe(source_recipe, selected)
    names = source_recipe['list']['columns']
    take = [names.index(c) for c in selected]
    remap = {old: new for new, old in enumerate(take)}
    compact = copy.deepcopy(actor).eval().requires_grad_(False)
    encoder = compact.encoder
    if getattr(encoder, 'context', None) is not None:
        raise ValueError('COMPACT_LONG_CONTEXT_UNSUPPORTED')
    if encoder.shape != (source_recipe['tensor']['window']['observations'], len(names)):
        raise ValueError('COMPACT_SOURCE_SHAPE')
    groups = source_recipe['representation']['resolved']
    constants, partial = [], []
    with torch.no_grad():
        for i, (name, indices) in enumerate(zip(groups['names'], groups['indices'])):
            branch = encoder.branches[i]
            if getattr(encoder, f'indices_{i}').tolist() != indices:
                raise ValueError('COMPACT_SOURCE_GROUP_ORDER')
            keep = [j for j, old in enumerate(indices) if old in remap]
            first = branch[0].conv
            if (first.kernel_size != (1,) or first.groups != 1
                    or first.stride != (1,) or first.in_channels != len(indices)):
                raise ValueError('COMPACT_UNSUPPORTED_CONV')
            device, dtype = first.weight.device, first.weight.dtype
            if not keep:
                zeros = torch.zeros(1, len(indices), encoder.shape[0], device=device, dtype=dtype)
                encoder.branches[i] = ConstantResponse(branch(zeros))
                constants.append(dict(index=i, name=name, original_columns=[names[j] for j in indices]))
            elif len(keep) < len(indices):
                first.weight = nn.Parameter(first.weight[:, keep, :].clone(), requires_grad=False)
                first.in_channels = len(keep)
                partial.append(name)
            setattr(encoder, f'indices_{i}', torch.tensor(
                [remap[old] for old in indices if old in remap], device=device, dtype=torch.long))
    encoder.shape = (encoder.shape[0], projected['list']['n'])
    encoder.transport_shape = encoder.shape
    info = dict(parameters_before=sum(p.numel() for p in actor.parameters()),
        parameters_after=sum(p.numel() for p in compact.parameters()),
        constant_groups=constants, partial_groups=partial,
        input_features=len(selected), surviving_learned_groups=len(groups['names'])-len(constants),
        fusion_slots=len(groups['names']), constant_values=sum(
            b.response.numel() for b in encoder.branches if isinstance(b, ConstantResponse)),
        training_supported=False)
    return compact, info


def compare_logits(expected, actual):
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('LOGIT_PARITY_SHAPE')
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('LOGIT_PARITY_FINITE')
    da = prepared_signals(a)[0]; db = prepared_signals(b)[0]
    errors = np.abs(a.astype('float64')-b.astype('float64'))
    within = bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))
    disagreements = int(np.count_nonzero(da != db))
    margins = np.sort(a, axis=1)[:, -1] - np.sort(a, axis=1)[:, -2]
    return dict(rows=len(a), bitwise_exact=bool(np.array_equal(a,b)),
        max_abs_error=float(errors.max()), mean_abs_error=float(errors.mean()),
        atol=1e-5, rtol=1e-5, within_tolerance=within,
        action_disagreements=disagreements, minimum_reference_margin=float(margins.min()),
        qualified=within and disagreements == 0)


def save_compact(path, actor, source_recipe, input_recipe, effective, lineage, probe, info):
    from tools.run_discrete_curve_probe import write
    path = Path(path)
    if path.exists():
        raise ValueError('COMPACT_BUNDLE_EXISTS')
    if input_recipe != subset_recipe(source_recipe, input_recipe['list']['columns']):
        raise ValueError('COMPACT_INPUT_RECIPE')
    path.mkdir(parents=True)
    config = dict(source_recipe=source_recipe, input_recipe=input_recipe, effective=effective,
                  graph=info, lineage=lineage, input_recipe_scope='preprocessing_only; graph preserves source fusion slots')
    write(path/'CONFIG.json', config)
    torch.save(actor.state_dict(), path/'actor.pt')
    device = next(actor.parameters()).device
    with torch.inference_mode():
        logits = actor.surface(torch.tensor(probe, device=device)).cpu().numpy()
    np.savez_compressed(path/'PROBE.npz', inputs=probe, logits=logits)
    manifest = dict(schema='frozen_actor.compact.v1', training_supported=False,
        codec='NOP_BUY_SELL_CLOSE_OPPOSITE_V1',
        files={name:file_sha(path/name) for name in ('CONFIG.json','actor.pt','PROBE.npz')})
    write(path/'MANIFEST.json', manifest)
    restored, _ = load_compact(path, str(device))
    with torch.inference_mode():
        replay = restored.surface(torch.tensor(probe, device=device)).cpu().numpy()
    if not np.array_equal(logits, replay):
        raise ValueError('COMPACT_RELOAD_NOT_EXACT')
    write(path/'RELOAD.json', dict(exact=True, rows=len(probe), max_abs_error=0.))
    durable_files(path)
    return manifest


def load_compact(path, device='cpu'):
    path = Path(path)
    manifest = json.loads((path/'MANIFEST.json').read_text())
    if (manifest.get('schema') != 'frozen_actor.compact.v1'
            or manifest.get('training_supported') is not False
            or manifest.get('codec') != 'NOP_BUY_SELL_CLOSE_OPPOSITE_V1'
            or set(manifest['files']) != {'CONFIG.json','actor.pt','PROBE.npz'}):
        raise ValueError('COMPACT_BUNDLE_SCHEMA')
    verify_files(path, manifest['files'])
    config = json.loads((path/'CONFIG.json').read_text())
    source, recipe = config['source_recipe'], config['input_recipe']
    if recipe != subset_recipe(source, recipe['list']['columns']):
        raise ValueError('COMPACT_INPUT_RECIPE')
    # Reconstruct shapes only; load every tensor strictly. No pickled module,
    # optimizer, replay buffer or fresh training participates in inference.
    with torch.random.fork_rng(devices=[]):
        template = ThreeLogits(source, config['effective']['architecture'])
        actor, info = compact_actor(template, source, recipe['list']['columns'])
    if info != config['graph']:
        raise ValueError('COMPACT_GRAPH')
    actor.load_state_dict(torch.load(path/'actor.pt', map_location='cpu', weights_only=True), strict=True)
    return actor.to(device).eval().requires_grad_(False), manifest
