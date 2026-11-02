"""Explicit genome decoding and independent combined-score patience counters."""
import math

GENES = [
    ('branch_width', 'choice', [4, 8, 16]),
    ('core_width', 'choice', [16, 32]),
    ('branch_blocks', 'choice', [1, 2]),
    ('core_blocks', 'choice', [1, 2, 3]),
    ('head_width', 'choice', [4, 8, 16, 32]),
    ('learning_rate', 'log', [-5., -3.]),
    ('gamma', 'linear', [.95, .9995]),
    ('tau', 'log', [-3., math.log10(.02)]),
    ('entropy_fraction', 'linear', [.3, .9]),
    ('weight_decay', 'choice', [0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    ('batch_size', 'choice', [64, 128, 256]),
    ('update_every', 'choice', [2, 4, 8]),
]
BASELINE = dict(branch_width=8, core_width=16, branch_blocks=2, core_blocks=2,
                head_width=8, learning_rate=.0003, gamma=.99, tau=.005,
                entropy_fraction=.6, weight_decay=.01, batch_size=128, update_every=4)
ARCHITECTURE = ('branch_width', 'core_width', 'branch_blocks', 'core_blocks', 'head_width')


def decode(genome):
    if len(genome) != len(GENES) or any(not math.isfinite(g) or not 0 <= g <= 1 for g in genome):
        raise ValueError('GENOME_DOMAIN')
    result = {}
    for g, (name, kind, bounds) in zip(genome, GENES):
        if kind == 'choice':
            result[name] = bounds[min(int(g * len(bounds)), len(bounds)-1)]
        else:
            value = bounds[0] + g * (bounds[1]-bounds[0])
            result[name] = float(f'{(10**value if kind == "log" else value):.12g}')
    return result


def baseline_genome():
    result = []
    for name, kind, bounds in GENES:
        x = BASELINE[name]
        result.append((bounds.index(x)+.5)/len(bounds) if kind == 'choice' else
                      ((math.log10(x) if kind == 'log' else x)-bounds[0])/(bounds[1]-bounds[0]))
    return result


def combined(train, validation):
    if not all(math.isfinite(v) for v in (train, validation)):
        raise ValueError('NONFINITE_PERFORMANCE')
    return .5 * (train + validation)


def make_model(recipe, config, seed, device):
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    return ThreeSignalSAC(recipe, device, seed, gamma=config['gamma'], alpha=1.,
        lr=config['learning_rate'], entropy_fraction=config['entropy_fraction'],
        architecture={k:config[k] for k in ARCHITECTURE}, tau=config['tau'],
        weight_decay=config['weight_decay'])


def assert_consumed(model, config, design):
    """Inspect actual modules/optimizers, not just copied configuration values."""
    for net in (model.actor_net, model.q_net):
        e = net.encoder
        if (e.fusion.conv.out_channels != config['core_width'] or
                len(e.core) != config['core_blocks'] or
                net.head[0].out_features != config['head_width']):
            raise ValueError('GENE_NETWORK_NOT_CONSUMED')
        for b in e.branches:
            if b[0].conv.out_channels != config['branch_width'] or len(b)-1 != config['branch_blocks']:
                raise ValueError('GENE_BRANCH_NOT_CONSUMED')
    for p in model.optimizer.param_groups:
        if p['lr'] != config['learning_rate'] or p['weight_decay'] != config['weight_decay']:
            raise ValueError('GENE_OPTIMIZER_NOT_CONSUMED')
    if not math.isclose(float(model.loss.value_estimator.gamma), config['gamma'], rel_tol=1e-6):
        raise ValueError('GENE_DISCOUNT_NOT_CONSUMED')
    if not math.isclose(1-model.updater.eps, config['tau'], rel_tol=1e-9):
        raise ValueError('GENE_TAU_NOT_CONSUMED')
    if model.entropy_fraction != config['entropy_fraction']:
        raise ValueError('GENE_ENTROPY_NOT_CONSUMED')
    for k in ('batch_size', 'update_every'):
        if design[k] != config[k]:
            raise ValueError('GENE_TRAINING_NOT_CONSUMED')
    return dict(config=config, effective=model.effective,
                parameters=sum(p.numel() for p in model.loss.parameters()))
