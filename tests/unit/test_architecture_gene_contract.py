import json
from pathlib import Path

import pytest

from app.architecture_gene_contract import append_event, decode, encode, read_ledger, value_at


SPACE = json.loads((Path(__file__).resolve().parents[2] /
    'examples/config/phase2_architecture_gene_contract.json').read_text())


def baseline():
    return dict(architecture=dict(branch_width=8, branch_blocks=2, core_width=16,
        core_blocks=2, head_width=8, actor_head_width=8, temporal_attention=dict(
        kind='causal_patch', patch_len=3, embedding=8, heads=2, feedforward=16, gate_init=.1)),
        learning_rate=3e-5, gamma=.99, tau=.005, weight_decay=.01,
        entropy_fraction=.6, alpha=1., critic_regularization=dict(kind='core_dropout', p=.1))


def test_baseline_roundtrip_and_all_boundaries():
    base = baseline()
    raw = encode(SPACE, base)
    decoded = decode(SPACE, raw, base)
    for g in SPACE['genes']:
        assert value_at(decoded, g['path']) == pytest.approx(value_at(base, g['path']))
    for i, g in enumerate(SPACE['genes']):
        values = g.get('values', g.get('bounds'))
        for x, expected in ((0., values[0]), (1., values[-1])):
            trial = list(raw)
            trial[i] = x
            assert value_at(decode(SPACE, trial, base), g['path']) == expected
    assert base == baseline()


def test_every_choice_reachable_and_log_midpoint_geometric():
    base = baseline()
    raw = encode(SPACE, base)
    for i, g in enumerate(SPACE['genes']):
        if g['kind'] == 'choice':
            for j, expected in enumerate(g['values']):
                trial = list(raw)
                trial[i] = (j + .5) / len(g['values'])
                assert value_at(decode(SPACE, trial, base), g['path']) == expected
        elif g['kind'] == 'log10':
            trial = list(raw)
            trial[i] = .5
            assert value_at(decode(SPACE, trial, base), g['path']) == pytest.approx(
                (g['bounds'][0] * g['bounds'][1]) ** .5)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.01, 1.01, True, '0.5'])
def test_invalid_genome_not_silently_clipped(bad):
    raw = encode(SPACE, baseline())
    raw[0] = bad
    with pytest.raises(ValueError, match='DOMAIN'):
        decode(SPACE, raw, baseline())


def event(status='COMPLETE'):
    return dict(campaign_id='test', candidate_id='a', stage='construction', status=status,
                raw_genome=[.5], decoded_config={'width':8}, seed=7, parents=[],
                metrics={}, artifacts={})


def test_ledger_keeps_failures_and_duplicates_and_detects_tampering(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    for status in ('REQUESTED', 'FAILED', 'CACHE_HIT', 'COMPLETE'):
        append_event(path, event(status))
    rows = read_ledger(path)
    assert [r['event']['status'] for r in rows] == ['REQUESTED', 'FAILED', 'CACHE_HIT', 'COMPLETE']
    rows[1]['event']['status'] = 'COMPLETE'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with pytest.raises(ValueError, match='CHAIN'):
        read_ledger(path)


def test_ledger_refuses_partial_or_nonfinite_receipts(tmp_path):
    with pytest.raises(ValueError, match='FIELDS'):
        append_event(tmp_path / 'x', {})
    with pytest.raises(ValueError):
        append_event(tmp_path / 'x', dict(event(), metrics={'score':float('nan')}))
