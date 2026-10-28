import numpy as np
import pytest

from env_plugins.optimal_exit_diagnostic import OptimalExitDiagnostic
from app.optimal_exit_search import objective_scales, objective_value


def fixture():
    case = dict(close_net=[30., 0., 10., -10.], terminal_net=-20.,
                risk_usd=10., direction=-1)
    return case, np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)


@pytest.mark.parametrize('reward', ['dense_regret', 'terminal_regret'])
def test_reset_suffix_uses_current_optimum_and_preserves_observation(reward):
    case, x = fixture()
    env = OptimalExitDiagnostic([case], [x], reward, start_mode='uniform')
    obs, _ = env.reset(options={'case_index': 0, 'start_index': 1})
    assert np.array_equal(obs['features'], x[1])
    assert set(obs) == {'features', 'action_mask'}
    total = 0.
    for cmd in (0, 4):
        _, r, done, _, info = env.step([cmd, .5, .5, .5])
        total += r
    assert done and total == 0.
    assert info['outcome']['optimal_net_usd'] == 10.
    assert info['outcome']['exit_index'] == 2
    assert info['start_index'] == 1


def test_uniform_start_is_seeded_and_exposes_late_decisions():
    c, x = fixture()
    env = OptimalExitDiagnostic([c], [x], start_mode='uniform')
    def sample():
        env.reset(seed=7)
        return [env.reset()[1]['start_index'] for _ in range(100)]
    assert sample() == sample()
    assert set(sample()) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match='START_INDEX'):
        env.reset(options={'start_index': -1})


def test_default_start_and_executed_command_are_explicit():
    c, x = fixture()
    env = OptimalExitDiagnostic([c], [x])
    _, info = env.reset()
    assert info['start_index'] == 0
    _, _, _, _, info = env.step([4, .5, .5, .5])
    assert info['executed_command'] == 4
    assert info['oracle_close_here'] is True
    assert info['decision_index'] == 0


def test_combined_objective_unit_invariance_and_zero_scale_rejection():
    controls = [dict(sharpe_episode_unannualized=s, mean_return_fraction=r)
                for s, r in [(2., .001), (1., .0005)]]
    scales = objective_scales(controls)
    m = dict(controls[0], mean_regret_risk=.1, max_ideal_regret=0.)
    value = objective_value(m, 'sharpe_return', scales)
    scaled_controls = [dict(c, mean_return_fraction=c['mean_return_fraction'] * 100)
                       for c in controls]
    m['mean_return_fraction'] *= 100
    assert objective_value(m, 'sharpe_return', objective_scales(scaled_controls)) == value
    with pytest.raises(ValueError, match='SCALE'):
        objective_scales([dict(sharpe_episode_unannualized=0., mean_return_fraction=0.)])


def test_gradient_trace_observes_updates_and_exact_commands(tmp_path):
    import torch
    from app.exit_learning_trace import ExitLearningTrace, gradient_norms
    from tools.rl_subset_bridge import build_model
    from tests.unit.test_frozen_recipe_inputs import recipe
    c, x = fixture()
    other = dict(c, direction=1)
    env = OptimalExitDiagnostic([c, other], [x, x], start_mode='uniform')
    budget = dict(n_steps=16, batch_size=8, n_epochs=1)
    model = build_model(env, recipe(), 7, 'cpu', budget)
    trace = ExitLearningTrace(tmp_path, 'synthetic', 32)
    model.learn(32, callback=trace)
    report = trace.report()
    assert sum(report['commands'].values()) == 32
    assert sum(report['visits_by_direction_oracle_close'].values()) == 32
    assert report['optimizer_updates'] == 4
    assert report['gradients']['commands']['max'] > 0
    assert report['gradients']['core']['max'] > 0
    assert report['gradients']['risks']['max'] == 0
    model.policy.commands.weight.grad.fill_(float('nan'))
    with pytest.raises(ValueError, match='NONFINITE_GRADIENT'):
        gradient_norms(model.policy)


def test_paired_comparison_resamples_whole_mirror_quality_clusters():
    from app.exit_comparison import paired_comparison
    from app.optimal_exit_search import panel
    cases=[dict(close_net=[0.,10.,float(p),-5.], terminal_net=-40.,risk_usd=50.,
                direction=d,peak=p,gain=.5,offset=0.,spread=1.) for p in (20,30,40) for d in (1,-1)]
    result=paired_comparison([0.,0.],[[0.,0.]],panel(cases,7),replicates=20)
    assert result['clusters']==3
    for row in result['comparisons'][0]['delta_candidate_minus_control'].values():
        assert row==dict(mean=0.,ci95=[0.,0.])


def test_reference_scope_uses_actual_position_mode():
    from tools.audit_early_close_grid import reference_scope
    identity=dict(design=dict(axes=dict(mode=['position_optimal_long','position_optimal_short'])))
    assert 'FULL_REMAINING_EXECUTABLE_POSITION' in reference_scope(identity)
    identity['design']['axes']['mode'].append('ideal')
    assert reference_scope(identity).startswith('MIXED_')


def test_alias_detector_catches_identical_input_with_conflicting_label():
    from tools.exit_learning_repair import alias_audit
    c,x=fixture()
    d=dict(c,close_net=[0.,30.,10.,-10.])
    assert alias_audit([c,d],[x,x])['contradictory_groups'] > 0


def test_optimization_freezes_choices_replays_and_does_not_claim_full_strategy(tmp_path):
    from tools.exit_learning_repair import optimize_exits
    from tools.recipe_sac_queue import read
    cases=[dict(close_net=[0.,10.,float(p),-5.],terminal_net=-40.,risk_usd=50.,
                direction=d,peak=p,gain=.5,offset=0.,spread=1.) for p in (20,30,40) for d in (1,-1)]
    cfg=dict(seeds=[7],deap=dict(controls=[[0.,0.],[1.,1.],[-1.,-1.]],
        objectives=['negative_exit_regret','sharpe','sharpe_return'],bounds=[-5.,5.],
        population=4,generations=1,patience=1,tolerance=1e-8,bootstrap_replicates=20))
    def status(*args,**kwargs):pass
    def register(*paths):assert all(p.is_file() for p in paths)
    optimize_exits(tmp_path,cfg,cases,cases,register,status)
    path=tmp_path/'deap'/'FINALISTS_BEFORE_CONFIRMATION.json';before=path.read_bytes()
    optimize_exits(tmp_path,cfg,cases,cases,register,status)
    assert path.read_bytes()==before
    report=read(tmp_path/'deap'/'COMPARISON.json')
    assert report['objective_winner'] is None and not report['full_strategy_optimized']
    assert len(report['reports'])==3
