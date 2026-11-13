from tools.analyze_training_forensics import frozen_selection


def point(step,value):
    return dict(step=step,metrics=dict(stop=dict(mean_regret=value)))


def test_dense_early_observations_do_not_change_original_selection_schedule():
    c=[point(0,0),point(1,0),point(32,0),point(64,2),point(128,1)]
    assert frozen_selection(c)==(128,None)


def test_later_recovery_cannot_retroactively_reset_original_stopper():
    c=[point(0,0),point(64,1)]+[point(64*i,2) for i in range(2,22)]+[point(1408,.1)]
    assert frozen_selection(c)==(64,1344)
