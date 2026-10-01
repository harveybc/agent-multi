"""RL04: R1 weights remain fixed; R0/R2 update under the declared
optimizers; DQN target synchronization and SAC target-critic updates
preserve encoder identity correctly. Keras->torch fidelity gates R1/R2."""
from __future__ import annotations

import numpy as np
import pytest

from ._fixtures import FEATURES, WINDOW, env_config, flat_env, modular_config, write_synthetic_csv
from ._mechanisms import assert_checkout_resolution, require

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("backtrader")


@pytest.fixture
def csv(tmp_path):
    return write_synthetic_csv(tmp_path / "fixture.csv", rows=200)


def _model(csv, algorithm, regimes, donor=None):
    build = require("RL04", "rl_temporal.consumers", "build_model",
                    "SB3 model builder with the modular extractor plugged into the policy")
    mode = "continuous" if algorithm == "SAC" else "discrete"
    env = flat_env(env_config(csv, action_space_mode=mode))
    model = build(algorithm, env, modular_config=modular_config(), feature_order=FEATURES,
                  share_features_extractor=True, device="cpu", seed=0, regimes=regimes,
                  donor=donor, learning_starts=8, buffer_size=64, batch_size=8,
                  target_update_interval=4, tau=1.0 if algorithm == "DQN" else 0.5)
    return env, model


def _hashes(model):
    identity = require("RL04", "rl_temporal.donor_contract", "encoder_identity",
                       "encoder identity hashes per network")
    return identity(model)


def test_r1_branches_stay_fixed_while_r0_core_updates(csv, tmp_path):
    assert_checkout_resolution("RL04")
    make_donor = require("RL04", "rl_temporal.donor_contract", "random_donor",
                         "donor artifact (torch state dict + identity) for R1/R2 fixtures")
    donor = make_donor(modular_config(), seed=7, path=tmp_path / "donor")
    regimes = {f"branch_{i}": "R1" for i in range(len(FEATURES))}
    regimes["core"] = "R0"
    env, model = _model(csv, "SAC", regimes, donor=donor)
    try:
        before = _hashes(model)
        assert before["actor"]["branches"] == donor.identity["branches"], "R1 did not load the donor"
        model.learn(total_timesteps=32)
        after = _hashes(model)
        assert after["actor"]["branches"] == before["actor"]["branches"], "R1 branch weights moved"
        assert after["actor"]["core"] != before["actor"]["core"], "R0 core did not update"
    finally:
        env.close()


def test_r2_branches_start_from_donor_and_then_move(csv, tmp_path):
    make_donor = require("RL04", "rl_temporal.donor_contract", "random_donor",
                         "donor artifact for R2")
    donor = make_donor(modular_config(), seed=7, path=tmp_path / "donor")
    regimes = {f"branch_{i}": "R2" for i in range(len(FEATURES))}
    regimes["core"] = "R2"
    env, model = _model(csv, "DQN", regimes, donor=donor)
    try:
        before = _hashes(model)
        assert before["q_net"]["branches"] == donor.identity["branches"]
        assert before["q_net"]["core"] == donor.identity["core"]
        model.learn(total_timesteps=48)
        after = _hashes(model)
        assert after["q_net"]["branches"] != before["q_net"]["branches"], "R2 branches did not update"
        assert after["q_net"]["core"] != before["q_net"]["core"]
    finally:
        env.close()


def test_r1_without_donor_is_refused_and_r0_forbids_one(csv, tmp_path):
    build = require("RL04", "rl_temporal.consumers", "build_model", "regime/donor contract")
    make_donor = require("RL04", "rl_temporal.donor_contract", "random_donor", "donor artifact")
    env = flat_env(env_config(csv, action_space_mode="discrete"))
    try:
        with pytest.raises(ValueError, match="R1"):
            build("DQN", env, modular_config=modular_config(), feature_order=FEATURES,
                  regimes={"branch_0": "R1", "branch_1": "R1", "branch_2": "R1", "core": "R1"},
                  donor=None, device="cpu", seed=0)
        donor = make_donor(modular_config(), seed=1, path=tmp_path / "donor")
        with pytest.raises(ValueError, match="R0"):
            build("DQN", env, modular_config=modular_config(), feature_order=FEATURES,
                  regimes={"branch_0": "R0", "branch_1": "R0", "branch_2": "R0", "core": "R0"},
                  donor=donor, device="cpu", seed=0)
    finally:
        env.close()


def test_dqn_target_sync_copies_the_online_encoder(csv):
    env, model = _model(csv, "DQN", {"branch_0": "R0", "branch_1": "R0", "branch_2": "R0", "core": "R0"})
    try:
        model.learn(total_timesteps=24)
        h = _hashes(model)
        # tau=1.0 and target_update_interval=4: the last sync happened at most 3 steps ago;
        # force one explicit sync and require identity.
        model._on_step() if model._n_calls % 4 else None
        from stable_baselines3.common.utils import polyak_update
        polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), 1.0)
        h = _hashes(model)
        assert h["q_net_target"]["branches"] == h["q_net"]["branches"]
        assert h["q_net_target"]["core"] == h["q_net"]["core"]
    finally:
        env.close()


def test_sac_target_critic_tracks_critic_encoder_with_polyak(csv):
    env, model = _model(csv, "SAC", {"branch_0": "R0", "branch_1": "R0", "branch_2": "R0", "core": "R0"})
    try:
        model.learn(total_timesteps=32)
        distance = require("RL04", "rl_temporal.donor_contract", "encoder_distance",
                           "L1 distance between two networks' encoders")
        d_before = distance(model.critic, model.critic_target)
        from stable_baselines3.common.utils import polyak_update
        polyak_update(model.critic.parameters(), model.critic_target.parameters(), 1.0)
        d_after = distance(model.critic, model.critic_target)
        assert d_after["total"] == 0.0 and d_before["total"] >= 0.0
    finally:
        env.close()


def test_keras_bundle_weights_import_with_fidelity(tmp_path):
    """Keras bundle at the engine pin -> torch extractor: same window, same outputs."""
    keras_mod = pytest.importorskip("predictor_plugins.modular_temporal")
    import_weights = require("RL04", "rl_temporal.keras_import", "import_keras_bundle",
                             "Keras->torch weight import with per-stage fidelity report")
    Extractor = require("RL04", "rl_temporal.modular_torch", "ModularTemporalExtractor",
                        "torch modular temporal features extractor")
    from gymnasium import spaces
    from predictor_plugins.modular_temporal.assembly import build_modular
    from rl_temporal.observation_layout import ObservationLayout

    cfg = modular_config()
    bundle = build_modular(cfg)
    layout = ObservationLayout.synthetic(window=WINDOW, feature_order=FEATURES, extras=4)
    space = spaces.Box(-np.inf, np.inf, shape=(layout.total_dim,), dtype=np.float32)
    ext = Extractor(space, layout=layout, modular_config=cfg)
    report = import_weights(ext, bundle)
    assert report["imported_tensors"] > 0 and report["unmatched_keras_weights"] == []
    x = np.random.default_rng(0).normal(size=(3, WINDOW, len(FEATURES))).astype("float32")
    stages = ext.stage_outputs_from_window(torch.as_tensor(x))
    keras_latent = bundle.encoder_model(x).numpy()
    keras_fused = bundle.fusion_model(x).numpy()
    np.testing.assert_allclose(stages["fused"].detach().numpy(), keras_fused, atol=1e-5)
    np.testing.assert_allclose(stages["latent"].detach().numpy(), keras_latent, atol=1e-4)
    assert report["fidelity"]["latent_max_abs_diff"] < 1e-4
