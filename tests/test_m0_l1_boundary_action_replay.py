"""Socket-free proofs for the WP2 same-artifact threshold replay
(order 2026-08-11 §6).

No GPU, no real artifacts, no sockets: a tiny fake env/policy pair
exercises the flat-hold replay loop, and the pure primitives prove

* the analytic threshold math on a known raw vector (both thresholds,
  gym-fx ``_coerce_action`` parity including the exact-zero and
  boundary-value semantics);
* near-unique counting under the declared dtype-precision tolerance;
* observation-manifest hash invariance between two different fake
  artifacts replaying the same stream (and sensitivity to a different
  stream), with only forced-hold zero actions ever reaching the env;
* the typed refusal on artifact sha mismatch (including the
  ``run_replay`` early-refusal path that returns BEFORE any model
  construction); and
* every typed outcome branch of the documented rule, including the
  audit-observed D2 epoch-1 action band.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import m0_l1_boundary_action_replay as replay  # noqa: E402


class _Box:
    def __init__(self, shape=(1,), dtype=np.float32):
        self.shape = shape
        self.dtype = np.dtype(dtype)


class FakeEnv:
    """Deterministic observation stream, independent of the policy.

    Terminates after ``n_steps``. Records every action it receives so
    the test can prove the env only ever saw the forced hold action.
    """

    def __init__(self, n_steps: int = 6, obs_dim: int = 4,
                 offset: float = 0.0):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.offset = offset
        self.action_space = _Box()
        self.received: list[np.ndarray] = []
        self.i = 0

    def _obs(self, i: int) -> np.ndarray:
        base = np.arange(self.obs_dim, dtype=np.float32)
        return base * np.float32(0.01) + np.float32(i) + np.float32(
            self.offset)

    def reset(self, *, seed=None):
        self.i = 0
        self.seed = seed
        return self._obs(0), {}

    def step(self, action):
        self.received.append(np.array(action, copy=True))
        self.i += 1
        terminated = self.i >= self.n_steps
        return self._obs(self.i), 0.0, terminated, False, {}


def _policy_varied(obs):
    """Fake trained policy: raw action is a function of the obs."""
    return np.asarray([np.tanh(float(obs[0])) * 0.05], dtype=np.float32)


def _policy_constant(obs):
    """Fake collapsed policy: constant raw action."""
    return np.asarray([0.003], dtype=np.float32)


def _facts(vec: np.ndarray) -> dict:
    vec = np.asarray(vec, dtype=np.float32)
    return replay.split_facts(
        vec, observation_count=int(vec.size),
        observation_manifest_sha256="test-manifest")


# ---------------------------------------------------------------------------
# threshold math
# ---------------------------------------------------------------------------

def test_threshold_mapping_on_known_raw_vector():
    vec = np.asarray([-0.5, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
                     dtype=np.float32)
    easy = replay.map_actions(vec, 0.0)
    # exact zero stays HOLD at threshold 0.0; every nonzero is a side
    assert easy == {"threshold": 0.0, "long": 3, "short": 3, "hold": 1,
                    "non_hold": 6}
    normal = replay.map_actions(vec, 0.1)
    # boundary values map INCLUSIVELY at t>0 (a>=t / a<=-t)
    assert normal == {"threshold": 0.1, "long": 2, "short": 2, "hold": 3,
                      "non_hold": 4}

    facts = _facts(vec)
    assert facts["count_abs_gt_zero"] == 6
    assert facts["fraction_abs_gt_zero"] == pytest.approx(6 / 7)
    assert facts["count_abs_ge_normal_threshold"] == 4
    assert facts["fraction_abs_ge_normal_threshold"] == pytest.approx(
        4 / 7)
    assert facts["mapped_actions"]["threshold_0.0"] == easy
    assert facts["mapped_actions"]["threshold_0.1"] == normal
    # both threshold arms consumed the identical raw vector
    assert facts["action_vector_sha256"] == replay.action_vector_sha256(
        vec)
    declared = facts["raw_actions"]["abs_quantiles_declared"]
    assert declared == list(replay.ABS_QUANTILES)
    assert set(facts["raw_actions"]["abs_quantiles"]) == {
        f"q{q:.2f}" for q in replay.ABS_QUANTILES}
    assert facts["raw_actions"]["interquartile_range"] == pytest.approx(
        facts["raw_actions"]["q75"] - facts["raw_actions"]["q25"])


def test_action_vector_sha_is_order_and_value_sensitive():
    vec = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
    assert replay.action_vector_sha256(vec) == \
        replay.action_vector_sha256(vec.copy())
    assert replay.action_vector_sha256(vec) != \
        replay.action_vector_sha256(vec[::-1])
    assert replay.action_vector_sha256(vec) != \
        replay.action_vector_sha256(vec * 2)


# ---------------------------------------------------------------------------
# near-unique counting under the declared tolerance
# ---------------------------------------------------------------------------

def test_near_unique_counting_with_dtype_tolerance():
    base = np.float32(0.5)
    near = np.nextafter(base, np.float32(1.0))  # one float32 ulp away
    vec = np.asarray([base, near, 0.7], dtype=np.float32)
    tol = replay.near_unique_tolerance(vec)
    eps = float(np.finfo(np.float32).eps)
    # scale floored at 1.0: max|a| = 0.7 -> tolerance = eps * 1.0
    assert tol["value"] == pytest.approx(eps)
    assert tol["formula"] == replay.TOLERANCE_FORMULA
    assert tol["observed_scale_max_abs"] == pytest.approx(0.7, abs=1e-6)
    # the one-ulp pair is indistinguishable at float32 precision
    assert float(near) - float(base) < tol["value"]
    assert replay.near_unique_count(vec, tol["value"]) == 2
    facts = _facts(vec)
    assert facts["raw_actions"]["unique_count_exact"] == 3
    assert facts["raw_actions"]["near_unique_count"] == 2

    # scale above 1.0 grows the tolerance proportionally
    wide = np.asarray([0.0, 2.0], dtype=np.float32)
    assert replay.near_unique_tolerance(wide)["value"] == pytest.approx(
        eps * 2.0)
    assert replay.near_unique_count(np.asarray([], dtype=np.float32),
                                    eps) == 0


# ---------------------------------------------------------------------------
# flat-hold replay: obs-hash invariance across artifacts
# ---------------------------------------------------------------------------

def test_obs_hash_identical_across_fake_artifacts_actions_differ():
    env_a = FakeEnv()
    env_b = FakeEnv()
    run_a = replay.flat_hold_replay(env_a, _policy_varied, seed=101)
    run_b = replay.flat_hold_replay(env_b, _policy_constant, seed=101)

    # ONE canonical observation stream: both artifacts hash identically
    assert run_a["observation_count"] == env_a.n_steps
    assert run_a["observation_count"] == run_b["observation_count"]
    assert run_a["observation_manifest_sha256"] == \
        run_b["observation_manifest_sha256"]
    # ... while the raw action vectors are the artifacts' own
    assert replay.action_vector_sha256(run_a["raw_actions"]) != \
        replay.action_vector_sha256(run_b["raw_actions"])
    assert run_a["raw_actions"].dtype == np.float32
    np.testing.assert_allclose(run_b["raw_actions"], 0.003, rtol=1e-6)

    # the env only ever stepped the forced hold action
    for env in (env_a, env_b):
        assert len(env.received) == env.n_steps
        for action in env.received:
            assert action.shape == (1,)
            assert action.dtype == np.float32
            assert float(action[0]) == 0.0

    # a different observation stream changes the manifest hash
    env_c = FakeEnv(offset=1.0)
    run_c = replay.flat_hold_replay(env_c, _policy_varied, seed=101)
    assert run_c["observation_manifest_sha256"] != \
        run_a["observation_manifest_sha256"]

    # determinism: same env/policy/seed -> byte-identical evidence
    rerun = replay.flat_hold_replay(FakeEnv(), _policy_varied, seed=101)
    assert rerun["observation_manifest_sha256"] == \
        run_a["observation_manifest_sha256"]
    assert replay.action_vector_sha256(rerun["raw_actions"]) == \
        replay.action_vector_sha256(run_a["raw_actions"])


def test_flat_hold_action_requires_box_like_space():
    forced = replay.flat_hold_action(_Box())
    assert forced.shape == (1,)
    assert forced.dtype == np.float32
    assert float(forced[0]) == 0.0
    with pytest.raises(ValueError, match="Box action space"):
        replay.flat_hold_action(object())


# ---------------------------------------------------------------------------
# typed refusal on artifact sha mismatch
# ---------------------------------------------------------------------------

def test_verify_sha256_exact_refusal_strings(tmp_path):
    artifact = tmp_path / "artifact.zip"
    artifact.write_bytes(b"weights")
    observed = hashlib.sha256(b"weights").hexdigest()
    expected = "0" * 64

    refusal = replay.verify_sha256(artifact, expected, "artifact demo")
    assert refusal == (
        f"artifact demo: sha256 mismatch at {artifact} — expected "
        f"{expected}, observed {observed}")
    assert replay.verify_sha256(artifact, observed, "artifact demo") \
        is None
    missing = tmp_path / "absent.zip"
    assert replay.verify_sha256(missing, expected, "artifact anchor") \
        == f"artifact anchor: file missing at {missing}"


def test_run_replay_refuses_typed_before_model_construction(
        tmp_path, monkeypatch):
    wrong = tmp_path / "model.zip"
    wrong.write_bytes(b"not the sealed artifact")
    monkeypatch.setitem(replay.ARTIFACTS, "d2_post_easy", {
        "path": str(wrong),
        "sha256": "f" * 64,
        "role": "test double",
    })
    payload = replay.run_replay(
        "d2_post_easy", device="cpu",
        contract_path=tmp_path / "missing_contract.json")
    assert payload["outcome"] == replay.OUTCOME_INCONCLUSIVE
    assert any("sha256 mismatch" in r for r in payload["refusals"])
    assert any("file missing" in r for r in payload["refusals"])
    # refused BEFORE any env/model construction: no split evidence
    assert payload["splits"] == {}
    assert payload["split_outcomes"] == {}
    assert replay.EXIT_CLASS[payload["outcome"]] == 4


# ---------------------------------------------------------------------------
# typed outcome branches
# ---------------------------------------------------------------------------

def test_outcome_exposes_preexisting_collapse_on_constant_vector():
    facts = _facts(np.full(50, 0.0031, dtype=np.float32))
    assert facts["variation_retained"] is False
    assert facts["count_abs_ge_normal_threshold"] == 0
    assert replay.classify_split(facts) == replay.OUTCOME_EXPOSES


def test_outcome_causes_mapping_collapse_with_variation_retained():
    facts = _facts(np.linspace(0.008, 0.015, 50, dtype=np.float32))
    assert facts["variation_retained"] is True
    assert facts["count_abs_ge_normal_threshold"] == 0
    # all hold at 0.1, non-hold at 0.0 — the defining signature
    assert facts["mapped_actions"]["threshold_0.1"]["hold"] == 50
    assert facts["mapped_actions"]["threshold_0.0"]["non_hold"] == 50
    assert replay.classify_split(facts) == replay.OUTCOME_CAUSES


def test_outcome_causes_on_audit_observed_d2_epoch1_band():
    # AUDIT_SATOSHI_III_RETURN_209_220 §3: D2 easy epoch-1 raw range
    # 0.014293..0.014354 over 13700 steps — the expected live branch.
    vec = np.linspace(0.014293193817138672, 0.014354467391967773,
                      13700, dtype=np.float32)
    facts = _facts(vec)
    assert replay.classify_split(facts) == replay.OUTCOME_CAUSES
    assert facts["mapped_actions"]["threshold_0.1"]["hold"] == 13700
    assert facts["mapped_actions"]["threshold_0.0"]["long"] == 13700


def test_outcome_no_threshold_collapse_branches():
    # some actions cross the deadband
    crossing = np.asarray([0.01, 0.02, 0.2, -0.15], dtype=np.float32)
    assert replay.classify_split(_facts(crossing)) == replay.OUTCOME_NONE
    # constant ABOVE the threshold: collapsed policy, but the threshold
    # is not what silences it
    high_constant = np.full(50, 0.5, dtype=np.float32)
    facts = _facts(high_constant)
    assert facts["variation_retained"] is False
    assert replay.classify_split(facts) == replay.OUTCOME_NONE


def test_overall_outcome_join_rules():
    agree, refusals = replay.classify_overall({
        "train_tail": replay.OUTCOME_CAUSES,
        "inner_validation": replay.OUTCOME_CAUSES,
    })
    assert (agree, refusals) == (replay.OUTCOME_CAUSES, [])

    disagree, refusals = replay.classify_overall({
        "train_tail": replay.OUTCOME_EXPOSES,
        "inner_validation": replay.OUTCOME_NONE,
    })
    assert disagree == replay.OUTCOME_INCONCLUSIVE
    assert refusals == [
        "split outcomes disagree: inner_validation="
        f"{replay.OUTCOME_NONE}, train_tail={replay.OUTCOME_EXPOSES}"]

    empty, refusals = replay.classify_overall({})
    assert empty == replay.OUTCOME_INCONCLUSIVE
    assert refusals == ["no split produced a replay"]

    payload = replay.refusal_result(["exact refusal"])
    assert payload["outcome"] == replay.OUTCOME_INCONCLUSIVE
    assert payload["refusals"] == ["exact refusal"]
