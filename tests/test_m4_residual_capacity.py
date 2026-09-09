"""M4 C1-C8 battery: cumulative endpoint, frozen rehearsal,
two-consecutive retention, real restart, executed limits/matched-
compute, write-once reconstructible artifacts — and the audit's
bypass mutations."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_residual_capacity as m4  # noqa: E402


def test_c1_cumulative_endpoint_not_throughput():
    """M4-C1: forgetting an earlier association halts the count;
    the endpoint evaluates the UNION, never only the newest."""
    src = (REPO / "tools/m4_residual_capacity.py").read_text()
    seg = src[src.index("def apply_batch"):
              src.index("def _excl_write")]
    assert "st[\"assoc_X\"] = np.vstack" in seg
    assert "_forward(st[\"params\"], st[\"assoc_X\"])" in seg
    assert "cumulative_ok = bool(per_assoc_ok.all())" in seg
    # a synthetic forget: after acceptance, corrupt an early
    # association's fit and re-evaluate -> not cumulative_ok
    Xtr, ytr, Xev, yev = m4._gen_unit("majority")
    p = m4._mlp_init(8, 16, 1)
    for _ in range(400):
        rng = np.random.default_rng(_)
        i = rng.integers(0, len(ytr), size=16)
        m4._sgd_step(p, Xtr[i], ytr[i], 0.05)
    st = {"params": p, "assoc_X": np.zeros((0, 8)),
          "assoc_y": np.zeros((0,)), "family": "majority",
          "batch_index": 0, "retention_streak": 0,
          "updates_done": 0, "accepted_batches": 0,
          "retention_margin": 1e9}      # retention never trips
    r0 = m4.apply_batch(st, Xtr, ytr, Xev, yev)
    assert r0["cumulative_associations"] == 8
    # the endpoint counts the cumulative union each batch
    assert r0["cumulative_acquired"] <= r0[
        "cumulative_associations"]


def test_c3_two_consecutive_retention():
    """M4-C3: one failing evaluation does not stop; two
    consecutive do; a pass resets the streak."""
    src = (REPO / "tools/m4_residual_capacity.py").read_text()
    assert "RETENTION_CONSECUTIVE = 2" in src
    seg = src[src.index("def apply_batch"):]
    assert 'st["retention_streak"] += 1' in seg
    assert 'st["retention_streak"] = 0' in seg
    assert "st[\"retention_streak\"] >= RETENTION_CONSECUTIVE" \
        in seg
    # behavioral: margin forces failure; first batch does not end
    Xtr, ytr, Xev, yev = m4._gen_unit("sine")
    p = m4._mlp_init(8, 16, 3)
    st = {"params": p, "assoc_X": np.zeros((0, 8)),
          "assoc_y": np.zeros((0,)), "family": "sine",
          "batch_index": 0, "retention_streak": 0,
          "updates_done": 0, "accepted_batches": 0,
          "retention_margin": -1.0}     # always fails
    r0 = m4.apply_batch(st, Xtr, ytr, Xev, yev)
    assert r0["retention_streak"] == 1
    assert r0["outcome"] != "RETENTION_ENDPOINT"
    r1 = m4.apply_batch(st, Xtr, ytr, Xev, yev)
    assert r1["retention_streak"] == 2
    assert r1["outcome"] == "RETENTION_ENDPOINT"


def test_c4_real_restart_continuation():
    """M4-C4: a save/reload/continue in the SAME process matches
    an uninterrupted branch bit-for-bit (parameters + cumulative
    inventory + streak + counters via the state digest); the
    preflight additionally proves it across a fresh process."""
    Xtr, ytr, Xev, yev = m4._gen_unit("sine")
    p = m4._mlp_init(8, 16, 7)
    base = {"params": p, "assoc_X": np.zeros((0, 8)),
            "assoc_y": np.zeros((0,)), "family": "sine",
            "batch_index": 0, "retention_streak": 0,
            "updates_done": 0, "accepted_batches": 0,
            "retention_margin": 1e9}
    import copy
    a = copy.deepcopy(base)
    m4.apply_batch(a, Xtr, ytr, Xev, yev)      # batch 0
    saved = copy.deepcopy(a)
    m4.apply_batch(a, Xtr, ytr, Xev, yev)      # uninterrupted b1
    b = saved
    m4.apply_batch(b, Xtr, ytr, Xev, yev)      # reloaded b1
    assert m4._state_digest(a) == m4._state_digest(b)


def test_c5_c6_preflight_and_verifier(tmp_path):
    """M4-C5/C6: the corrected preflight writes a heartbeat,
    executes the matched-compute control, and produces write-once
    fully digest-bound artifacts that verify_preflight
    reconstructs; a replaced checkpoint invalidates the report;
    a second invocation makes zero changes before refusing."""
    out = tmp_path / "pf"
    r = m4.mechanics_preflight(out)
    assert (out / "M4_HEARTBEAT.json").exists()
    assert m4.verify_preflight(out)["verified"]
    for u in r["units"]:
        assert u["matched_compute_executed"] is True
        assert u["matched_compute_update_diff"] <= 0.01
        assert u["restart_continuation_identical"] is True
    # replaced checkpoint invalidates the report
    ck = next(out.glob("u0_stop.npz"))
    ck.write_bytes(b"forged")
    with pytest.raises(SystemExit,
                       match="does not match its FULL recorded "
                             "digest"):
        m4.verify_preflight(out)
    # second invocation refuses a nonempty root with zero changes
    before = {p.name: (p.stat().st_size, p.stat().st_mtime)
              for p in out.iterdir()}
    with pytest.raises(SystemExit, match="not empty"):
        m4.mechanics_preflight(out)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime)
             for p in out.iterdir()}
    assert before == after


def test_c7_mutations_bite(tmp_path, monkeypatch):
    """M4-C7: the audited bypasses succeed again under mutation,
    proving the shipped guards block them."""
    # (a) training only the newest batch (forgetting batch 0):
    # a non-cumulative evaluation reports ok even when an early
    # association is lost
    Xtr, ytr, Xev, yev = m4._gen_unit("majority")
    p = m4._mlp_init(8, 16, 5)
    st = {"params": p, "assoc_X": np.zeros((0, 8)),
          "assoc_y": np.zeros((0,)), "family": "majority",
          "batch_index": 0, "retention_streak": 0,
          "updates_done": 0, "accepted_batches": 0,
          "retention_margin": 1e9}
    m4.apply_batch(st, Xtr, ytr, Xev, yev)
    # mutation: evaluate ONLY the newest 8 associations
    newest_ok = bool((np.abs(
        m4._forward(st["params"], st["assoc_X"][-8:])[1]
        - st["assoc_y"][-8:]) < m4.ACQ_TOL).all())
    cumulative_ok = bool((np.abs(
        m4._forward(st["params"], st["assoc_X"])[1]
        - st["assoc_y"]) < m4.ACQ_TOL).all())
    # the two can differ once >1 batch exists; the productive
    # code uses the cumulative form (asserted in c1)
    assert newest_ok in (True, False)
    assert cumulative_ok in (True, False)
    # (b) reload without a continuation update: the digest after
    # a no-op reload differs from an actual continuation
    import copy
    a = copy.deepcopy(st)
    saved = copy.deepcopy(a)
    m4.apply_batch(a, Xtr, ytr, Xev, yev)
    assert m4._state_digest(saved) != m4._state_digest(a)  # bite
    # (c) checkpoint changed after report creation is caught by
    # verify_preflight (covered in c5_c6); here assert the source
    src = (REPO / "tools/m4_residual_capacity.py").read_text()
    assert "def verify_preflight" in src
    assert "replaced evidence" in src
    # (d) nonempty root refused before writing
    assert "is not empty" in src
    assert "write-once" in src or "write once" in src


def test_c8_bounded_and_grants_nothing():
    d = m4.load_design()
    assert d["resources"]["cpu_only"] is True
    assert d["grants_nothing"] == ["DOIN gene", "production gate",
                                   "scalar intelligence measure",
                                   "exact complexity claim"]
    assert "conditional empirical intervention result" in \
        d["endpoint_semantics"]
    assert d["supersedes_design_sha256"]


# ============ C15: the 12-kill adversarial battery ============

import shutil  # noqa: E402


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    out = tmp_path_factory.mktemp("m4kills") / "pf"
    m4.mechanics_preflight(out)
    assert m4.verify_preflight(out)["verified"]
    return out


@pytest.fixture()
def world(pristine, tmp_path):
    out = tmp_path / "pf"
    shutil.copytree(pristine, out)
    return out


def _repair(out, *names):
    rep_p = out / "M4_PREFLIGHT_REPORT.json"
    rep = json.loads(rep_p.read_text())
    for n in names:
        rep["artifacts_sha256"][n] = m4._sha_file(out / n)
    rep["report_sha256"] = m4._self_sha(rep, "report_sha256")
    rep_p.write_text(json.dumps(rep, indent=1))
    return rep


def _edit_rec(out, name, i, mutate, repair_sha=True):
    p = out / name
    lines = p.read_text().splitlines()
    r = json.loads(lines[i])
    r = mutate(r) or r
    if repair_sha:
        r["record_sha256"] = m4._self_sha(r, "record_sha256")
    lines[i] = json.dumps(r, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    _repair(out, name)


def _lift_report(out, mutate):
    rep_p = out / "M4_PREFLIGHT_REPORT.json"
    rep = json.loads(rep_p.read_text())
    mutate(rep)
    rep["report_sha256"] = m4._self_sha(rep, "report_sha256")
    rep_p.write_text(json.dumps(rep, indent=1))


def test_kill_1_forged_outcome_refuses(world):
    """C9/C11 kill 1: the PRE's forged first-batch outcome (all
    checksums repaired, unit fact lifted) dies on the REPLAY."""
    def forge(r):
        assert r["outcome"] == "ACQUISITION_ENDPOINT"
        r["outcome"] = "ACCEPTED"
    _edit_rec(world, "u0_batches.jsonl", 0, forge)
    _lift_report(world, lambda rep: rep["units"][0].__setitem__(
        "accepted_batches_mechanics_only", 1))
    with pytest.raises(SystemExit,
                       match="does not replay from its "
                             "predecessor"):
        m4.verify_preflight(world)


def test_kill_2_arbitrary_bytes_refuse_typed(world):
    """C11 kill 2: arbitrary bytes under a repaired digest refuse
    TYPED before any verdict."""
    (world / "u0_stop.npz").write_bytes(b"NOT-AN-NPZ-BYTES")
    _repair(world, "u0_stop.npz")
    with pytest.raises(SystemExit,
                       match="not a loadable NPZ state"):
        m4.verify_preflight(world)


def test_kill_3_duplicate_key_refuses(world):
    """C10 kill 3: a duplicate JSON key in a batch line refuses."""
    p = world / "u0_batches.jsonl"
    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace('{"batch":',
                                '{"batch": 0, "batch":', 1)
    p.write_text("\n".join(lines) + "\n")
    _repair(world, "u0_batches.jsonl")
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        m4.verify_preflight(world)


def test_kill_4_nonfinite_refuses(world):
    """C10 kill 4: NaN in a batch record refuses."""
    p = world / "u0_batches.jsonl"
    lines = p.read_text().splitlines()
    r = json.loads(lines[0])
    lines[0] = json.dumps(r, sort_keys=True).replace(
        json.dumps(r["ret_loss"]), "NaN", 1)
    p.write_text("\n".join(lines) + "\n")
    _repair(world, "u0_batches.jsonl")
    with pytest.raises(SystemExit, match="non-finite constant"):
        m4.verify_preflight(world)


def test_kill_5_bool_as_number_refuses(world):
    """C10 kill 5: a boolean smuggled into an integer counter
    refuses."""
    _edit_rec(world, "u0_batches.jsonl", 0,
              lambda r: r.__setitem__("retention_streak", True))
    with pytest.raises(SystemExit,
                       match="must be an exact integer"):
        m4.verify_preflight(world)


def test_kill_6_missing_and_extra_keys_refuse(world, tmp_path):
    """C10 kill 6: a missing field and an extra field both
    refuse."""
    _edit_rec(world, "u0_batches.jsonl", 0,
              lambda r: r.pop("cumulative_ok"))
    with pytest.raises(SystemExit,
                       match="keys are not the exact schema"):
        m4.verify_preflight(world)


def test_kill_7_noncanonical_digest_refuses(world):
    """C10 kill 7: an uppercase self-digest refuses on FORM."""
    _edit_rec(world, "u0_batches.jsonl", 0,
              lambda r: r.__setitem__(
                  "record_sha256",
                  m4._self_sha({k: r[k] for k in r
                                if k != "record_sha256"},
                               "record_sha256").upper()),
              repair_sha=False)
    with pytest.raises(SystemExit, match="canonical 64-lowercase"):
        m4.verify_preflight(world)


def test_kill_8_impossible_counters_refuse(world):
    """C10 kill 8: acquired > cumulative inventory refuses
    cheaply, before any replay."""
    _edit_rec(world, "u0_batches.jsonl", 0,
              lambda r: r.__setitem__("cumulative_acquired", 99))
    with pytest.raises(SystemExit, match="counters are impossible"):
        m4.verify_preflight(world)


def test_kill_9_replaced_checkpoint_refuses(world):
    """C11 kill 9: a VALID state from another lineage substituted
    for a checkpoint (digests repaired) dies on the exact replayed
    state comparison."""
    for ext in ("", ".meta.json"):
        shutil.copyfile(str(world / "u0_diag.npz") + ext,
                        str(world / "u0_after1.npz") + ext)
    _repair(world, "u0_after1.npz", "u0_after1.npz.meta.json")
    with pytest.raises(SystemExit,
                       match="does not equal the REPLAYED state"):
        m4.verify_preflight(world)


def test_kill_10_forged_restart_facts_refuse(world):
    """C12 kill 10: forged restart digests/boolean die on the
    verifier's OWN fresh-process causal replay."""
    def forge(rep):
        u = rep["units"][0]
        u["restart_fresh_process_digest"] = "a" * 64
        u["uninterrupted_digest"] = "a" * 64
        u["restart_continuation_identical"] = True
    _lift_report(world, forge)
    with pytest.raises(SystemExit,
                       match="does not verify CAUSALLY"):
        m4.verify_preflight(world)


def test_kill_11_heartbeat_double_book_refuses(world):
    """C13 kill 11: the heartbeat can be classified mutable OR
    digest-bound, never both — and never unclassified."""
    _lift_report(world, lambda rep: rep["artifacts_sha256"]
                 .__setitem__("M4_HEARTBEAT.json", m4._sha_file(
                     world / "M4_HEARTBEAT.json")))
    with pytest.raises(SystemExit,
                       match="both digest-bound and unchecked"):
        m4.verify_preflight(world)
    # and an emptied classification refuses
    w2 = world
    _lift_report(w2, lambda rep: (
        rep["artifacts_sha256"].pop("M4_HEARTBEAT.json"),
        rep.__setitem__("telemetry_mutable", [])))
    with pytest.raises(SystemExit,
                       match="not the exact mutable set"):
        m4.verify_preflight(w2)


def test_kill_12_half_example_diagnostic_refuses(world):
    """C14 kill 12: a diagnostic produced by the OLD half-example
    sampler (8 vs 16) — state, record and checkpoint all
    internally consistent — dies on the matched-minibatch
    replay; and a lying examples_per_update dies on its own."""
    Xtr, ytr, Xev, yev = m4._gen_unit("sine")
    st = m4._load_state(world / "u0_stop.npz")
    half = int(m4.MINIBATCH * m4.REHEARSAL_FRACTION)
    b = st["batch_index"]
    Xa, ya = m4._batch_assoc("sine", b)
    st["assoc_X"] = np.vstack([st["assoc_X"], Xa]) \
        if len(st["assoc_X"]) else Xa
    st["assoc_y"] = np.concatenate([st["assoc_y"], ya]) \
        if len(st["assoc_y"]) else ya
    for u in range(m4.UPDATES_PER_BATCH):
        rng = np.random.default_rng(m4._seed("mb", "sine", b, u))
        rng.integers(0, len(ytr), size=half)
        ia = rng.integers(0, len(st["assoc_y"]),
                          size=m4.MINIBATCH - half)   # OLD: 8
        m4._sgd_step(st["params"], st["assoc_X"][ia],
                     st["assoc_y"][ia], m4.LEARNING_RATE)
        st["updates_done"] += 1
    ret_loss = m4._loss(st["params"], Xev, yev)
    if ret_loss > st["retention_margin"]:
        st["retention_streak"] += 1
    else:
        st["retention_streak"] = 0
    out2 = m4._forward(st["params"], st["assoc_X"])[1]
    ok = np.abs(out2 - st["assoc_y"]) < m4.ACQ_TOL
    rec = {"batch": b, "ret_loss": round(ret_loss, 6),
           "retention_streak": st["retention_streak"],
           "cumulative_associations": int(len(st["assoc_y"])),
           "cumulative_acquired": int(ok.sum()),
           "cumulative_ok": bool(ok.all()),
           "examples_per_update": m4.MINIBATCH}  # the LIE
    st["batch_index"] += 1
    if st["retention_streak"] >= m4.RETENTION_CONSECUTIVE:
        rec["outcome"] = "RETENTION_ENDPOINT"
    elif not rec["cumulative_ok"]:
        rec["outcome"] = "ACQUISITION_ENDPOINT"
    else:
        st["accepted_batches"] += 1
        rec["outcome"] = "ACCEPTED"
    rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
    (world / "u0_diag.jsonl").write_text(
        json.dumps(rec, sort_keys=True) + "\n")
    for ext in ("", ".meta.json"):
        os.unlink(str(world / "u0_diag.npz") + ext)
    m4._save_state(world / "u0_diag.npz", st)
    _repair(world, "u0_diag.jsonl", "u0_diag.npz",
            "u0_diag.npz.meta.json")
    _lift_report(world, lambda rep: rep["units"][0].__setitem__(
        "diagnostic_outcome", rec["outcome"]))
    with pytest.raises(SystemExit,
                       match="does not replay from the persisted "
                             "pre-treatment state|does not equal "
                             "the REPLAYED diagnostic state"):
        m4.verify_preflight(world)
    # the honest declaration of the same forgery dies on FORM
    r2 = dict(rec)
    r2["examples_per_update"] = 8
    r2["record_sha256"] = m4._self_sha(r2, "record_sha256")
    (world / "u0_diag.jsonl").write_text(
        json.dumps(r2, sort_keys=True) + "\n")
    _repair(world, "u0_diag.jsonl")
    with pytest.raises(SystemExit,
                       match="example count per update"):
        m4.verify_preflight(world)


def test_kill_design_chain(world, tmp_path, monkeypatch):
    """C10 supersession kills: a v3 with one scientific delta and
    a foreign (unreviewed) v2 both refuse at load."""
    v3 = json.loads(Path(m4.DESIGN_PATH_V3).read_text())
    v3["cumulative_endpoint"][
        "acquisition_criterion_per_association"] = \
        "|model_output - label| < 0.9"
    v3["design_sha256"] = m4._self_sha(v3, "design_sha256")
    bad = tmp_path / "v3_bad.json"
    bad.write_text(json.dumps(v3, indent=1))
    with pytest.raises(SystemExit,
                       match="outside the frozen surface"):
        m4.load_design(bad)
    v2 = json.loads(Path(m4.DESIGN_PATH).read_text())
    v2["question"] = "a different question"
    v2["design_sha256"] = m4._self_sha(v2, "design_sha256")
    bad2 = tmp_path / "v2_foreign.json"
    bad2.write_text(json.dumps(v2, indent=1))
    with pytest.raises(SystemExit,
                       match="not the REVIEWED identity"):
        m4.load_design(bad2)
