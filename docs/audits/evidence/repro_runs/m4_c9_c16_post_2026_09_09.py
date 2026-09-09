"""POST for order M4 C9-C16: the replay verifier refuses the
audited forgery, and every named guard BITES under mutation.

Phase 1 (corrected, in-process): the PRE's exact forgery — first
sine batch ACQUISITION_ENDPOINT->ACCEPTED with repaired digests
and lifted unit fact; u0_stop.npz replaced by arbitrary bytes
under a repaired digest — now refuses TYPED (replay mismatch;
unreadable state before any verdict).

Phases 2-5 (subprocess, one mutant each): disabling ONE guard
surface lets its dedicated adversary verify again —
  A. transition-replay equality OFF  -> forged ret_loss verifies
  B. checkpoint-vs-replay compare OFF -> lineage-swapped after2
     verifies
  C. telemetry double-booking refusal OFF -> heartbeat both
     digest-bound and unchecked verifies
  D. diagnostic replay+state guards OFF -> the OLD half-example
     (8 vs 16) diagnostic under a lying label verifies
Each adversary is first shown REFUSED by the corrected verifier.

CPU only, fresh tmp roots, zero scientific claims; the committed
v3 preflight evidence is untouched."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CHILD = os.environ.get("M4_C9_POST_CHILD")


def _repair(m4, out, *names):
    rep_p = out / "M4_PREFLIGHT_REPORT.json"
    rep = json.loads(rep_p.read_text())
    for n in names:
        rep["artifacts_sha256"][n] = m4._sha_file(out / n)
    rep["report_sha256"] = m4._self_sha(rep, "report_sha256")
    rep_p.write_text(json.dumps(rep, indent=1))


def _lift(m4, out, mutate):
    rep_p = out / "M4_PREFLIGHT_REPORT.json"
    rep = json.loads(rep_p.read_text())
    mutate(rep)
    rep["report_sha256"] = m4._self_sha(rep, "report_sha256")
    rep_p.write_text(json.dumps(rep, indent=1))


def adversary(m4, out, name):
    import numpy as np
    if name == "forge_outcome":
        p = out / "u0_batches.jsonl"
        lines = p.read_text().splitlines()
        r = json.loads(lines[0])
        assert r["outcome"] == "ACQUISITION_ENDPOINT"
        r["outcome"] = "ACCEPTED"
        r["record_sha256"] = m4._self_sha(r, "record_sha256")
        lines[0] = json.dumps(r, sort_keys=True)
        p.write_text("\n".join(lines) + "\n")
        _repair(m4, out, "u0_batches.jsonl")
        _lift(m4, out, lambda rep: rep["units"][0].__setitem__(
            "accepted_batches_mechanics_only", 1))
    elif name == "arbitrary_bytes":
        (out / "u0_stop.npz").write_bytes(b"NOT-AN-NPZ-BYTES")
        _repair(m4, out, "u0_stop.npz")
    elif name == "retloss":
        p = out / "u1_batches.jsonl"
        lines = p.read_text().splitlines()
        r = json.loads(lines[0])
        r["ret_loss"] = round(r["ret_loss"] + 0.001234, 6)
        r["record_sha256"] = m4._self_sha(r, "record_sha256")
        lines[0] = json.dumps(r, sort_keys=True)
        p.write_text("\n".join(lines) + "\n")
        _repair(m4, out, "u1_batches.jsonl")
    elif name == "after2swap":
        for ext in ("", ".meta.json"):
            shutil.copyfile(str(out / "u1_diag.npz") + ext,
                            str(out / "u1_after2.npz") + ext)
        _repair(m4, out, "u1_after2.npz",
                "u1_after2.npz.meta.json")
    elif name == "hbdouble":
        _lift(m4, out, lambda rep: rep["artifacts_sha256"]
              .__setitem__("M4_HEARTBEAT.json", m4._sha_file(
                  out / "M4_HEARTBEAT.json")))
    elif name == "diag8":
        Xtr, ytr, Xev, yev = m4._gen_unit("sine")
        st = m4._load_state(out / "u0_stop.npz")
        half = int(m4.MINIBATCH * m4.REHEARSAL_FRACTION)
        b = st["batch_index"]
        Xa, ya = m4._batch_assoc("sine", b)
        st["assoc_X"] = np.vstack([st["assoc_X"], Xa]) \
            if len(st["assoc_X"]) else Xa
        st["assoc_y"] = np.concatenate([st["assoc_y"], ya]) \
            if len(st["assoc_y"]) else ya
        for u in range(m4.UPDATES_PER_BATCH):
            rng = np.random.default_rng(
                m4._seed("mb", "sine", b, u))
            rng.integers(0, len(ytr), size=half)
            ia = rng.integers(0, len(st["assoc_y"]),
                              size=m4.MINIBATCH - half)  # OLD 8
            m4._sgd_step(st["params"], st["assoc_X"][ia],
                         st["assoc_y"][ia], m4.LEARNING_RATE)
            st["updates_done"] += 1
        ret_loss = m4._loss(st["params"], Xev, yev)
        if ret_loss > st["retention_margin"]:
            st["retention_streak"] += 1
        else:
            st["retention_streak"] = 0
        o = m4._forward(st["params"], st["assoc_X"])[1]
        ok = np.abs(o - st["assoc_y"]) < m4.ACQ_TOL
        rec = {"batch": b, "ret_loss": round(ret_loss, 6),
               "retention_streak": st["retention_streak"],
               "cumulative_associations":
                   int(len(st["assoc_y"])),
               "cumulative_acquired": int(ok.sum()),
               "cumulative_ok": bool(ok.all()),
               "examples_per_update": m4.MINIBATCH}   # the LIE
        st["batch_index"] += 1
        if st["retention_streak"] >= m4.RETENTION_CONSECUTIVE:
            rec["outcome"] = "RETENTION_ENDPOINT"
        elif not rec["cumulative_ok"]:
            rec["outcome"] = "ACQUISITION_ENDPOINT"
        else:
            st["accepted_batches"] += 1
            rec["outcome"] = "ACCEPTED"
        rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
        (out / "u0_diag.jsonl").write_text(
            json.dumps(rec, sort_keys=True) + "\n")
        for ext in ("", ".meta.json"):
            os.unlink(str(out / "u0_diag.npz") + ext)
        m4._save_state(out / "u0_diag.npz", st)
        _repair(m4, out, "u0_diag.jsonl", "u0_diag.npz",
                "u0_diag.npz.meta.json")
        _lift(m4, out, lambda rep: rep["units"][0].__setitem__(
            "diagnostic_outcome", rec["outcome"]))
    else:
        raise AssertionError(name)


def run_one(m4, name):
    tmp = Path(tempfile.mkdtemp(prefix="m4_post_"))
    try:
        out = tmp / "pf"
        m4.mechanics_preflight(out)
        adversary(m4, out, name)
        try:
            v = m4.verify_preflight(out)
            return {"adversary": name, "result": "VERIFIED",
                    "verified": v["verified"]}
        except SystemExit as exc:
            return {"adversary": name, "result": "REFUSED",
                    "reason": str(exc)[:160]}
    finally:
        shutil.rmtree(tmp)


if CHILD:
    tools_dir = Path(os.environ["M4_C9_TOOLS_DIR"])
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(tools_dir))
    import m4_residual_capacity as m4  # noqa: E402
    assert Path(m4.__file__).parent == tools_dir
    # the mutant copy lives outside the checkout — anchor its
    # design paths back to the REAL sealed designs
    m4.REPO = REPO
    base = REPO / "docs/research/model_capacity"
    m4.DESIGN_PATH_V1 = base / "M4_SEALED_DESIGN_2026_09_08.json"
    m4.DESIGN_PATH = base / "M4_SEALED_DESIGN_V2_2026_09_08.json"
    m4.DESIGN_PATH_V3 = \
        base / "M4_SEALED_DESIGN_V3_2026_09_09.json"
    print(json.dumps(run_one(m4, CHILD)))
    sys.exit(0)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import m4_residual_capacity as m4  # noqa: E402

# ---- Phase 1: the corrected verifier refuses the PRE forgery ----
for adv, needle in (
        ("forge_outcome", "does not replay from its predecessor"),
        ("arbitrary_bytes", "not a loadable NPZ state"),
        ("retloss", "does not replay from its predecessor"),
        ("after2swap", "does not equal the REPLAYED state"),
        ("hbdouble", "both digest-bound and unchecked"),
        ("diag8", "does not replay from the persisted")):
    r = run_one(m4, adv)
    print("corrected:", json.dumps(r))
    assert r["result"] == "REFUSED" and needle in r["reason"], r

# ---- Phases 2-5: one mutant per guard; its adversary verifies --
SRC = (REPO / "tools/m4_residual_capacity.py").read_text()
MUTANTS = {
    "A_replay_off": (
        [("            if replay != claimed:",
          "            if False and replay != claimed:")],
        "retloss"),
    "B_ck_compare_off": (
        [("            if _state_digest(stored) != want:",
          "            if False and "
          "_state_digest(stored) != want:")],
        "after2swap"),
    "C_double_book_off": (
        [("        if name in report[\"telemetry_mutable\"]:",
          "        if False and "
          "name in report[\"telemetry_mutable\"]:")],
        "hbdouble"),
    "D_diag_guards_off": (
        [("        if replay_d != claimed_d:",
          "        if False and replay_d != claimed_d:"),
         ("        if _state_digest(stored_diag) != "
          "_state_digest(st_d):",
          "        if False and _state_digest(stored_diag) != "
          "_state_digest(st_d):")],
        "diag8"),
}
TMP = Path(tempfile.mkdtemp(prefix="m4_post_mut_"))
try:
    for mname, (subs, adv) in MUTANTS.items():
        mut = SRC
        for old, new in subs:
            assert old in mut, (mname, old)
            mut = mut.replace(old, new)
        d = TMP / mname
        d.mkdir()
        (d / "m4_residual_capacity.py").write_text(mut)
        env = {**os.environ, "M4_C9_POST_CHILD": adv,
               "M4_C9_TOOLS_DIR": str(d)}
        rc = subprocess.run([sys.executable, __file__],
                            capture_output=True, text=True,
                            env=env)
        assert rc.returncode == 0, rc.stderr[-300:]
        r = json.loads(rc.stdout.strip().splitlines()[-1])
        print(f"mutant {mname}:", json.dumps(r))
        assert r["result"] == "VERIFIED", \
            f"mutant {mname} must accept its adversary"
finally:
    shutil.rmtree(TMP)

print("\nPOST CONFIRMED: the forgery and every guard-specific "
      "adversary REFUSE under the corrected replay verifier, and "
      "each guard surface BITES — disabling it alone re-admits "
      "its adversary")
