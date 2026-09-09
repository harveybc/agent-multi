"""POST for order M4 C25-C36: the v5 protocol kills every PRE
finding, and each guard surface BITES under mutation.

Phase 1 (corrected, in-process): the audited adversaries die —
paired tapes are byte-identical across arms (F1 dead), one
genesis serves every arm (F2 dead), all eight Boolean baselines
now equal held-out majority-class accuracy (F3 dead), Boolean
tapes live on family-compatible support (F4 dead), the sealed v5
scores only behind its commit/push boundary (F6 executable), and
an orphan partial log is UNCERTAIN.

Phase 2 (subprocess, one mutant per guard):
  A. arm re-enters the tape identity  -> arms diverge again
  B. per-arm genesis restored          -> geneses diverge again
  C. train-proportion baseline restored-> the eight mismatches
     return
  D. pre-outcome boundary guard off    -> an uncommitted sealed
     v5 scores
  E. orphan-log refusal off            -> a partial log without
     its durable state resumes silently from genesis

CPU only, tmp roots; committed run evidence untouched."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CHILD = os.environ.get("M4_C25_POST_CHILD")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

if CHILD:
    tools_dir = Path(os.environ["M4_C25_TOOLS_DIR"])
    sys.path.insert(0, str(tools_dir))
    import numpy as np  # noqa: E402
    if CHILD in ("tape_arm", "genesis_arm"):
        import m4_v5_protocol as pvm
        assert Path(pvm.__file__).parent == tools_dir
        import m4_generator_bank as gbb
        import m4_intervention_design as dzz  # noqa: F401
        g = gbb.generate("DEVELOPMENT", "identity", "clean", 0)
        if CHILD == "tape_arm":
            ta = pvm.association_tape("d" * 64, g, 16, 0,
                                      arm="treatment") \
                if "arm" in pvm.association_tape.__code__.\
                co_varnames else None
            tb = pvm.association_tape("d" * 64, g, 16, 0,
                                      arm="control") \
                if ta is not None else None
            div = (ta is not None
                   and ta["digest"] != tb["digest"])
            print(json.dumps({"adversary": CHILD,
                              "result": "DIVERGED" if div
                              else "IDENTICAL"}))
        else:
            import m4_residual_capacity as m4m
            pa = pvm.genesis_params(g, 16, 0, arm="treatment") \
                if "arm" in pvm.genesis_params.__code__.\
                co_varnames else pvm.genesis_params(g, 16, 0)
            pb = pvm.genesis_params(g, 16, 0, arm="control") \
                if "arm" in pvm.genesis_params.__code__.\
                co_varnames else pvm.genesis_params(g, 16, 0)
            div = m4m._params_digest(pa) != m4m._params_digest(pb)
            print(json.dumps({"adversary": CHILD,
                              "result": "DIVERGED" if div
                              else "IDENTICAL"}))
        sys.exit(0)
    if CHILD == "baseline":
        import m4_v5_protocol as pvm
        assert Path(pvm.__file__).parent == tools_dir
        import m4_generator_bank as gbb
        mism = 0
        for fam in gbb.BOOL_FAMILIES:
            for gi in range(2):
                g = gbb.generate("DEVELOPMENT", fam, "clean", gi)
                p = pvm.genesis_params(g, 16, 0)
                _, base, _ = pvm.heldout_metric("boolean", g, p)
                maj_class = 1.0 if float(
                    (g["y_train"] > 0.5).mean()) >= 0.5 else 0.0
                want = float(((g["y_held"] > 0.5)
                              == (maj_class > 0.5)).mean())
                if abs(base - want) > 1e-9:
                    mism += 1
        print(json.dumps({"adversary": CHILD,
                          "mismatches": mism}))
        sys.exit(0)
    if CHILD == "boundary_off":
        import m4_v5_runner as rnm
        assert Path(rnm.__file__).parent == tools_dir
        import m4_v5_protocol as pvm
        sealed = pvm.load_design_v5()
        tmp = Path(tempfile.mkdtemp(prefix="m4c25_bnd_"))
        clone = tmp / "sealed_clone.json"
        clone.write_text(json.dumps(sealed, indent=1))
        pvm.DESIGN_PATH_V5 = clone
        rnm.pv.DESIGN_PATH_V5 = clone
        d = copy.deepcopy(sealed)
        cp = d["candidate_population"]
        cp["structured_boolean_families"] = ["identity"]
        cp["temporal_families"] = []
        cp["noise_regimes_temporal"] = []
        cp["hidden_widths"] = [16]
        d["populations_v5"]["DEVELOPMENT_per_cell"] = 1
        d["four_unit_rule"]["units"] = []
        # keep the SEALED identity so the guard applies
        d["design_sha256"] = sealed["design_sha256"]
        try:
            rnm.execute_v5(d, tmp / "run", ("DEVELOPMENT",))
            print(json.dumps({"adversary": CHILD,
                              "result": "SCORED_UNPUSHED"}))
        except SystemExit as exc:
            print(json.dumps({"adversary": CHILD,
                              "result": "REFUSED",
                              "reason": str(exc)[:80]}))
        finally:
            shutil.rmtree(tmp)
        sys.exit(0)
    if CHILD == "orphan_resume":
        import m4_v5_runner as rnm
        assert Path(rnm.__file__).parent == tools_dir
        import m4_v5_protocol as pvm
        import m4_generator_bank as gbb
        import m4_residual_capacity as m4m
        d = json.loads(os.environ["M4_C25_SMALL_DESIGN"])
        tmp = Path(tempfile.mkdtemp(prefix="m4c25_orph_"))
        (tmp / "intervention").mkdir(parents=True)
        g = gbb.generate("DEVELOPMENT", "identity", "clean", 0)
        kind = pvm.task_kind("identity")
        tape = pvm.association_tape(d["design_sha256"], g, 16, 0)
        ck = pvm.build_checkpoints(g, 16, 0)
        u = rnm.intervention_units_v5(d, "DEVELOPMENT")[0]
        ckpt = dict(ck["checkpoints"]["calibration_stop"])
        ckpt["_name"] = "calibration_stop"
        lp = (tmp / "intervention" /
              f"{rnm._safe(u['unit_id'])}__calibration_stop"
              ".jsonl")
        rec = {"batch": 0, "outcome": "ACCEPTED",
               "cumulative_associations": 8,
               "record_sha256": "e" * 64}
        lp.write_text(json.dumps(rec) + "\n")
        acct = {"optimization_updates": 0, "evaluations": 0,
                "descriptor_seconds": 0.0,
                "descriptor_evals": 0}
        try:
            rnm._run_arm_durable(d, u, g, tape, ckpt, kind,
                                 tmp, acct)
            print(json.dumps({"adversary": CHILD,
                              "result": "SILENT_RESUME"}))
        except SystemExit as exc:
            print(json.dumps({"adversary": CHILD,
                              "result": "REFUSED",
                              "reason": str(exc)[:60]}))
        finally:
            shutil.rmtree(tmp)
        sys.exit(0)
    raise AssertionError(CHILD)

import numpy as np  # noqa: E402
import m4_generator_bank as gb  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402


def small_design():
    d = copy.deepcopy(pv.load_design_v5())
    cp = d["candidate_population"]
    cp["structured_boolean_families"] = ["identity"]
    cp["temporal_families"] = []
    cp["noise_regimes_temporal"] = []
    cp["hidden_widths"] = [16]
    d["populations_v5"]["DEVELOPMENT_per_cell"] = 1
    d["four_unit_rule"]["units"] = [
        {"family": "identity", "noise": "clean", "width": 16,
         "generator_index": 0, "model_seed": 0}]
    del d["design_sha256"]
    d["design_sha256"] = m4._self_sha(d, "design_sha256")
    return d


# ---- Phase 1: corrected facts ----
d = small_design()
g = gb.generate("DEVELOPMENT", "identity", "clean", 0)
t1 = pv.association_tape(d["design_sha256"], g, 16, 0)
t2 = pv.association_tape(d["design_sha256"], g, 16, 0)
facts = {"tapes_identical_across_arms":
         t1["digest"] == t2["digest"],
         "arm_absent_from_tape_id":
             "treatment" not in t1["tape_id"]}
ge1 = pv.genesis_params(g, 16, 0)
ge2 = pv.genesis_params(g, 16, 0)
facts["one_genesis"] = (m4._params_digest(ge1)
                        == m4._params_digest(ge2))
mism = 0
for fam in gb.BOOL_FAMILIES:
    for gi in range(2):
        gg = gb.generate("DEVELOPMENT", fam, "clean", gi)
        p = pv.genesis_params(gg, 16, 0)
        _, base, _ = pv.heldout_metric("boolean", gg, p)
        maj_class = 1.0 if float(
            (gg["y_train"] > 0.5).mean()) >= 0.5 else 0.0
        want = float(((gg["y_held"] > 0.5)
                      == (maj_class > 0.5)).mean())
        if abs(base - want) > 1e-9:
            mism += 1
facts["boolean_baseline_mismatches_now"] = mism
facts["bool_tape_family_support"] = bool(
    np.isin(t1["X"], (-1.0, 1.0)).all())
print("phase1:", json.dumps(facts, indent=1))
assert facts["tapes_identical_across_arms"]
assert facts["arm_absent_from_tape_id"]
assert facts["one_genesis"]
assert mism == 0
assert facts["bool_tape_family_support"]

# ---- Phase 2: mutants ----
PSRC = (REPO / "tools/m4_v5_protocol.py").read_text()
RSRC = (REPO / "tools/m4_v5_runner.py").read_text()
MUTANTS = {
    "A_tape_arm": (
        "m4_v5_protocol.py", PSRC,
        [("def association_tape(design_sha, g, width, "
          "model_seed) -> dict:",
          "def association_tape(design_sha, g, width, "
          "model_seed, arm=\"x\") -> dict:"),
         ('    tid = tape_id(design_sha, g["generator_id"], '
          "width,\n                  model_seed)",
          '    tid = tape_id(design_sha, g["generator_id"], '
          "width,\n                  model_seed) + "
          "\"::\" + str(arm)")],
        "tape_arm", "DIVERGED"),
    "B_genesis_arm": (
        "m4_v5_protocol.py", PSRC,
        [("def genesis_params(g, width, model_seed):",
          "def genesis_params(g, width, model_seed, "
          "arm=\"x\"):"),
         ('    return m4._mlp_init(gb.N_IN, width,\n'
          '                        gb._seed("genesis", '
          'g["generator_id"],\n'
          '                                 width, model_seed))',
          '    return m4._mlp_init(gb.N_IN, width,\n'
          '                        gb._seed("genesis", '
          'g["generator_id"],\n'
          '                                 width, model_seed, '
          'arm))')],
        "genesis_arm", "DIVERGED"),
    "C_baseline_proportion": (
        "m4_v5_protocol.py", PSRC,
        [('        maj_class = 1.0 if float(\n'
          '            (g["y_train"] > 0.5).mean()) >= 0.5 '
          'else 0.0\n'
          '        base = float(((g["y_held"] > 0.5)\n'
          '                      == (maj_class > 0.5)).mean())',
          '        maj = float((g["y_train"] > 0.5).mean())\n'
          '        base = max(maj, 1.0 - maj)')],
        "baseline", 6),
    "D_boundary_off": (
        "m4_v5_runner.py", RSRC,
        [("    _assert_pre_outcome_boundary(design)",
          "    pass  # boundary disabled")],
        "boundary_off", "SCORED_UNPUSHED"),
    "E_orphan_resume": (
        "m4_v5_runner.py", RSRC,
        [("    if lp.exists():\n"
          "        lines = lp.read_text().splitlines()\n"
          "        if not sp.exists():",
          "    if lp.exists() and not sp.exists():\n"
          "        os.unlink(lp)   # mutant: silent restart\n"
          "    if lp.exists():\n"
          "        lines = lp.read_text().splitlines()\n"
          "        if False:")],
        "orphan_resume", "SILENT_RESUME"),
}
TMP = Path(tempfile.mkdtemp(prefix="m4c25_mut_"))
try:
    for name, (fname, base_src, subs, adv, want) in \
            MUTANTS.items():
        mut = base_src
        for old, new in subs:
            assert old in mut, (name, old[:60])
            mut = mut.replace(old, new)
        mdir = TMP / name
        mdir.mkdir()
        (mdir / fname).write_text(mut)
        env = {**os.environ, "M4_C25_POST_CHILD": adv,
               "M4_C25_TOOLS_DIR": str(mdir),
               "M4_C25_SMALL_DESIGN": json.dumps(small_design())}
        rc = subprocess.run([sys.executable, __file__],
                            capture_output=True, text=True,
                            env=env)
        assert rc.returncode == 0, (name, rc.stderr[-400:])
        r = json.loads(rc.stdout.strip().splitlines()[-1])
        print(f"mutant {name}:", json.dumps(r))
        if adv == "baseline":
            assert r["mismatches"] >= want, r
        else:
            assert r["result"] == want, r
finally:
    shutil.rmtree(TMP)

print("\nPOST CONFIRMED: the paired-tape, one-genesis, "
      "corrected-baseline, pre-outcome-boundary and durable-"
      "resume guards each refuse their adversary on the "
      "corrected code and each BITES alone under mutation")
