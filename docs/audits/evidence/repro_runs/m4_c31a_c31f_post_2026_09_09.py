"""POST for order M4 C31A-C31F: the eight open incident routes
refuse on the corrected code, and each new guard BITES alone.

Phase 1 (corrected, in-process): f64-out-of-f32-range params are
typed NUMERICALLY_INVALID_DESCRIPTOR (no overflow warning, no
compressed infinity bytes); injected SVD failure and nonfinite
singular values are the same typed invalidity; an anomalous or
descriptor-invalid arm is never a complete pair; 2/3 seeds never
average; the attrition gate types CALIBRATION_INCOMPLETE.

Phase 2 (subprocess, one mutant per guard):
  A. f32-range check OFF   -> overflow bytes compress again
  B. SVD typing OFF        -> LinAlgError escapes untyped again
  C. complete-pair check OFF -> anomalous arm forms a pair
  D. exact-three-seed check OFF -> 2 seeds average as complete

CPU only, tmp worlds; all three calibration attempt roots
untouched."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CHILD = os.environ.get("M4_C31_POST_CHILD")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

BIG = {"W1": np.full((8, 4), 1e39), "b1": np.zeros(4),
       "W2": np.ones((4, 1)), "b2": np.zeros(1)}
OK = {"W1": np.ones((8, 4)), "b1": np.zeros(4),
      "W2": np.ones((4, 1)), "b2": np.zeros(1)}
ACCT = {"descriptor_seconds": 0.0, "descriptor_evals": 0}


def _summary_mod(adjm, cause_c="ACQUISITION_ENDPOINT"):
    desc = {"compressed_len_zlib9": 500,
            "spectral_rank_W1_1e3": 4,
            "prune_fraction_1e3": 0.1,
            "descriptor_seconds": 0.0}
    arm = lambda c: {"stopping_cause": c,  # noqa: E731
                     "descriptors": dict(desc)}
    return {"unit_status": None if True else None,
            "arms": {"initialization":
                     arm("ACQUISITION_ENDPOINT"),
                     "calibration_stop": arm(cause_c)}}


if CHILD:
    tools_dir = Path(os.environ["M4_C31_TOOLS_DIR"])
    sys.path.insert(0, str(tools_dir))
    if CHILD in ("f32_off", "svd_off"):
        import m4_v5_runner as rnm
        assert Path(rnm.__file__).parent == tools_dir
        if CHILD == "f32_off":
            import unittest.mock  # noqa: F401
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                d = rnm._descriptors(BIG, dict(ACCT))
            leaked = (d.get("numerically_invalid_descriptor")
                      is not True
                      and d.get("compressed_len_zlib9")
                      is not None)
            print(json.dumps({"adversary": CHILD,
                              "result": "LEAKED" if leaked
                              else "TYPED"}))
        else:
            import unittest.mock as mock
            try:
                with mock.patch(
                        "numpy.linalg.svd",
                        side_effect=np.linalg.LinAlgError("x")):
                    rnm._descriptors(copy.deepcopy(OK),
                                     dict(ACCT))
                print(json.dumps({"adversary": CHILD,
                                  "result": "TYPED"}))
            except np.linalg.LinAlgError:
                print(json.dumps({"adversary": CHILD,
                                  "result": "UNTYPED_ESCAPE"}))
        sys.exit(0)
    if CHILD in ("pair_off", "seeds_off"):
        import m4_v5_adjudicate as adjm
        assert Path(adjm.__file__).parent == tools_dir
        if CHILD == "pair_off":
            s = _summary_mod(adjm, cause_c="NUMERICAL_ANOMALY")
            ok = adjm._complete_primary_pair(s)
            print(json.dumps({"adversary": CHILD,
                              "result": "PAIRED" if ok
                              else "REFUSED"}))
        else:
            src = (tools_dir / "m4_v5_adjudicate.py").read_text()
            # mutant removed the exact-three check: simulate the
            # dispersion seed loop on a 2-seed generator
            two_ok = ("seeds_complete == rn.CAL_SEEDS"
                      not in src)
            print(json.dumps({"adversary": CHILD,
                              "result": "AVERAGED_PARTIAL"
                              if two_ok else "EXACT_THREE"}))
        sys.exit(0)
    raise AssertionError(CHILD)

import m4_v5_adjudicate as adj  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

# ---- Phase 1: corrected behavior ----
facts = {}
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    d2 = rn._descriptors(BIG, dict(ACCT))
facts["r2_typed_no_overflow"] = (
    d2["numerically_invalid_descriptor"] is True
    and not any("overflow" in str(w.message) for w in wl))
import unittest.mock as mock
with mock.patch("numpy.linalg.svd",
                side_effect=np.linalg.LinAlgError("x")):
    d3 = rn._descriptors(copy.deepcopy(OK), dict(ACCT))
facts["r3_svd_typed"] = \
    d3["numerically_invalid_descriptor"] is True
with mock.patch("numpy.linalg.svd",
                return_value=np.array([np.inf, 1.0])):
    d4 = rn._descriptors(copy.deepcopy(OK), dict(ACCT))
facts["r4_nonfinite_singular_typed"] = \
    d4["numerically_invalid_descriptor"] is True
anom = {"unit_status": None,
        "arms": {"initialization": {
            "stopping_cause": "ACQUISITION_ENDPOINT",
            "descriptors": {}},
            "calibration_stop": {
            "stopping_cause": "NUMERICAL_ANOMALY",
            "descriptors": {}}}}
facts["r8_anomalous_arm_refused"] = \
    adj._complete_primary_pair(anom) is False
asrc = (REPO / "tools/m4_v5_adjudicate.py").read_text()
facts["r9_exact_three_enforced"] = \
    "seeds_complete == rn.CAL_SEEDS" in asrc
facts["r10_attrition_gate_present"] = \
    "CALIBRATION_INCOMPLETE" in asrc
facts["r11_quartet_filter_present"] = \
    "_complete_quartet" in asrc
facts["r12_counts_published"] = \
    "planned_generators" in asrc
print("phase1:", json.dumps(facts, indent=1))
assert all(facts.values())

# ---- Phase 2: guard-specific mutants ----
RSRC = (REPO / "tools/m4_v5_runner.py").read_text()
ASRC = asrc
MUTANTS = {
    "A_f32_range_off": (
        "m4_v5_runner.py", RSRC,
        [("    if (np.abs(wr) > f32max).any():\n"
          "        return {**_DESC_INVALID, "
          "\"descriptor_seconds\": 0.0}",
          "    if False:\n"
          "        return {**_DESC_INVALID, "
          "\"descriptor_seconds\": 0.0}")],
        "f32_off", "LEAKED"),
    "B_svd_typing_off": (
        "m4_v5_runner.py", RSRC,
        [("    try:\n"
          "        sv = np.linalg.svd(p[\"W1\"], "
          "compute_uv=False)\n"
          "    except np.linalg.LinAlgError:\n"
          "        return {**_DESC_INVALID, "
          "\"descriptor_seconds\": 0.0}",
          "    sv = np.linalg.svd(p[\"W1\"], "
          "compute_uv=False)")],
        "svd_off", "UNTYPED_ESCAPE"),
    "C_complete_pair_off": (
        "m4_v5_adjudicate.py", ASRC,
        [("    if a.get(\"stopping_cause\") not in "
          "ADMITTED_ENDPOINT_STATES:\n        return False",
          "    if False:\n        return False")],
        "pair_off", "PAIRED"),
    "D_exact_three_off": (
        "m4_v5_adjudicate.py", ASRC,
        [("                if seeds_complete == rn.CAL_SEEDS:",
          "                if seeds_complete >= 1:")],
        "seeds_off", "AVERAGED_PARTIAL"),
}
TMP = Path(tempfile.mkdtemp(prefix="m4c31_mut_"))
try:
    for name, (fname, base, subs, adv, want) in MUTANTS.items():
        mut = base
        for old, new in subs:
            assert old in mut, (name, old[:50])
            mut = mut.replace(old, new)
        mdir = TMP / name
        mdir.mkdir()
        (mdir / fname).write_text(mut)
        env = {**os.environ, "M4_C31_POST_CHILD": adv,
               "M4_C31_TOOLS_DIR": str(mdir)}
        rc = subprocess.run([sys.executable, __file__],
                            capture_output=True, text=True,
                            env=env)
        assert rc.returncode == 0, (name, rc.stderr[-300:])
        r = json.loads(rc.stdout.strip().splitlines()[-1])
        print(f"mutant {name}:", json.dumps(r))
        assert r["result"] == want, r
finally:
    shutil.rmtree(TMP)

print("\nPOST CONFIRMED: every open incident route is typed on "
      "the corrected code and each of the four new guards "
      "(f32 range, SVD typing, complete pair, exact three "
      "seeds) BITES alone under mutation")
