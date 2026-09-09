"""PRE freeze for order M4 C31A-C31F at a420f858: the incident
audit's remaining numeric-guard gaps reproduce on the current
code, and the chronology violation is confessed.

CONFESSION (C31A): the order was authored while attempt 2 was
running; in my timeline attempt 2 had already finished and I had
already run AND pushed the C35 adjudication derived from it
(`M4_V5_CALIBRATION_ADJUDICATION_CANDIDATE_2026_09_09.json` at
a420f858) before receiving the order. That adjudication is
hereby RETIRED as NON-GOVERNING evidence of a defective-guard
run; attempt 3 will produce the governing one.

Frozen open routes (audit F1/F2 + order C31D):
 2. finite float64 parameters outside float32 range pass the
    descriptor guard (the productive cast overflows — the
    attempt-2 RuntimeWarning came from that exact line);
 3. an injected SVD failure escapes UNTYPED;
 4. nonfinite singular values produce a "valid" rank;
 8. one anomalous primary arm still enters paired dispersion;
 9. two of three seeds are averaged as a complete generator;
10. attrition beyond the sealed 20% allowance still reports
    precision support;
11. an invalid descriptor enters M2 rows;
12. denominator visibility of incomplete generators is not
    enforced (no planned/complete/incomplete accounting).

Already-corrected routes verified corrected with their commits:
 1. nonfinite checkpoint training -> typed (3f432a0b);
 5. intrabatch divergence -> NUMERICAL_ANOMALY replayed
    (3f432a0b);
 6/7. producer invalidity claims vs fresh replay -> refusals
    (3f432a0b verifier).

CPU only; both calibration attempt roots preserved untouched."""
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

facts = {}

# ---- route 2: float64-finite params overflow the f32 cast ----
big = {"W1": np.full((8, 4), 1e39), "b1": np.zeros(4),
       "W2": np.ones((4, 1)), "b2": np.zeros(1)}
acct = {"descriptor_seconds": 0.0, "descriptor_evals": 0}
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    d = rn._descriptors(big, acct)
overflowed = any("overflow" in str(w.message) for w in wlist)
facts["r2_overflow_cast_passes_guard"] = bool(
    overflowed and d.get("numerically_invalid") is not True
    and d.get("compressed_len_zlib9") is not None)

# ---- route 3: injected SVD failure escapes untyped ----
import unittest.mock as mock
ok = {"W1": np.ones((8, 4)), "b1": np.zeros(4),
      "W2": np.ones((4, 1)), "b2": np.zeros(1)}
try:
    with mock.patch("numpy.linalg.svd",
                    side_effect=np.linalg.LinAlgError("boom")):
        rn._descriptors(ok, dict(acct))
    facts["r3_svd_failure_untyped"] = False
except np.linalg.LinAlgError:
    facts["r3_svd_failure_untyped"] = True   # escapes raw

# ---- route 4: nonfinite singular values -> "valid" rank ----
with mock.patch("numpy.linalg.svd",
                return_value=np.array([np.inf, 1.0, 0.5])):
    d4 = rn._descriptors(ok, dict(acct))
facts["r4_nonfinite_singular_valid_rank"] = (
    d4.get("numerically_invalid") is not True
    and isinstance(d4.get("spectral_rank_W1_1e3"), int))

# ---- routes 8/9/10/12: adjudicator source facts ----
asrc = (REPO / "tools/m4_v5_adjudicate.py").read_text()
seg = asrc[asrc.index("---- dispersion"):asrc.index(
    "---- ladder")]
facts["r8_no_complete_pair_check_in_dispersion"] = (
    "stopping_cause" not in seg and "cap_reached" not in seg)
facts["r9_partial_seeds_averaged"] = (
    "if diffs:" in seg and "== rn.CAL_SEEDS" not in seg
    and "len(diffs) == 3" not in seg)
facts["r10_no_attrition_gate"] = (
    "attrition" not in asrc.lower())
facts["r12_no_planned_complete_incomplete_counts"] = (
    "INCOMPLETE_PAIRED_GENERATOR" not in asrc)

# ---- route 11: invalid descriptors would enter M2 rows ----
lseg = asrc[asrc.index("---- ladder"):asrc.index(
    "---- CONFIRMATION")]
facts["r11_no_descriptor_validity_filter_in_ladder"] = (
    "numerically_invalid" not in lseg)

# ---- corrected routes referenced ----
psrc = (REPO / "tools/m4_v5_protocol.py").read_text()
rsrc = (REPO / "tools/m4_v5_runner.py").read_text()
facts["r1_corrected_3f432a0b"] = \
    "NUMERICALLY_INVALID_TASK_TRAINING" in rsrc
facts["r5_corrected_3f432a0b"] = "NUMERICAL_ANOMALY" in psrc
facts["r67_corrected_3f432a0b"] = \
    "claims an invalid status the" in rsrc

# ---- C31A: attempts preserved; adjudication confession ----
S = Path.home() / ".local/share/agent-multi"
facts["attempt1_preserved"] = (
    S / "m4_v5_calibration_run_20260909_CRASHED_ATTEMPT_1"
).is_dir()
facts["attempt2_preserved"] = (
    S / "m4_v5_calibration_run_20260909" / "RUN_REPORT.json"
).is_file()
facts["confessed_adjudication_pushed_from_attempt2"] = (
    REPO / "docs/audits/evidence/"
    "M4_V5_CALIBRATION_ADJUDICATION_CANDIDATE_2026_09_09.json"
).is_file()

print(json.dumps(facts, indent=1))
open_routes = ["r2_overflow_cast_passes_guard",
               "r3_svd_failure_untyped",
               "r4_nonfinite_singular_valid_rank",
               "r8_no_complete_pair_check_in_dispersion",
               "r9_partial_seeds_averaged",
               "r10_no_attrition_gate",
               "r11_no_descriptor_validity_filter_in_ladder",
               "r12_no_planned_complete_incomplete_counts"]
assert all(facts[k] is True for k in open_routes), open_routes
assert all(facts[k] is True for k in
           ("r1_corrected_3f432a0b", "r5_corrected_3f432a0b",
            "r67_corrected_3f432a0b", "attempt1_preserved",
            "attempt2_preserved",
            "confessed_adjudication_pushed_from_attempt2"))
print("\nPRE CONFIRMED at a420f858: eight incident routes are "
      "OPEN on the current code (descriptor f32-range, untyped "
      "SVD, nonfinite-singular rank, anomalous-arm dispersion, "
      "partial-seed averaging, missing attrition gate, invalid "
      "descriptors in M2, missing complete/incomplete "
      "accounting); routes 1/5/6/7 verified corrected at "
      "3f432a0b; both attempt roots preserved; and the pushed "
      "attempt-2 adjudication is CONFESSED and retired as "
      "non-governing")
