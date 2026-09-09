"""PRE freeze for order M4 C25-C36 at 47427d74: every blocking
audit finding (F1-F6 + runtime findings) reproduces byte-faithful
on the v4 protocol BEFORE any correction.

Also recorded: the pushed C17-C24 packet at 47427d74 already
carries the exact final-suite facts (3195/3/5/1, isolated-flake
and branch-inherited classifications) with zero placeholders —
the audit reviewed the pre-commit candidate view; the final
commit landed minutes later. The order's precondition is
satisfied by the pushed tip.

CPU only, tmp roots, v4 evidence untouched, no CALIBRATION/
CONFIRMATION outcome scored or persisted."""
import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import m4_generator_bank as gb  # noqa: E402
import m4_intervention_design as dz  # noqa: E402
import m4_intervention_runner as rn  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402

facts = {}

# ---- F1: paired arms do not receive the same associations ----
uid = "intervention::identity::clean::w16::g0::s0"
Xt, yt = m4._batch_assoc(f"{uid}::treatment", 0)
Xc, yc = m4._batch_assoc(f"{uid}::control", 0)
facts["f1_paired_assoc_X_equal"] = bool(np.array_equal(Xt, Xc))
facts["f1_paired_assoc_y_equal"] = bool(np.array_equal(yt, yc))
facts["f1_treatment_assoc_y"] = [int(v) for v in yt]
facts["f1_control_assoc_y"] = [int(v) for v in yc]
assert facts["f1_paired_assoc_X_equal"] is False
assert facts["f1_paired_assoc_y_equal"] is False
src = (REPO / "tools/m4_intervention_runner.py").read_text()
facts["f1_arm_enters_state_family"] = \
    '"family": f"{u[\'unit_id\']}::{arm}"' in src

# ---- F2: control genesis is not the treatment genesis ----
g = gb.generate("DEVELOPMENT", "identity", "clean", 0)
d4 = dz.load_design_v4()
p_treat0 = m4._mlp_init(gb.N_IN, 16,
                        gb._seed("fit_init", g["generator_id"],
                                 16, 0))
p_ctrl = m4._mlp_init(gb.N_IN, 16,
                      gb._seed("ctrl_init", g["generator_id"],
                               16, 0))
facts["f2_genesis_digest_equal"] = (
    m4._params_digest(p_treat0) == m4._params_digest(p_ctrl))
assert facts["f2_genesis_digest_equal"] is False
facts["f2_only_one_checkpoint_materialized"] = (
    "pre_stop" not in src and "post_stop" not in src
    and src.count("_fit(design, g,") >= 2)

# ---- F3: Boolean baseline and head semantics ----
mismatches = {}
for fam in gb.BOOL_FAMILIES:
    for gi in range(2):
        gg = gb.generate("DEVELOPMENT", fam, "clean", gi)
        maj = float((gg["y_train"] > 0.5).mean())
        reported_base = max(maj, 1.0 - maj)
        maj_class = 1.0 if maj >= 0.5 else 0.0
        held_maj_acc = float(
            ((gg["y_held"] > 0.5) == (maj_class > 0.5)).mean())
        if abs(reported_base - held_maj_acc) > 1e-9:
            mismatches[f"{fam} g{gi}"] = {
                "reported": round(reported_base, 6),
                "held_out_majority_acc": round(held_maj_acc, 6)}
facts["f3_boolean_baseline_mismatches"] = mismatches
assert len(mismatches) == 8      # every Boolean dev generator
p_fit = rn._fit(d4, g, 16, 0)
out = m4._forward(p_fit, g["X_held"])[1]
facts["f3_linear_head_out_of_unit_range"] = bool(
    (out < 0).any() or (out > 1).any())
facts["f3_design_declares_sigmoid"] = "sigmoid" in json.dumps(
    d4.get("architectures", {}))

# ---- F4: one association distribution for every family ----
aseg = (REPO / "tools/m4_residual_capacity.py").read_text()
seg = aseg[aseg.index("def _batch_assoc"):
           aseg.index("def apply_batch")]
facts["f4_std_normal_binary_for_all"] = (
    "standard_normal" in seg and "choice([0.0, 1.0]" in seg)
Xb, _ = m4._batch_assoc("identity", 0)
facts["f4_bool_assoc_inputs_not_pm1"] = bool(
    ~np.isin(Xb, (-1.0, 1.0)).all())

# ---- F5: only train/held roles exist in the bank ----
facts["f5_no_stop_slice"] = ("X_stop" not in
                             (REPO / "tools/m4_generator_bank.py"
                              ).read_text())

# ---- F6: v4 chronology (design + outcomes land together) ----
show = subprocess.run(
    ["git", "-C", str(REPO), "show", "--stat", "--format=",
     "f0e25dec"], capture_output=True, text=True).stdout
facts["f6_v4_seal_and_outcomes_same_commit"] = (
    "M4_SEALED_DESIGN_V4_2026_09_09.json" in show
    and "m4_development_run_20260909/RUN_REPORT.json" in show)

# ---- runtime findings: modes, lookalikes, wall contradiction --
TMP = Path(tempfile.mkdtemp(prefix="m4_c25_pre_"))
try:
    d_small = copy.deepcopy(d4)
    cp = d_small["candidate_population"]
    cp["structured_boolean_families"] = ["identity"]
    cp["temporal_families"] = []
    cp["noise_regimes_temporal"] = []
    cp["hidden_widths"] = [16]
    d_small["population_census"][
        "development_generators_per_cell"] = 1
    d_small["four_unit_rule"]["units"] = [
        {"family": "identity", "noise": "clean", "width": 16,
         "generator_index": 0, "model_seed": 0}]
    del d_small["design_sha256"]
    d_small["design_sha256"] = m4._self_sha(d_small,
                                            "design_sha256")
    out = TMP / "run"
    rn.execute(d_small, out)
    jl = next((out / "intervention").glob("*_treatment.jsonl"))
    mode = stat.S_IMODE(jl.stat().st_mode)
    facts["rt_jsonl_mode_group_readable"] = oct(mode)
    facts["rt_jsonl_not_0600"] = mode != 0o600
    # RUN_REPORT lookalike escapes the inventory
    (out / "RUN_REPORT_forged_lookalike.json").write_text("{}")
    facts["rt_lookalike_escapes_inventory"] = (
        "RUN_REPORT_forged_lookalike.json"
        not in rn._inventory(out))
    v = rn.verify_run(d_small, out)
    facts["rt_verify_passes_with_lookalike_present"] = \
        v["verified"]
    # execute returns success WITHOUT calling the verifier
    esrc = src[src.index("def execute"):src.index("def _inventory")]
    facts["rt_execute_never_calls_verifier"] = \
        "verify_run" not in esrc
    # second invocation writes a random extra report
    r2 = rn.execute(d_small, out)
    extra = [q.name for q in out.glob("RUN_REPORT_resume_*")]
    facts["rt_second_invocation_extra_report"] = len(extra) >= 1
finally:
    shutil.rmtree(TMP)
facts["rt_wall_limit_contradiction"] = {
    "enforced_resources_max_wall_seconds":
        d4["resources"]["max_wall_seconds"],
    "declared_reduction_ceiling_hours":
        d4["reduction_rule"]["cpu_resource_ceiling_hours"],
    "contradict": d4["resources"]["max_wall_seconds"]
    != d4["reduction_rule"]["cpu_resource_ceiling_hours"] * 3600}

# ---- inherited contract contradictions ----
tf = json.dumps(d4["task_families"])
facts["ct_family_spelling_conflict"] = (
    "discontinuous" in tf
    and "discontinuity" in json.dumps(
        d4["candidate_population"]["temporal_families"]))
gen_contract = json.dumps(d4.get("seeds", {})) + tf
facts["ct_inherited_numeric_ranges_vs_role_namespaces"] = (
    "0-15" in gen_contract or "16-23" in gen_contract
    or "sha" in gen_contract.lower())

# ---- packet completeness at the pushed tip ----
pk = subprocess.run(
    ["git", "-C", str(REPO), "show",
     "47427d74:docs/handoffs/GENERAL_SATOSHI_TO_MUSASHI_M4_"
     "C17_C24_RETURN_2026_09_09.md"],
    capture_output=True, text=True).stdout
facts["packet_47427d74_zero_placeholders"] = "__" not in pk
facts["packet_47427d74_final_counts_present"] = \
    "3195 passed, 3 failed, 5 skipped" in pk

print(json.dumps(facts, indent=1))
required_true = [k for k in facts
                 if k.startswith(("f1_arm", "f2_only", "f3_lin",
                                  "f4_", "f5_", "f6_",
                                  "rt_jsonl_not", "rt_look",
                                  "rt_verify", "rt_exec",
                                  "rt_second", "ct_",
                                  "packet_"))]
assert all(facts[k] is True for k in required_true), required_true
print("\nPRE CONFIRMED at 47427d74: arms consume DIFFERENT "
      "association tapes and DIFFERENT geneses, only one "
      "checkpoint exists, all eight Boolean baselines mismatch "
      "held-out majority accuracy under a linear head with a "
      "sigmoid-declaring design, one std-normal/binary "
      "association distribution serves every family, the bank "
      "has no STOP role, v4 sealed alongside its own outcomes, "
      "intervention logs are group-readable, report lookalikes "
      "escape inventory, execute never verifies, a second "
      "invocation mints extra reports, the wall limits "
      "contradict, and the inherited generator contract "
      "contradicts the implemented one; the pushed C17-C24 "
      "packet already carries the exact final counts")
