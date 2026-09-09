"""PRE freeze for order M4 C17-C24 at 318cfdb8: the accepted v3
mechanics carry NO intervention foundation — the absences this
order builds are frozen here as executable/source facts BEFORE
any construction.

1. POPULATION: the sealed v3 design DECLARES intent inherited
   from v1 (temporal family names, noise-regime names, the
   hidden-width grid, a textual Bonferroni sentence) but
   MATERIALIZES none of it — no candidate-population census, no
   unit/update counts, no meaningful-product rule, no reduction
   rule, no generator roles; the code implements exactly two
   families at one hardcoded width with one fixed noise draw.
2. GENERATOR CUSTODY: _gen_unit draws every consumer from ONE
   undifferentiated seed namespace — DEVELOPMENT, CALIBRATION and
   CONFIRMATION identities do not exist, no manifest is emitted,
   and no latent-signal/disturbance separation exists for
   temporal tasks.
3. LEARNABILITY GATE: nothing distinguishes structured learning
   from memorization or optimization failure. Executable demo: a
   width-16 MLP trained on RANDOM LABELS reaches high train fit
   while held-out accuracy stays at chance — and no productive
   artifact refuses or types this today (no held-out criterion,
   no baseline, no LEARNABLE/OPTIMIZATION_LIMITED outcome
   exists).
4. ESTIMANDS/PRECISION: no frozen estimand text beyond the
   cumulative endpoint, no censoring analysis for MAX_BATCHES,
   no paired-contrast structure, no precision or multiplicity
   calculation, no confirmatory-population requirement.
5. RUNNER: only the two-unit mechanics preflight exists — no
   config-driven scientific runner, no pre-result ledger, no
   resume, no cost accounting for a population run.

CPU only, tmp roots, zero CALIBRATION/CONFIRMATION material,
v1-v3 and all sealed evidence untouched."""
import json
import sys
import tempfile
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402

# ---- fact 1: declared intent, materialized nothing ----
design = m4.load_design()
dtxt = json.dumps(design)
facts = {
    "design_schema": design["schema"],
    "population_keys_absent": not any(
        k in design for k in
        ("candidate_population", "population_census",
         "unit_census", "reduction_rule", "generator_roles")),
    "temporal_families_declared": "chirp" in dtxt,
    "noise_regimes_declared": "noise_regimes" in dtxt,
    "width_grid_declared": "hidden_units_grid" in dtxt,
    "bonferroni_sentence_declared": "Bonferroni" in dtxt,
}

# ---- fact 2: one undifferentiated generator namespace ----
src = (REPO / "tools/m4_residual_capacity.py").read_text()
gseg = src[src.index("def _gen_unit"):src.index("def _batch_assoc")]
facts["generator_roles_absent_in_code"] = all(
    tok not in src for tok in
    ("DEVELOPMENT", "CALIBRATION", "CONFIRMATION"))
facts["generator_roles_absent_in_design_keys"] = not any(
    "DEVELOPMENT" in str(v) for v in
    (design.get("seeds"), design.get("statistics")))
facts["gen_single_namespace"] = '_seed("gen", family)' in gseg
facts["no_latent_disturbance_split"] = \
    "latent" not in gseg and "disturbance" not in gseg
facts["code_implements_two_families_one_width"] = (
    '("sine", "majority")' in src
    and 'if family == "sine"' in gseg
    and "_mlp_init(8, 16," in src)
facts["no_noise_regime_implementation"] = all(
    tok not in src for tok in
    ("colored", "impulsive", "heteroscedastic"))

# ---- fact 3: memorization is indistinguishable today ----
rng = np.random.default_rng(m4._seed("pre_c17", "randomlabel"))
n, n_in = 64, 8
X = rng.standard_normal((n, n_in))
y = rng.choice([0.0, 1.0], size=n)          # RANDOM labels
Xh = rng.standard_normal((256, n_in))       # held-out
yh = rng.choice([0.0, 1.0], size=256)
p = m4._mlp_init(n_in, 16, m4._seed("pre_c17", "init"))
for u in range(4000):
    r2 = np.random.default_rng(m4._seed("pre_c17", "mb", u))
    i = r2.integers(0, n, size=16)
    m4._sgd_step(p, X[i], y[i], m4.LEARNING_RATE)
tr_acc = float(((m4._forward(p, X)[1] > 0.5) == (y > 0.5)).mean())
ho_acc = float(((m4._forward(p, Xh)[1] > 0.5) == (yh > 0.5)).mean())
facts["random_label_train_acc"] = round(tr_acc, 3)
facts["random_label_heldout_acc"] = round(ho_acc, 3)
facts["memorization_visible_no_gate_exists"] = (
    tr_acc >= 0.9 and abs(ho_acc - 0.5) < 0.12)
facts["no_learnability_gate_in_code"] = all(
    tok not in src for tok in
    ("LEARNABLE_UNDER_FROZEN_BUDGET", "NUMERICALLY_INVALID",
     "held_out", "heldout", "baseline"))

# ---- facts 4/5: estimands, precision, runner absent ----
facts["no_censoring_analysis"] = "censor" not in dtxt
facts["no_precision_or_power_calc"] = all(
    t not in dtxt for t in ("power", "smallest_effect",
                            "minimum_confirmation_generators",
                            "precision_target"))
facts["bonferroni_family_never_frozen"] = \
    "confirmatory_contrast_family" not in dtxt and \
    "family list frozen before" in dtxt
tools = {q.name for q in (REPO / "tools").glob("m4_*.py")}
facts["runner_absent"] = tools == {"m4_residual_capacity.py"}

print(json.dumps(facts, indent=1))
assert all(v is True for k, v in facts.items()
           if k not in ("design_schema",
                        "random_label_train_acc",
                        "random_label_heldout_acc"))

print("\nPRE CONFIRMED at 318cfdb8: no candidate population, one "
      "undifferentiated generator namespace with no role custody, "
      "a random-label task memorized to train acc "
      f"{tr_acc:.3f} with held-out {ho_acc:.3f} and NOTHING that "
      "types or refuses it, no frozen estimands/censoring/"
      "precision/multiplicity, and no scientific runner — the "
      "intervention foundation does not exist yet")
