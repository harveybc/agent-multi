"""R3 PRE (order @8fce8da0 §4): demonstrate both statistics defects
in the FROZEN N1 runner by executing its literal source lines.

Defect A: zero-variance paired differences -> t=inf ->
          survival_t3 -> linspace(inf, inf+60) -> NaN p-value.
Defect B: Holm without cumulative maximum -> non-monotone adjusted
          p-values (adjusted p DECREASES with increasing raw p).
"""
import math
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[4] / \
    "tools/target_identifiability_audit.py"
lines = SRC.read_text().splitlines()

# --- verify we are executing the real frozen code, not a paraphrase
assert lines[412].strip() == \
    't_stat = m / se if se > 0 else float("inf")', lines[412]
assert lines[445].strip() == \
    'holm[arm] = min(1.0, pvals[arm] * (len(direct_arms) - rank))', \
    lines[445]

# --- Defect A: execute the literal survival_t3 body (lines 428-438)
ns = {"math": math}
exec(textwrap.dedent("\n".join(lines[427:438])), ns)
survival_t3 = ns["survival_t3"]

t_inf = float("inf")  # what line 413 yields for zero-variance diffs
p = survival_t3(t_inf)
print(f"DEFECT A: survival_t3(inf) = {p!r}  (NaN propagated: "
      f"{math.isnan(p)})")
assert math.isnan(p), "expected NaN from the inf->linspace path"

# sanity: the same zero-variance diffs the frozen paired() would hit
diffs = [0.2, 0.2, 0.2, 0.2]
from statistics import mean, stdev
m, sd = mean(diffs), stdev(diffs)
se = sd / math.sqrt(len(diffs))
t_stat = m / se if se > 0 else float("inf")  # literal line 413
print(f"DEFECT A path: diffs={diffs} -> sd={sd} -> t_stat={t_stat}"
      f" -> p={survival_t3(t_stat)!r}")

# --- Defect B: execute the literal Holm loop (lines 443-446)
direct_arms = ["direct_linear", "direct_temporal"]
pvals = {"direct_linear": 0.01, "direct_temporal": 0.011}
holm = {}
for rank, arm in enumerate(sorted(direct_arms,
                                  key=lambda a: pvals[a])):
    holm[arm] = min(1.0, pvals[arm] * (len(direct_arms) - rank))
print(f"DEFECT B: raw={pvals} -> adjusted={holm}")
in_sorted_order = [holm[a] for a in
                   sorted(direct_arms, key=lambda a: pvals[a])]
print(f"DEFECT B: adjusted in ascending-raw order = "
      f"{in_sorted_order}  (monotone: "
      f"{in_sorted_order == sorted(in_sorted_order)})")
assert in_sorted_order != sorted(in_sorted_order), \
    "expected NON-monotone adjusted p-values"
print("PRE CONFIRMED: both defects reproduce from the frozen "
      "source lines")
