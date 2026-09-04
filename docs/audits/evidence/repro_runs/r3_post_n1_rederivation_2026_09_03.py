"""R3 POST (order @8fce8da0 §4.5): re-derive the N1 interpretation
from the frozen 28 terminal records through the REPAIRED
aggregate_final, prove the primary verdict is unchanged, and persist
the corrected trace as a NEW artifact (the frozen N1 artifacts are
never rewritten)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.target_identifiability_audit import aggregate_final  # noqa: E402

RUN_ROOT = (Path.home() / ".local/share/agent-multi/"
            "target_identifiability_n1_20260903")
FROZEN = (REPO / "docs/audits/evidence/"
          "TARGET_IDENTIFIABILITY_N1_INTERPRETATION_2026_09_03.json")
OUT = (REPO / "docs/audits/evidence/"
       "TARGET_IDENTIFIABILITY_N1_R3_REDERIVATION_2026_09_03.json")

trace = aggregate_final(RUN_ROOT)
frozen = json.loads(FROZEN.read_text())

report = {
    "schema": "agent_multi.n1_r3_rederivation.v1",
    "order": "agent-multi@8fce8da0 §4 item 5",
    "method": ("aggregate_final re-run over the frozen 28 terminal "
               "unit records with the repaired statistics helper "
               "(agent_plugins/paired_inference.py); frozen "
               "interpretation artifact untouched"),
    "primary_verdict_repaired": trace["verdict"],
    "primary_verdict_frozen": frozen["verdict"],
    "verdict_unchanged": trace["verdict"] == frozen["verdict"],
    "holm_pvalues_repaired": trace["holm_pvalues"],
    "holm_pvalues_frozen": frozen["holm_pvalues"],
    "holm_monotone_in_sorted_order": (
        sorted(trace["holm_pvalues"].values())
        == [trace["holm_pvalues"][k] for k in
            sorted(trace["holm_pvalues"],
                   key=trace["holm_pvalues"].get)]),
    "paired_analysis_repaired": trace["paired_analysis"],
    "advancing_arms": [a for a in ("direct_linear",
                                   "direct_temporal")
                       if trace["paired_analysis"][a]["advances"]],
}
assert report["verdict_unchanged"], (
    f"VERDICT CHANGED: {frozen['verdict']} -> {trace['verdict']}")
assert trace["verdict"] == "PREDICTABILITY_NOT_DEMONSTRATED"
assert not report["advancing_arms"]
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in (
    "primary_verdict_repaired", "primary_verdict_frozen",
    "verdict_unchanged", "holm_pvalues_repaired",
    "holm_pvalues_frozen", "advancing_arms")}, indent=2))
print(f"written: {OUT.relative_to(REPO)}")
