"""Reproducer for AUD-F1-20260821-PLR-01..04 corrections.

Run from the repo root. Exit 0 with {"reproduced": false} when every
finding no longer reproduces; exit 1 with the surviving cases.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

surviving = {}

# PLR-01: plateau runs must be non-resumable fail-closed.
from pipeline_plugins import _sac_plateau_lr as pl  # noqa: E402
with tempfile.TemporaryDirectory() as td:
    model = Path(td) / "best_model.zip"
    model.write_bytes(b"x")
    (Path(td) / "best_model.plateau_lr_state.json").write_text("{}")
    try:
        pl.assert_not_resuming_plateau_run(str(model))
        surviving["PLR-01"] = ("sidecar beside warm start was NOT "
                               "refused")
    except pl.SacPlateauLrError:
        pass
src_pipeline = (REPO / "pipeline_plugins" /
                "rl_pipeline_with_validation.py").read_text()
if "assert_not_resuming_plateau_run" not in src_pipeline:
    surviving["PLR-01-wiring"] = "guard absent from executing pipeline"
mod_doc = pl.__doc__ or ""
if "NON-RESUMABLE" not in mod_doc:
    surviving["PLR-01-relabel"] = "module still claims resume support"

# PLR-02: no long-horizon label; bounded screen label present.
src_tool = (REPO / "tools" / "wp4_cpu_smoke.py").read_text()
if "long_horizon_contract" in src_tool:
    surviving["PLR-02"] = "long_horizon_contract label still present"
if "BOUNDED_120_40_40_DAY_SCHEDULER_SCREEN" not in src_tool:
    surviving["PLR-02-label"] = "bounded screen label missing"

# PLR-03: aggregator excludes wall clock from causal conclusion.
src_agg = (REPO / "tools" / "plateau_screen_aggregate.py").read_text()
if "excluded_from_causal_conclusion" not in src_agg:
    surviving["PLR-03"] = "no causal exclusion of wall-clock facts"
if "PREDECLARED DECISION RULE" not in src_agg:
    surviving["PLR-03-rule"] = "decision rule not predeclared"

# PLR-04: diagnostic holdout naming.
if "internal_test_split" in src_tool:
    surviving["PLR-04"] = "internal_test_split key still present"
if '"diagnostic_holdout"' not in src_tool:
    surviving["PLR-04-rename"] = "diagnostic_holdout key missing"

print(json.dumps({"reproduced": bool(surviving),
                  "surviving": surviving}, indent=1))
sys.exit(1 if surviving else 0)
