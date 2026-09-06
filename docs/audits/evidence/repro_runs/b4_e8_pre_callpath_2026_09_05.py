"""PRE freeze for order @e8bb500f (E8-E12): the call-path report —
does ANY current executable consume a B4 cell and perform the five
campaign stages? Expected: no complete executor or adjudicator
exists; tools/b4_run_cell.py proves one bounded mechanics segment
only."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TOOLS = sorted((REPO / "tools").glob("*.py"))

# stages 1-4 map TEXTUAL evidence (mentions included); the stage-5
# check looks for actual adjudication MACHINERY definitions — a
# materializer that talks about "12 cells" in prose aggregates
# nothing.
STAGES = {
    "1_epoch_loop_under_materialized_terms": (
        r"run_pipeline|max_epochs.*patience|_train_epochs"),
    "2_causal_validation_checkpoint_selection": (
        r"selection_metric|checkpoint|_eval_on_split"),
    "3_frozen_checkpoint_outer_origin_eval": (
        r"outer_validation|frozen.*checkpoint|score.*outer"),
    "4_per_bar_net_returns_reviewed_costs": (
        r"per_bar|net_return|close_reason"),
    "5_twelve_cell_aggregation_g1_decision_machinery": (
        r"def .*(spa|hansen|politis|iqm|dsr)|"
        r"stationary_bootstrap|block_length"),
}

consumes_b4_cell = []
for t in TOOLS:
    src = t.read_text()
    if "B4_CELL_CONFIGS" in src:
        consumes_b4_cell.append(t.name)
print("executables that consume B4_CELL_CONFIGS:", consumes_b4_cell)

report = {}
for t in TOOLS:
    src = t.read_text()
    if "B4_CELL_CONFIGS" not in src:
        continue
    hits = {k: bool(re.search(rx, src)) for k, rx in STAGES.items()}
    report[t.name] = hits
print(json.dumps(report, indent=1))

runner_src = (REPO / "tools" / "b4_run_cell.py").read_text()
facts = {
    "runner_has_epoch_loop": "max_epochs" in runner_src
    and "run_pipeline" in runner_src,
    "runner_has_checkpoint_selection":
        "selection_metric" in runner_src
        and "_eval_on_split" in runner_src,
    "runner_scores_outer_origin": "outer_validation" in runner_src,
    "runner_emits_per_bar_returns": "per_bar" in runner_src,
    "any_tool_aggregates_12_cells_g1": any(
        r["5_twelve_cell_aggregation_g1_decision_machinery"]
        for r in report.values()),
    "adjudicator_exists": (REPO / "tools" /
                           "b4_adjudicator.py").exists(),
    "campaign_ledger_exists": (REPO / "tools" /
                               "b4_campaign_ledger.py").exists(),
    "campaign_executor_exists": (REPO / "tools" /
                                 "b4_campaign_executor.py").exists(),
}
print(json.dumps(facts, indent=1))
assert not facts["runner_has_epoch_loop"]
assert not facts["runner_has_checkpoint_selection"]
assert not facts["runner_scores_outer_origin"]
assert not facts["any_tool_aggregates_12_cells_g1"]
assert not facts["adjudicator_exists"]
assert not facts["campaign_ledger_exists"]
assert not facts["campaign_executor_exists"]

# the real lifecycle EXISTS in the pipeline plugin but nothing maps a
# B4 cell into it:
pipe = (REPO / "pipeline_plugins/rl_pipeline_with_validation.py"
        ).read_text()
print("pipeline has run_pipeline lifecycle:",
      "def run_pipeline" in pipe)
print("pipeline consumes nested_split_contract:",
      "nested_split_contract" in pipe)
b4_to_pipeline = any(
    "run_pipeline" in (REPO / "tools" / n).read_text()
    for n in consumes_b4_cell)
print("any B4-cell consumer invokes run_pipeline:", b4_to_pipeline)
assert not b4_to_pipeline

print("\nPRE CONFIRMED: one bounded mechanics segment exists "
      "(b4_run_cell.py); NO complete campaign executor, ledger or "
      "doc-41 adjudicator exists on any call path")
