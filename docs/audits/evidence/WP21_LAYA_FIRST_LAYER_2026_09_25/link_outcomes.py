"""WP23 step 1 for WP21(b): link every trading decision to the closure-table row that judged it.

Nothing here decides whether a link may exist -- `m5phet.decide.outcome` does, and every refusal it returns is kept
with its name and counted. The point of running it against a table whose policy row is refused is to record WHY the
count is zero, in the framework's own words, instead of leaving the calibration report to be read as if the links had
simply not been attempted.

    python link_outcomes.py <table.json> <decisions dir> <outcomes dir> <summary.json>
"""

import json
import sys
from collections import Counter
from pathlib import Path

from m5phet import decide


def main(table_path, decisions_dir, outcomes_dir, summary_path, stage="laya_first_layer"):
    table = json.loads(Path(table_path).read_text())
    rows = [row for area in table["areas"] for row in area["rows"] if row["stage"] == stage]
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one row for stage {stage!r}, found {len(rows)}")
    row = rows[0]

    attempted, linked, refusals = 0, [], Counter()
    detail = {}
    for path in sorted(Path(decisions_dir).expanduser().glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        if record.get("kind") != "trading_decision":
            continue
        attempted += 1
        answer = decide.outcome(str(path), row, out_dir=outcomes_dir)
        if answer.get("status") == "OK":
            linked.append(answer["record_path"])
        else:
            refusals[answer.get("refusal")] += 1
            detail.setdefault(answer.get("refusal"), str(answer.get("why"))[:400])

    summary = {"schema": "m5phet.wp21_outcome_linking.v1",
               "execution_authorized": False,
               "stage": stage,
               "table_row": {field: row.get(field) for field in ("stage", "status", "comparability", "rank",
                                                                 "metric", "reason")},
               "decision_records_of_this_kind": attempted,
               "linked": len(linked),
               "refused": dict(sorted(refusals.items())),
               "refusal_reasons": detail,
               "reading": ("a decision becomes evidence about the chooser only when a measured, COMPARABLE, ranked "
                           "closure-table row judged the pipeline it led to. Both refusals below are that rule "
                           "working, not a failure to run it")}
    Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:])
