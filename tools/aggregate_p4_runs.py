"""Aggregate Part III P4 training run summaries into a CSV + markdown table.

Walks logs/partIII/*_p4*/ directories, reads summary.json and config_out.json
from each run, and emits:
  - logs/partIII/p4_aggregate.csv   (one row per run)
  - logs/partIII/p4_aggregate.md    (grouped markdown tables)

Runs without summary.json (still training / crashed) are listed under "IN PROGRESS / FAILED".

Usage:
    python tools/aggregate_p4_runs.py
    python tools/aggregate_p4_runs.py --log_root /path/to/logs/partIII
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "partIII"

FIELDS = [
    "run_id",
    "asset",
    "algo",
    "features",
    "strategy",
    "seed",
    "tag",
    "total_timesteps",
    "trades_total",
    "trades_won",
    "total_return",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sqn",
    "final_equity",
    "episode_length",
    "episode_reward",
    "status",
    "run_dir",
]


def _parse_run_id(run_id: str) -> Dict[str, str]:
    # <asset>_<algo>_<features>_<strategy>_s<seed>_<utc>[_tag]
    parts = run_id.split("_")
    out = {"run_id": run_id, "asset": "?", "algo": "?", "features": "?", "strategy": "?", "seed": "?", "tag": "-"}
    for i, p in enumerate(parts):
        if p.startswith("s") and p[1:].isdigit() and i >= 3:
            out["seed"] = p[1:]
            out["asset"] = "_".join(parts[:2]) if len(parts) > 2 else parts[0]
            out["algo"] = parts[2] if len(parts) > 2 else "?"
            if i > 3:
                out["features"] = parts[3]
                out["strategy"] = "_".join(parts[4:i])
            # Remaining parts after seed: [<utc_timestamp>, <tag>?]
            tail = parts[i + 1:]
            if len(tail) >= 2:
                out["tag"] = "_".join(tail[1:])
            break
    return out


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect(log_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in sorted(log_root.iterdir()):
        if not d.is_dir() or "_p4" not in d.name:
            continue
        meta = _parse_run_id(d.name)
        row: Dict[str, Any] = {f: "" for f in FIELDS}
        row.update(meta)
        row["run_dir"] = str(d)

        summary = _load_json(d / "summary.json")
        cfg_out = _load_json(d / "config_out.json") or _load_json(d / "config.json") or {}
        row["total_timesteps"] = cfg_out.get("total_timesteps", "")

        if summary is not None:
            row["status"] = "done"
            for k in ("trades_total", "trades_won", "total_return", "max_drawdown_pct",
                      "sharpe_ratio", "sqn", "final_equity", "episode_length", "episode_reward"):
                row[k] = summary.get(k, "")
        else:
            tlog = d / "train.log"
            sz = tlog.stat().st_size if tlog.exists() else 0
            row["status"] = f"running (log={sz}B)" if tlog.exists() else "missing"
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _fmt(v: Any, nd: int = 4) -> str:
    if v in (None, ""):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_md(rows: List[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    done = [r for r in rows if r.get("status") == "done"]
    pending = [r for r in rows if r.get("status") != "done"]

    lines: List[str] = ["# P4 Aggregate — Part III training runs", ""]
    lines.append(f"- Log root: `{DEFAULT_LOG_ROOT}`")
    lines.append(f"- Runs found: {len(rows)}  (done={len(done)}, pending/other={len(pending)})")
    lines.append("")

    # Group done rows by (asset, algo, tag)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in done:
        key = f"{r.get('asset','?')} / {r.get('algo','?')} / {r.get('tag','-')}"
        groups.setdefault(key, []).append(r)

    lines.append("## Completed runs")
    lines.append("")
    lines.append("| asset/algo/tag | seed | steps | trades | ret | DD% | Sharpe | SQN | final_eq |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        for r in sorted(groups[key], key=lambda x: str(x.get("seed", ""))):
            lines.append("| {k} | {s} | {ts} | {tr} | {rt} | {dd} | {sh} | {sq} | {eq} |".format(
                k=key,
                s=r.get("seed", "-"),
                ts=_fmt(r.get("total_timesteps"), 0),
                tr=_fmt(r.get("trades_total"), 0),
                rt=_fmt(r.get("total_return"), 5),
                dd=_fmt(r.get("max_drawdown_pct"), 2),
                sh=_fmt(r.get("sharpe_ratio"), 3),
                sq=_fmt(r.get("sqn"), 2),
                eq=_fmt(r.get("final_equity"), 2),
            ))
    lines.append("")

    # Group aggregate stats (mean across seeds)
    import statistics
    lines.append("## Group aggregates (mean across seeds)")
    lines.append("")
    lines.append("| asset/algo/tag | n_seeds | ret_mean | ret_std | Sharpe_mean | Sharpe_std | DD_mean | trades_mean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        grp = groups[key]
        def _vals(k):
            out = []
            for r in grp:
                v = r.get(k)
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out
        ret = _vals("total_return"); sh = _vals("sharpe_ratio"); dd = _vals("max_drawdown_pct"); tr = _vals("trades_total")
        def _mean(xs): return f"{statistics.mean(xs):.4f}" if xs else "-"
        def _std(xs): return f"{statistics.stdev(xs):.4f}" if len(xs) > 1 else "-"
        lines.append(f"| {key} | {len(grp)} | {_mean(ret)} | {_std(ret)} | {_mean(sh)} | {_std(sh)} | {_mean(dd)} | {_mean(tr)} |")
    lines.append("")

    if pending:
        lines.append("## Pending / non-terminal runs")
        lines.append("")
        lines.append("| run_id | status |")
        lines.append("|---|---|")
        for r in pending:
            lines.append(f"| {r['run_id']} | {r['status']} |")
        lines.append("")

    out_md.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    log_root = Path(args.log_root)
    rows = collect(log_root)

    out_csv = Path(args.out_csv) if args.out_csv else log_root / "p4_aggregate.csv"
    out_md = Path(args.out_md) if args.out_md else log_root / "p4_aggregate.md"
    write_csv(rows, out_csv)
    write_md(rows, out_md)

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"total runs: {len(rows)}  done: {sum(1 for r in rows if r['status']=='done')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
