"""P5.6 — rank P5 hold-out groups for P5b trade-rate fine-tuning candidates.

Reads logs/partIII/p5_eval_holdout.csv (produced by p5_eval_holdout.py), groups
by (asset, algo, tag), and ranks groups by mean hold-out Sharpe with optional
drawdown tiebreaker. Emits:

  logs/partIII/p5_rank.csv   - group aggregates, one row per (asset, algo, tag)
  logs/partIII/p5_rank.md    - human-readable ranking with top-N highlighted

Trade rate is reported as trades/week based on episode_length hours
(1 week = 168 hours).

Usage:
  python tools/p5_rank.py                 # default top 3
  python tools/p5_rank.py --top 5
  python tools/p5_rank.py --min_seeds 3   # ignore groups with <3 seeds
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs" / "partIII"


def _to_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _mean_std(vs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vs:
        return (None, None)
    m = statistics.mean(vs)
    s = statistics.stdev(vs) if len(vs) > 1 else 0.0
    return (m, s)


def _load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _group(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r.get("asset", ""), r.get("algo", ""), r.get("tag", ""))
        out.setdefault(key, []).append(r)
    return out


def _aggregate(groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    aggs: List[Dict[str, Any]] = []
    for (asset, algo, tag), items in groups.items():
        sharpe = [v for v in (_to_float(r.get("sharpe_ratio")) for r in items) if v is not None]
        ret = [v for v in (_to_float(r.get("total_return")) for r in items) if v is not None]
        dd = [v for v in (_to_float(r.get("max_drawdown_pct")) for r in items) if v is not None]
        trades = [v for v in (_to_float(r.get("trades_total")) for r in items) if v is not None]
        eplens = [v for v in (_to_float(r.get("episode_length")) for r in items) if v is not None]
        sh_m, sh_s = _mean_std(sharpe)
        ret_m, ret_s = _mean_std(ret)
        dd_m, _ = _mean_std(dd)
        tr_m, _ = _mean_std(trades)
        ep_m, _ = _mean_std(eplens)
        tr_per_week = None
        if tr_m is not None and ep_m and ep_m > 0:
            tr_per_week = tr_m / (ep_m / 168.0)
        aggs.append({
            "asset": asset,
            "algo": algo,
            "tag": tag,
            "n_seeds": len(items),
            "sharpe_mean": sh_m,
            "sharpe_std": sh_s,
            "ret_mean": ret_m,
            "ret_std": ret_s,
            "dd_mean": dd_m,
            "trades_mean": tr_m,
            "episode_length_mean": ep_m,
            "trades_per_week": tr_per_week,
        })
    return aggs


def _rank_key(g: Dict[str, Any]) -> Tuple[float, float]:
    # Primary: sharpe_mean desc. Tiebreaker: lower drawdown preferred.
    sh = g.get("sharpe_mean")
    dd = g.get("dd_mean")
    sh_v = -1e9 if sh is None else sh
    dd_v = 1e9 if dd is None else dd
    return (-sh_v, dd_v)


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "rank", "asset", "algo", "tag", "n_seeds",
        "sharpe_mean", "sharpe_std", "ret_mean", "ret_std",
        "dd_mean", "trades_mean", "trades_per_week", "episode_length_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(path: Path, rows: List[Dict[str, Any]], top: int) -> None:
    lines = [
        "# P5.6 Hold-out ranking",
        "",
        f"- Source: `logs/partIII/p5_eval_holdout.csv`",
        f"- Groups ranked: {len(rows)}",
        f"- Top-{top} highlighted as P5b fine-tune candidates.",
        "",
        "| rank | asset / algo / tag | n | Sharpe_mean ± std | ret_mean ± std | DD_mean | trades_mean | trades/wk | candidate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "**yes**" if r["rank"] <= top else ""
        lines.append(
            "| {rk} | {a} / {al} / {t} | {n} | {sm} ± {ss} | {rm} ± {rs} | {dm} | {tm} | {tw} | {mk} |".format(
                rk=r["rank"],
                a=r["asset"], al=r["algo"], t=r["tag"], n=r["n_seeds"],
                sm=_fmt(r["sharpe_mean"], 3), ss=_fmt(r["sharpe_std"], 3),
                rm=_fmt(r["ret_mean"], 5), rs=_fmt(r["ret_std"], 5),
                dm=_fmt(r["dd_mean"], 2),
                tm=_fmt(r["trades_mean"], 1),
                tw=_fmt(r["trades_per_week"], 2),
                mk=mark,
            )
        )
    lines += [
        "",
        "## P5b next steps (for top candidates only)",
        "",
        "For each candidate group, run a small grid fine-tune (no retraining from scratch):",
        "",
        "1. **k_sl / k_tp geometry** — sweep k_tp in {3.0, 2.5, 2.0} at fixed k_sl=2.5 (lower TP → higher turnover).",
        "2. **ATR no-trade band** — edit `strategy_plugin.direct_atr_sltp` threshold from 0.33·ATR → 0.20·ATR (more trades) or 0.50·ATR (fewer).",
        "3. **ent_coef** — short 200k-step PPO fine-tune from policy.zip with ent_coef reduced by 2–5×.",
        "4. **Soft rate bonus** — add Gaussian reward bonus centered on 3/week, σ=2/week, weight ≤ 0.05; 500k fine-tune from checkpoint.",
        "",
        "Select whichever knob moves trades/wk toward 3 without collapsing Sharpe.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(LOG_ROOT / "p5_eval_holdout.csv"),
                    help="path to p5_eval_holdout.csv")
    ap.add_argument("--top", type=int, default=3, help="top-N to highlight as P5b candidates")
    ap.add_argument("--min_seeds", type=int, default=2,
                    help="ignore groups with fewer seeds than this")
    ap.add_argument("--log_root", default=str(LOG_ROOT))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[p5_rank] ERROR: {csv_path} not found. Run p5_eval_holdout.py first.", file=sys.stderr)
        return 2

    rows = _load_rows(csv_path)
    groups = _group(rows)
    aggs = _aggregate(groups)
    aggs = [a for a in aggs if a["n_seeds"] >= args.min_seeds]
    aggs.sort(key=_rank_key)
    for i, a in enumerate(aggs, start=1):
        a["rank"] = i

    out_csv = Path(args.log_root) / "p5_rank.csv"
    out_md = Path(args.log_root) / "p5_rank.md"
    _write_csv(out_csv, aggs)
    _write_md(out_md, aggs, args.top)

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print()
    print(f"=== Top {args.top} P5b fine-tune candidates ===")
    for r in aggs[: args.top]:
        print(
            f"  #{r['rank']} {r['asset']}/{r['algo']}/{r['tag']} "
            f"n={r['n_seeds']} Sharpe={_fmt(r['sharpe_mean'],3)}±{_fmt(r['sharpe_std'],3)} "
            f"ret={_fmt(r['ret_mean'],5)} DD={_fmt(r['dd_mean'],2)}% "
            f"trades/wk={_fmt(r['trades_per_week'],2)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
