"""P5.5 — hold-out evaluation harness.

For every run_dir under logs/partIII/ with a saved policy.zip:
  1. Load its config_out.json (or config.json).
  2. Override input_data_file → data/project2/<asset>/d6.csv.
  3. Set load_model → policy.zip (forces inference mode).
  4. Write temp config + run `agent-multi --load_config` in a subprocess.
  5. Store rollout summary in <run_dir>/p5_eval_d6.json.

Then aggregate across seeds per (asset, algo, tag) and emit:
  logs/partIII/p5_eval_holdout.csv
  logs/partIII/p5_eval_holdout.md

Usage:
  python tools/p5_eval_holdout.py                    # evaluate every run
  python tools/p5_eval_holdout.py --pattern p4iter2  # only that tag
  python tools/p5_eval_holdout.py --skip-if-exists   # don't re-run if p5_eval_d6.json already present
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs" / "partIII"
DATA_ROOT = REPO_ROOT / "data" / "project2"

EVAL_FNAME = "p5_eval_d6.json"


def _load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    for name in ("config_out.json", "config.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def _asset_d6(asset: str) -> Path:
    return DATA_ROOT / asset / "d6.csv"


def _parse_run_id(run_id: str) -> Dict[str, str]:
    # <asset>_<algo>_<features...>_<strategy...>_s<seed>_<ts>_<tag?>
    parts = run_id.split("_")
    out: Dict[str, str] = {"run_id": run_id}
    asset = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
    out["asset"] = asset
    out["algo"] = parts[2] if len(parts) > 2 else ""
    seed_idx = next((i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit()), -1)
    if seed_idx > 0:
        out["seed"] = parts[seed_idx][1:]
        tail = parts[seed_idx + 1 :]
        if len(tail) >= 2:
            out["tag"] = "_".join(tail[1:])
        else:
            out["tag"] = ""
    return out


def _eval_one(run_dir: Path, *, skip_if_exists: bool) -> Optional[Dict[str, Any]]:
    cfg = _load_run_config(run_dir)
    if cfg is None:
        print(f"[p5_eval] skip {run_dir.name}: no config", flush=True)
        return None
    policy = run_dir / "policy.zip"
    # seed_sweep writes policy.zip, but older runs may save as agent_model.zip
    if not policy.exists():
        alt = run_dir / "agent_model.zip"
        if alt.exists():
            policy = alt
    if not policy.exists():
        print(f"[p5_eval] skip {run_dir.name}: no policy zip", flush=True)
        return None

    asset = cfg.get("asset") or _parse_run_id(run_dir.name).get("asset", "")
    d6 = _asset_d6(asset)
    if not d6.exists():
        print(f"[p5_eval] skip {run_dir.name}: d6 missing for {asset}", flush=True)
        return None

    eval_out = run_dir / EVAL_FNAME
    if skip_if_exists and eval_out.exists():
        try:
            return json.loads(eval_out.read_text(encoding="utf-8"))
        except Exception:
            pass

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["mode"] = "inference"
    eval_cfg["load_model"] = str(policy)
    eval_cfg["input_data_file"] = str(d6)
    eval_cfg["results_file"] = str(eval_out)
    eval_cfg.pop("save_model", None)
    eval_cfg.pop("save_config", None)
    eval_cfg["quiet_mode"] = True
    # Ensure we don't accidentally re-train if some upstream toggles it.
    eval_cfg["use_optimizer"] = False

    tmp_cfg = run_dir / "_p5_eval_cfg.json"
    tmp_cfg.write_text(json.dumps(eval_cfg, indent=2), encoding="utf-8")

    log_path = run_dir / "p5_eval.log"
    print(f"[p5_eval] {run_dir.name}  d6={d6.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            ["agent-multi", "--load_config", str(tmp_cfg), "--quiet_mode"],
            cwd=REPO_ROOT,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        tmp_cfg.unlink()
    except OSError:
        pass
    if proc.returncode != 0 or not eval_out.exists():
        print(f"[p5_eval] FAIL {run_dir.name} rc={proc.returncode} (see {log_path})", flush=True)
        return None
    try:
        return json.loads(eval_out.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[p5_eval] FAIL parse {run_dir.name}: {e}", flush=True)
        return None


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r.get("asset", ""), r.get("algo", ""), r.get("tag", ""))
        groups.setdefault(key, []).append(r)
    out = []
    for (asset, algo, tag), items in sorted(groups.items()):
        def col(name: str) -> List[float]:
            vals = []
            for it in items:
                v = it.get(name)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return vals

        def ms(name: str):
            vs = col(name)
            if not vs:
                return (None, None)
            m = statistics.mean(vs)
            s = statistics.stdev(vs) if len(vs) > 1 else None
            return (m, s)

        ret_m, ret_s = ms("total_return")
        sharpe_m, sharpe_s = ms("sharpe_ratio")
        dd_m, _ = ms("max_drawdown_pct")
        tr_m, _ = ms("trades_total")
        out.append({
            "asset": asset,
            "algo": algo,
            "tag": tag,
            "n_seeds": len(items),
            "ret_mean": ret_m,
            "ret_std": ret_s,
            "sharpe_mean": sharpe_m,
            "sharpe_std": sharpe_s,
            "dd_mean": dd_m,
            "trades_mean": tr_m,
        })
    return out


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def _write_md(path: Path, per_run: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> None:
    lines = [
        "# P5.5 Hold-out evaluation on d6",
        "",
        f"- Log root: `{LOG_ROOT}`",
        f"- Runs evaluated: {len(per_run)}",
        "",
        "## Per-run d6 metrics",
        "",
        "| asset/algo/tag | seed | trades | ret | DD% | Sharpe | final_eq |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(per_run, key=lambda x: (x.get("asset",""), x.get("algo",""), x.get("tag",""), str(x.get("seed","")))):
        lines.append(
            "| {a} / {al} / {t} | {s} | {tr} | {ret} | {dd} | {sh} | {fe} |".format(
                a=r.get("asset", ""), al=r.get("algo", ""), t=r.get("tag", ""),
                s=r.get("seed", ""),
                tr=r.get("trades_total", "-"),
                ret=_fmt(r.get("total_return"), 5),
                dd=_fmt(r.get("max_drawdown_pct"), 2),
                sh=_fmt(r.get("sharpe_ratio"), 3),
                fe=_fmt(r.get("final_equity"), 2),
            )
        )
    lines += [
        "",
        "## Group aggregates on d6 (mean ± std across seeds)",
        "",
        "| asset/algo/tag | n | ret_mean | ret_std | Sharpe_mean | Sharpe_std | DD_mean | trades_mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in groups:
        lines.append(
            "| {a} / {al} / {t} | {n} | {rm} | {rs} | {sm} | {ss} | {dm} | {tm} |".format(
                a=g["asset"], al=g["algo"], t=g["tag"], n=g["n_seeds"],
                rm=_fmt(g["ret_mean"]), rs=_fmt(g["ret_std"]),
                sm=_fmt(g["sharpe_mean"], 3), ss=_fmt(g["sharpe_std"], 3),
                dm=_fmt(g["dd_mean"], 2), tm=_fmt(g["trades_mean"], 1),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, per_run: List[Dict[str, Any]]) -> None:
    import csv
    fields = [
        "run_id", "asset", "algo", "tag", "seed",
        "trades_total", "total_return", "max_drawdown_pct",
        "sharpe_ratio", "sqn", "final_equity", "episode_length",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_run:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="", help="substring match on run_id (e.g. 'p4iter2')")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="reuse existing p5_eval_d6.json if present")
    ap.add_argument("--log_root", default=str(LOG_ROOT))
    args = ap.parse_args()

    log_root = Path(args.log_root)
    per_run: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        if args.pattern and args.pattern not in run_dir.name:
            continue
        summary = _eval_one(run_dir, skip_if_exists=args.skip_if_exists)
        if summary is None:
            continue
        meta = _parse_run_id(run_dir.name)
        row: Dict[str, Any] = {"run_id": run_dir.name}
        row.update(meta)
        row.update({
            k: summary.get(k)
            for k in (
                "trades_total", "total_return", "max_drawdown_pct",
                "sharpe_ratio", "sqn", "final_equity", "episode_length",
            )
        })
        per_run.append(row)

    groups = _aggregate(per_run)
    csv_path = log_root / "p5_eval_holdout.csv"
    md_path = log_root / "p5_eval_holdout.md"
    _write_csv(csv_path, per_run)
    _write_md(md_path, per_run, groups)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"evaluated {len(per_run)} runs, {len(groups)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
