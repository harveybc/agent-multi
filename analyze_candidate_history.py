#!/usr/bin/env python3
"""
Analyze NEAT candidate history CSVs from all nodes.

Usage:
    python analyze_candidate_history.py [--fetch]

With --fetch: pulls CSVs from dragon (192.168.0.107) and gamma (192.168.0.106)
              via scp before analysis.
Without:      analyzes only the local omega CSV.

Outputs a text report + saves plots to /tmp/neat_analysis/
"""
import argparse
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).parent / "examples" / "results" / "phase_1b_binary"
OMEGA_CSV = RESULTS_DIR / "phase_1b_binary_tft_buy_entry_1d_optimization_candidate_history.csv"
ANALYSIS_DIR = Path("/tmp/neat_analysis")

REMOTE_CSV = "Documents/GitHub/ioin/examples/results/phase_1b_binary/phase_1b_binary_tft_buy_entry_1d_optimization_candidate_history.csv"
NODES = {
    "omega": {"host": None, "csv": OMEGA_CSV},
    "dragon": {"host": "192.168.0.107", "csv": ANALYSIS_DIR / "dragon_history.csv"},
    "gamma": {"host": "192.168.0.106", "csv": ANALYSIS_DIR / "gamma_history.csv"},
}

GENOME_PARAMS = [
    "batch_size", "l2_reg", "learning_rate", "min_delta",
    "positional_encoding", "tft_dropout", "tft_hidden_units",
    "tft_lstm_layers", "tft_num_heads", "use_log1p_features", "window_size",
]

METRIC_COLS = [
    "fitness", "train_mcc", "train_f1", "val_mcc", "val_f1",
    "test_mcc", "test_f1",
]


def fetch_remote_csvs():
    """Pull CSVs from dragon and gamma via scp."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for name, info in NODES.items():
        if info["host"] is None:
            continue
        dest = info["csv"]
        src = f"harveybc@{info['host']}:/home/harveybc/{REMOTE_CSV}"
        print(f"  Fetching {name} ({info['host']})...", end=" ")
        try:
            r = subprocess.run(
                ["scp", "-P", "62024", "-o", "ConnectTimeout=10", src, str(dest)],
                capture_output=True, timeout=20,
            )
            if r.returncode == 0 and dest.exists():
                print(f"OK ({dest.stat().st_size} bytes)")
            else:
                print(f"FAILED (rc={r.returncode})")
        except Exception as e:
            print(f"ERROR: {e}")


def load_all(fetch: bool) -> dict[str, pd.DataFrame]:
    """Load CSVs from all available nodes."""
    if fetch:
        fetch_remote_csvs()

    frames = {}
    for name, info in NODES.items():
        p = info["csv"]
        if p.exists() and p.stat().st_size > 100:
            try:
                df = pd.read_csv(p)
                if len(df) > 0:
                    df["node"] = name
                    frames[name] = df
                    print(f"  {name}: {len(df)} candidates loaded")
            except Exception as e:
                print(f"  {name}: read error — {e}")
        else:
            print(f"  {name}: no CSV or empty")
    return frames


def analyze_diversity(df: pd.DataFrame, label: str):
    """Analyze genome diversity within a node."""
    print(f"\n{'='*70}")
    print(f"  GENOME DIVERSITY — {label} ({len(df)} candidates)")
    print(f"{'='*70}")

    # Unique genomes (round floats for comparison)
    rounded = df[GENOME_PARAMS].copy()
    for c in rounded.columns:
        if rounded[c].dtype in (float, np.float64):
            rounded[c] = rounded[c].round(6)
    unique_genomes = rounded.drop_duplicates().shape[0]
    dup_rate = 1.0 - unique_genomes / len(df) if len(df) > 0 else 0
    print(f"\n  Total candidates: {len(df)}")
    print(f"  Unique genomes:   {unique_genomes} ({100*unique_genomes/len(df):.1f}%)")
    print(f"  Duplicate rate:   {100*dup_rate:.1f}%")

    # Per-parameter diversity
    print(f"\n  Per-parameter diversity:")
    print(f"  {'Parameter':<25} {'Unique':>8} {'Min':>12} {'Max':>12} {'Std':>12} {'Mode':>12}")
    print(f"  {'-'*85}")
    for p in GENOME_PARAMS:
        if p not in df.columns:
            continue
        col = df[p]
        n_unique = col.nunique()
        try:
            mode_val = col.mode().iloc[0] if len(col.mode()) > 0 else "N/A"
            std_val = col.astype(float).std()
            min_val = col.min()
            max_val = col.max()
            print(f"  {p:<25} {n_unique:>8} {str(min_val):>12} {str(max_val):>12} {std_val:>12.6f} {str(mode_val):>12}")
        except (TypeError, ValueError):
            print(f"  {p:<25} {n_unique:>8} {'(non-numeric)':>12}")

    # Check for params that never change
    frozen = [p for p in GENOME_PARAMS if p in df.columns and df[p].nunique() <= 1]
    if frozen:
        print(f"\n  ⚠ FROZEN PARAMS (never vary): {', '.join(frozen)}")

    # Stage breakdown
    if "stage_name" in df.columns:
        print(f"\n  Stage breakdown:")
        for stage, grp in df.groupby("stage_name"):
            n = len(grp)
            champs = grp["is_champion"].sum() if "is_champion" in grp.columns else 0
            u = grp[GENOME_PARAMS].drop_duplicates().shape[0]
            print(f"    {stage:>15}: {n:>4} candidates, {u:>4} unique genomes, {champs} champions")


def analyze_fitness(df: pd.DataFrame, label: str):
    """Analyze fitness distribution and champion progression."""
    print(f"\n{'='*70}")
    print(f"  FITNESS ANALYSIS — {label}")
    print(f"{'='*70}")

    if "fitness" not in df.columns:
        print("  No fitness column found")
        return

    fitness = df["fitness"].astype(float)
    print(f"\n  Fitness stats:")
    print(f"    Mean:   {fitness.mean():.6f}")
    print(f"    Std:    {fitness.std():.6f}")
    print(f"    Min:    {fitness.min():.6f} (best)")
    print(f"    Max:    {fitness.max():.6f} (worst)")
    print(f"    Median: {fitness.median():.6f}")

    # Champion progression
    if "is_champion" in df.columns:
        champs = df[df["is_champion"] == 1]
        print(f"\n  Champions: {len(champs)} out of {len(df)} ({100*len(champs)/len(df):.1f}%)")
        if len(champs) > 0:
            print(f"  Champion progression:")
            for _, row in champs.iterrows():
                gen = row.get("generation", "?")
                eval_n = row.get("total_eval", "?")
                fit = row["fitness"]
                stage = row.get("stage_name", "?")
                ws = row.get("window_size", "?")
                print(f"    Eval #{eval_n:>4} | Gen {gen:>3} | {stage:<12} | "
                      f"fitness={fit:.6f} | window_size={ws}")

    # Metric correlations with fitness
    print(f"\n  Metric ranges:")
    for m in METRIC_COLS:
        if m in df.columns and m != "fitness":
            vals = pd.to_numeric(df[m], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"    {m:<15}: min={vals.min():.6f}  max={vals.max():.6f}  std={vals.std():.6f}")


def analyze_mutations(df: pd.DataFrame, label: str):
    """Detect if mutations/crossover are actually producing variation."""
    print(f"\n{'='*70}")
    print(f"  MUTATION EFFECTIVENESS — {label}")
    print(f"{'='*70}")

    if len(df) < 4:
        print("  Not enough data yet")
        return

    # Compare consecutive candidates within same generation
    gen_changes = []
    for gen, grp in df.groupby("generation"):
        if len(grp) < 2:
            continue
        prev = None
        for _, row in grp.iterrows():
            if prev is not None:
                diffs = 0
                for p in GENOME_PARAMS:
                    if p in row.index and p in prev.index:
                        try:
                            if str(row[p]) != str(prev[p]):
                                diffs += 1
                        except Exception:
                            pass
                gen_changes.append({"gen": gen, "param_diffs": diffs})
            prev = row

    if gen_changes:
        changes_df = pd.DataFrame(gen_changes)
        avg_diffs = changes_df["param_diffs"].mean()
        zero_diffs = (changes_df["param_diffs"] == 0).sum()
        print(f"\n  Between consecutive candidates (same gen):")
        print(f"    Avg params differing:     {avg_diffs:.2f} / {len(GENOME_PARAMS)}")
        print(f"    Zero-diff pairs:          {zero_diffs} / {len(changes_df)} "
              f"({100*zero_diffs/len(changes_df):.1f}%)")
        print(f"    Diff distribution:        {dict(Counter(changes_df['param_diffs']))}")

        if avg_diffs < 1.5:
            print(f"\n  ⚠ WARNING: Very low mutation diversity!")
            print(f"    Average only {avg_diffs:.1f} params differ between candidates.")
            print(f"    This suggests mutations are too conservative or too many params are frozen.")

    # Cross-generation: compare gen N best vs gen N+1 best
    gens = sorted(df["generation"].unique())
    if len(gens) >= 2:
        print(f"\n  Generation-to-generation best genome changes:")
        prev_best = None
        for gen in gens:
            gen_df = df[df["generation"] == gen]
            best_row = gen_df.loc[gen_df["fitness"].astype(float).idxmin()]
            if prev_best is not None:
                diffs = []
                for p in GENOME_PARAMS:
                    if p in best_row.index:
                        try:
                            if str(best_row[p]) != str(prev_best[p]):
                                diffs.append(p)
                        except Exception:
                            pass
                print(f"    Gen {int(gen)-1} → {int(gen)}: {len(diffs)} params changed: "
                      f"{', '.join(diffs) if diffs else '(none)'}")
            prev_best = best_row


def analyze_param_impact(df: pd.DataFrame, label: str):
    """Correlate each parameter with fitness."""
    print(f"\n{'='*70}")
    print(f"  PARAMETER IMPACT ON FITNESS — {label}")
    print(f"{'='*70}")

    if len(df) < 10:
        print("  Need ≥10 candidates for meaningful correlation")
        return

    fitness = pd.to_numeric(df["fitness"], errors="coerce")
    print(f"\n  {'Parameter':<25} {'Correlation':>12} {'Significance':>15}")
    print(f"  {'-'*55}")

    for p in GENOME_PARAMS:
        if p not in df.columns:
            continue
        vals = pd.to_numeric(df[p], errors="coerce")
        valid = fitness.notna() & vals.notna()
        if valid.sum() < 5:
            print(f"  {p:<25} {'(non-numeric)':>12}")
            continue
        corr = fitness[valid].corr(vals[valid])
        n = valid.sum()
        # Simple significance check
        if n > 3 and abs(corr) > 0:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-10))
            sig = "***" if abs(t_stat) > 3.3 else "**" if abs(t_stat) > 2.6 else "*" if abs(t_stat) > 2.0 else ""
        else:
            sig = ""
        direction = "← better" if corr > 0.1 else "→ better" if corr < -0.1 else "~neutral"
        print(f"  {p:<25} {corr:>12.4f} {sig:>5}  {direction}")


def cross_node_analysis(all_dfs: dict[str, pd.DataFrame]):
    """Compare genomes across nodes."""
    if len(all_dfs) < 2:
        print("\n  Need ≥2 nodes for cross-node comparison")
        return

    print(f"\n{'='*70}")
    print(f"  CROSS-NODE COMPARISON")
    print(f"{'='*70}")

    combined = pd.concat(all_dfs.values(), ignore_index=True)
    print(f"\n  Total candidates across all nodes: {len(combined)}")

    for name, df in all_dfs.items():
        f = pd.to_numeric(df["fitness"], errors="coerce")
        champs = df["is_champion"].sum() if "is_champion" in df.columns else 0
        print(f"    {name:>8}: {len(df):>4} candidates, {champs} champions, "
              f"best_fitness={f.min():.6f}")

    # Cross-node genome overlap
    rounded_sets = {}
    for name, df in all_dfs.items():
        rounded = df[GENOME_PARAMS].copy()
        for c in rounded.columns:
            try:
                rounded[c] = rounded[c].astype(str)
            except Exception:
                pass
        tuples = set(rounded.apply(tuple, axis=1))
        rounded_sets[name] = tuples

    names = list(rounded_sets.keys())
    print(f"\n  Cross-node genome overlap:")
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            overlap = len(rounded_sets[n1] & rounded_sets[n2])
            total = len(rounded_sets[n1] | rounded_sets[n2])
            print(f"    {n1} ∩ {n2}: {overlap} shared genomes out of "
                  f"{len(rounded_sets[n1])}+{len(rounded_sets[n2])} "
                  f"(Jaccard={overlap/max(total,1):.3f})")

    # Compare fitness distributions
    print(f"\n  Fitness distribution by node:")
    for name, df in all_dfs.items():
        f = pd.to_numeric(df["fitness"], errors="coerce").dropna()
        print(f"    {name:>8}: mean={f.mean():.6f}  std={f.std():.6f}  "
              f"best={f.min():.6f}  worst={f.max():.6f}")


def save_plots(all_dfs: dict[str, pd.DataFrame]):
    """Generate and save analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available — skipping plots")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_dfs.values(), ignore_index=True)

    # 1. Fitness over evaluations per node
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in all_dfs.items():
        f = pd.to_numeric(df["fitness"], errors="coerce")
        ax.scatter(df["total_eval"], f, alpha=0.4, s=15, label=f"{name} candidates")
        # Champion line
        if "champion_fitness" in df.columns:
            cf = pd.to_numeric(df["champion_fitness"], errors="coerce")
            ax.plot(df["total_eval"], cf, linewidth=2, label=f"{name} champion")
    ax.set_xlabel("Candidate Evaluation #")
    ax.set_ylabel("Fitness (more negative = better)")
    ax.set_title("Fitness Progression by Node")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "fitness_progression.png", dpi=150)
    plt.close()
    print(f"  Saved: {ANALYSIS_DIR}/fitness_progression.png")

    # 2. Parameter distributions (histograms)
    numeric_params = [p for p in GENOME_PARAMS
                      if p in combined.columns
                      and pd.to_numeric(combined[p], errors="coerce").notna().sum() > 5]
    if numeric_params:
        n_params = len(numeric_params)
        ncols = 3
        nrows = (n_params + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        axes = axes.flatten() if n_params > 1 else [axes]
        for i, p in enumerate(numeric_params):
            ax = axes[i]
            for name, df in all_dfs.items():
                vals = pd.to_numeric(df[p], errors="coerce").dropna()
                if len(vals) > 0:
                    ax.hist(vals, bins=20, alpha=0.5, label=name)
            ax.set_title(p, fontsize=10)
            ax.legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Parameter Distributions by Node", fontsize=13)
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "param_distributions.png", dpi=150)
        plt.close()
        print(f"  Saved: {ANALYSIS_DIR}/param_distributions.png")

    # 3. Parameter vs Fitness scatter
    if len(combined) >= 10:
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        axes = axes.flatten() if n_params > 1 else [axes]
        fitness = pd.to_numeric(combined["fitness"], errors="coerce")
        for i, p in enumerate(numeric_params):
            ax = axes[i]
            vals = pd.to_numeric(combined[p], errors="coerce")
            valid = fitness.notna() & vals.notna()
            for name, df in all_dfs.items():
                mask = combined["node"] == name
                v = vals[mask & valid]
                f = fitness[mask & valid]
                if len(v) > 0:
                    ax.scatter(v, f, alpha=0.4, s=15, label=name)
            corr = fitness[valid].corr(vals[valid]) if valid.sum() > 3 else 0
            ax.set_title(f"{p} (r={corr:.3f})", fontsize=10)
            ax.set_ylabel("fitness")
            ax.legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Parameter vs Fitness (lower=better)", fontsize=13)
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "param_vs_fitness.png", dpi=150)
        plt.close()
        print(f"  Saved: {ANALYSIS_DIR}/param_vs_fitness.png")

    # 4. Diversity heatmap: generation × param uniqueness
    if "generation" in combined.columns and len(combined) >= 20:
        fig, ax = plt.subplots(figsize=(12, 5))
        gens = sorted(combined["generation"].unique())
        diversity_data = []
        for gen in gens:
            gen_df = combined[combined["generation"] == gen]
            row = {}
            for p in numeric_params:
                vals = pd.to_numeric(gen_df[p], errors="coerce").dropna()
                row[p] = vals.nunique() / max(len(vals), 1)
            diversity_data.append(row)
        div_df = pd.DataFrame(diversity_data, index=gens)
        im = ax.imshow(div_df.T, aspect="auto", cmap="YlOrRd",
                        extent=[gens[0], gens[-1], 0, len(numeric_params)])
        ax.set_yticks(range(len(numeric_params)))
        ax.set_yticklabels(numeric_params, fontsize=8)
        ax.set_xlabel("Generation")
        ax.set_title("Parameter Diversity per Generation (brighter = more unique values)")
        plt.colorbar(im, ax=ax, label="Fraction unique")
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "diversity_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved: {ANALYSIS_DIR}/diversity_heatmap.png")

    print(f"\n  All plots saved to {ANALYSIS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze NEAT candidate history")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch CSVs from dragon/gamma via scp before analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("  NEAT CANDIDATE HISTORY ANALYSIS")
    print("=" * 70)

    print("\nLoading data...")
    all_dfs = load_all(fetch=args.fetch)

    if not all_dfs:
        print("\nNo data available yet. Wait for candidates to finish training.")
        sys.exit(1)

    total = sum(len(df) for df in all_dfs.values())
    print(f"\nTotal candidates across all nodes: {total}")

    for name, df in all_dfs.items():
        analyze_diversity(df, name)
        analyze_fitness(df, name)
        analyze_mutations(df, name)
        analyze_param_impact(df, name)

    if len(all_dfs) > 1:
        cross_node_analysis(all_dfs)

    # Combined analysis
    if total >= 5:
        combined = pd.concat(all_dfs.values(), ignore_index=True)
        analyze_diversity(combined, "ALL NODES COMBINED")
        analyze_mutations(combined, "ALL NODES COMBINED")
        analyze_param_impact(combined, "ALL NODES COMBINED")

    print("\n\nGenerating plots...")
    save_plots(all_dfs)

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE — {total} candidates analyzed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
