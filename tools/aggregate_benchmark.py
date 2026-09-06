#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# aggregate_benchmark.py — Publication Benchmark Aggregator & Visualizer
# Reads all multi-seed benchmark results (*_results.json) across models 00–26,
# computes rigorous cross-seed statistics (Mean ± Std), formats comparison tables
# (Terminal, CSV, Markdown, LaTeX), and generates publication-grade figures.
#
# Usage:
#   python tools/aggregate_benchmark.py
#   python tools/aggregate_benchmark.py --sort_by rmse --plots
#   python tools/aggregate_benchmark.py --latex --out_dir docs
# ==============================================================================

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Headless matplotlib for Linux / HPC compatibility
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Model Family Taxonomy
FAMILY_MAP = {
    "00_tfm_custom": ("Custom Proposed", "#1f77b4"),
    "01_tfm_enc": ("Transformer", "#ff7f0e"),
    "02_tfm_dec": ("Transformer", "#ff7f0e"),
    "03_tfm_encdec": ("Transformer", "#ff7f0e"),
    "04_tfm_ifm": ("Transformer", "#ff7f0e"),
    "05_tfm_afm": ("Transformer", "#ff7f0e"),
    "06_tfm_ptst": ("Transformer", "#ff7f0e"),
    "07_tfm_itfm": ("Transformer", "#ff7f0e"),
    "08_tfm_timesnet": ("CNN / 2D Temporal", "#2ca02c"),
    "09_lstm": ("Recurrent (RNN)", "#d62728"),
    "10_gru": ("Recurrent (RNN)", "#d62728"),
    "11_dlinear": ("Linear & Decomp", "#9467bd"),
    "12_nlinear": ("Linear & Decomp", "#9467bd"),
    "13_smamba": ("State Space Model", "#8c564b"),
    "14_powermamba": ("State Space Model", "#8c564b"),
    "15_timemachine": ("State Space Model", "#8c564b"),
    "16_s4d": ("State Space Model", "#8c564b"),
    "17_xgboost": ("Tree-based GBDT", "#e377c2"),
    "18_lightgbm": ("Tree-based GBDT", "#e377c2"),
    "19_sarima": ("Statistical Baseline", "#7f7f7f"),
    "20_tfm_mft": ("Transformer (MFT)", "#ff7f0e"),
    "21_cnn_lstm_tfm": ("Hybrid Architecture", "#bcbd22"),
    "22_tfm_fedformer": ("Transformer", "#ff7f0e"),
    "23_tcn": ("CNN / 2D Temporal", "#2ca02c"),
    "24_nhits": ("Basis Expansion", "#17becf"),
    "25_tide": ("Linear / Dense MLP", "#9467bd"),
    "26_nbeats": ("Basis Expansion", "#17becf"),
    "27_moderntcn": ("CNN (ModernTCN)", "#2ca02c"),
    "28_crossformer": ("Transformer (Crossformer)", "#ff7f0e"),
    "29_segrnn": ("Recurrent (SegRNN)", "#d62728"),
    "30_nstransformer": ("Transformer (NS-Tfm)", "#ff7f0e"),
    "31_scinet": ("Convolutional (SCINet)", "#2ca02c")
}

def get_family_info(model_name):
    for key, (family, color) in FAMILY_MAP.items():
        if key in model_name:
            return family, color
    return "Other Architecture", "#333333"

def find_result_files(search_paths):
    files = set()
    for base in search_paths:
        # Check outputs/*/*_results.json
        files.update(glob.glob(os.path.join(base, "outputs", "*", "*_results.json")))
        # Check root *_results.json
        files.update(glob.glob(os.path.join(base, "*_results.json")))
        # Check outputs/*.json
        files.update(glob.glob(os.path.join(base, "outputs", "*_results.json")))

    # Deduplicate by model_name
    models_dict = {}
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            m_name = data.get("model_name", os.path.basename(f).replace("_results.json", ""))
            # Prefer files inside outputs/ if duplicated
            if m_name not in models_dict or "outputs" in f:
                models_dict[m_name] = (f, data)
        except Exception:
            continue
    return models_dict

def extract_model_summary(model_name, filepath, data):
    summary = data.get("summary", {})
    seeds = data.get("seeds", {})
    config = data.get("config", {})
    params_count = data.get("total_parameters", config.get("total_parameters", None))

    family, color = get_family_info(model_name)

    # Metric extraction with fallback to seeds average
    metrics_to_read = ['mae', 'rmse', 'r2', 'wape', 'mape', 'mae_peak', 'wape_peak', 'bias', 'negative_pct', 'training_time_seconds', 'peak_gpu_memory_mb']
    row = {
        "model_name": model_name,
        "family": family,
        "color": color,
        "total_parameters": params_count,
        "num_seeds": len(seeds),
        "filepath": filepath
    }

    for m in metrics_to_read:
        if m in summary and isinstance(summary[m], dict) and "mean" in summary[m]:
            row[f"{m}_mean"] = float(summary[m]["mean"])
            row[f"{m}_std"]  = float(summary[m].get("std", 0.0))
        elif seeds:
            vals = []
            for s, sdata in seeds.items():
                omet = sdata.get("overall_metrics", {})
                if m in omet and omet[m] is not None and not np.isnan(omet[m]):
                    vals.append(omet[m])
                elif m in sdata and sdata[m] is not None and not np.isnan(sdata[m]):
                    vals.append(sdata[m])
            if vals:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"]  = float(np.std(vals))
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"]  = np.nan
        else:
            row[f"{m}_mean"] = np.nan
            row[f"{m}_std"]  = np.nan

    # Extract horizon step metrics
    step_48 = summary.get("mean_mae_by_step_48", None)
    if step_48 is None and seeds:
        all_s48 = []
        for s, sdata in seeds.items():
            s48 = sdata.get("step_48_metrics", {}).get("mae", None)
            if s48 and len(s48) == 48:
                all_s48.append(s48)
        if all_s48:
            step_48 = [float(v) for v in np.mean(all_s48, axis=0)]
    row["step_48_mae"] = step_48

    return row

def generate_markdown_table(df, sort_by="mae_mean"):
    df_sorted = df.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    best_mae = df_sorted["mae_mean"].min()
    best_rmse = df_sorted["rmse_mean"].min()
    best_wape = df_sorted["wape_mean"].min()
    best_peak_mae = df_sorted["mae_peak_mean"].min()
    best_r2 = df_sorted["r2_mean"].max()

    lines = [
        "# Comprehensive Benchmark Results (10-Seed Average)",
        "",
        "> Dataset: `acn_caltech_ready2.csv` (Lookback $L=96$, Horizon $H=48$)",
        f"> Sorted by: `{sort_by}` (Lowest is Best)",
        "",
        "| Rank | Model Architecture | Family | MAE (kWh) ↓ | RMSE (kWh) ↓ | WAPE (%) ↓ | Peak MAE ↓ | R² ↑ | Params | Time (s) |",
        "|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    for rank, r in df_sorted.iterrows():
        mae_str = f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}"
        if r['mae_mean'] == best_mae:
            mae_str = f"**{mae_str}** 🏆"

        rmse_str = f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
        if r['rmse_mean'] == best_rmse:
            rmse_str = f"**{rmse_str}**"

        wape_str = f"{r['wape_mean']:.2f}% ± {r['wape_std']:.2f}%"
        if r['wape_mean'] == best_wape:
            wape_str = f"**{wape_str}**"

        peak_str = f"{r['mae_peak_mean']:.4f}" if not np.isnan(r['mae_peak_mean']) else "N/A"
        if not np.isnan(r['mae_peak_mean']) and r['mae_peak_mean'] == best_peak_mae:
            peak_str = f"**{peak_str}**"

        r2_str = f"{r['r2_mean']:.4f}" if not np.isnan(r['r2_mean']) else "N/A"
        if not np.isnan(r['r2_mean']) and r['r2_mean'] == best_r2:
            r2_str = f"**{r2_str}**"

        params_str = f"{int(r['total_parameters']):,}" if pd.notna(r['total_parameters']) and r['total_parameters'] else "N/A"
        time_str = f"{r['training_time_seconds_mean']:.1f}s" if pd.notna(r['training_time_seconds_mean']) else "N/A"

        lines.append(f"| {rank + 1} | `{r['model_name']}` | {r['family']} | {mae_str} | {rmse_str} | {wape_str} | {peak_str} | {r2_str} | {params_str} | {time_str} |")

    return "\n".join(lines)

def generate_latex_table(df, sort_by="mae_mean"):
    df_sorted = df.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    best_mae = df_sorted["mae_mean"].min()
    best_rmse = df_sorted["rmse_mean"].min()
    best_wape = df_sorted["wape_mean"].min()

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Multi-seed benchmark results across architectures on Caltech ACN aggregate station load ($L=96, H=48$, 10 seeds). Mean and standard deviation are reported. Bold indicates best performance.}",
        r"\label{tab:benchmark_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l c c c c c}",
        r"\toprule",
        r"\textbf{Model Architecture} & \textbf{Family} & \textbf{MAE (kWh)} $\downarrow$ & \textbf{RMSE (kWh)} $\downarrow$ & \textbf{WAPE (\%)} $\downarrow$ & \textbf{$R^2$} $\uparrow$ & \textbf{Parameters} \\",
        r"\midrule"
    ]

    for _, r in df_sorted.iterrows():
        clean_name = r['model_name'].replace("_", r"\_")
        mae_val = f"${r['mae_mean']:.4f} \\pm {r['mae_std']:.4f}$"
        if r['mae_mean'] == best_mae:
            mae_val = f"\\textbf{{{mae_val}}}"

        rmse_val = f"${r['rmse_mean']:.4f} \\pm {r['rmse_std']:.4f}$"
        if r['rmse_mean'] == best_rmse:
            rmse_val = f"\\textbf{{{rmse_val}}}"

        wape_val = f"${r['wape_mean']:.2f} \\pm {r['wape_std']:.2f}$"
        if r['wape_mean'] == best_wape:
            wape_val = f"\\textbf{{{wape_val}}}"

        r2_val = f"${r['r2_mean']:.4f}$" if pd.notna(r['r2_mean']) else "N/A"
        params_val = f"{int(r['total_parameters']):,}" if pd.notna(r['total_parameters']) and r['total_parameters'] else "N/A"

        tex.append(f"{clean_name} & {r['family']} & {mae_val} & {rmse_val} & {wape_val} & {r2_val} & {params_val} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])
    return "\n".join(tex)

def generate_plots(df, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    df_sorted = df.dropna(subset=["mae_mean"]).sort_values(by="mae_mean", ascending=True)

    # 1. Bar Chart: MAE Ranking with Error Bars
    plt.figure(figsize=(14, 7), dpi=300)
    bars = plt.barh(
        range(len(df_sorted)),
        df_sorted["mae_mean"],
        xerr=df_sorted["mae_std"],
        color=df_sorted["color"],
        alpha=0.85,
        capsize=4,
        edgecolor="black",
        linewidth=0.7
    )
    plt.yticks(range(len(df_sorted)), df_sorted["model_name"], fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Absolute Error (MAE) [kWh] (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Benchmarking Aggregate EV Load Forecasting Architectures (10 Seeds Mean ± Std)", fontsize=14, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate bar values
    for idx, (_, r) in enumerate(df_sorted.iterrows()):
        val_str = f" {r['mae_mean']:.4f}"
        plt.text(r['mae_mean'] + (r['mae_std'] if not np.isnan(r['mae_std']) else 0), idx, val_str, va="center", fontsize=8)

    plt.tight_layout()
    bar_path = os.path.join(out_dir, "benchmark_mae_ranking.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved: {bar_path}")

    # 2. Horizon Degradation Curves (Top Models)
    top_models = df_sorted[df_sorted["step_48_mae"].notna()].head(8)
    if len(top_models) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        steps = np.arange(1, 49)
        for _, r in top_models.iterrows():
            plt.plot(steps, r["step_48_mae"], label=r["model_name"], linewidth=2.0, color=r["color"])

        plt.xlabel("Forecast Horizon Step (1 to 48, 30-min intervals = 24 Hours)", fontsize=12, fontweight="bold")
        plt.ylabel("Mean Absolute Error (MAE) [kWh]", fontsize=12, fontweight="bold")
        plt.title("Forecast Horizon Degradation Curve across Horizon Steps (Top Architectures)", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        horizon_path = os.path.join(out_dir, "benchmark_horizon_mae.png")
        plt.savefig(horizon_path)
        plt.close()
        print(f"Saved: {horizon_path}")

    # 3. Pareto Efficiency: MAE vs Trainable Parameters
    valid_params = df_sorted[df_sorted["total_parameters"].notna() & (df_sorted["total_parameters"] > 0)]
    if len(valid_params) > 0:
        plt.figure(figsize=(10, 6), dpi=300)
        scatter = plt.scatter(
            valid_params["total_parameters"],
            valid_params["mae_mean"],
            c=valid_params["color"],
            s=120,
            edgecolor="black",
            alpha=0.85,
            zorder=3
        )
        for _, r in valid_params.iterrows():
            plt.annotate(
                r["model_name"],
                (r["total_parameters"], r["mae_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.9
            )
        plt.xscale("log")
        plt.xlabel("Trainable Parameters (Log Scale)", fontsize=12, fontweight="bold")
        plt.ylabel("Test MAE (kWh) (Lower is Better)", fontsize=12, fontweight="bold")
        plt.title("Pareto Frontier: Forecast Accuracy vs. Model Complexity", fontsize=14, fontweight="bold")
        plt.grid(True, which="both", linestyle=":", alpha=0.6)
        plt.tight_layout()
        pareto_path = os.path.join(out_dir, "benchmark_pareto_efficiency.png")
        plt.savefig(pareto_path)
        plt.close()
        print(f"Saved: {pareto_path}")

def main():
    parser = argparse.ArgumentParser(description="Publication Benchmark Aggregator")
    parser.add_argument("--search_dirs", nargs="+", default=[".", "..", "outputs"], help="Directories to search for _results.json")
    parser.add_argument("--out_dir", default="outputs", help="Output directory for reports")
    parser.add_argument("--sort_by", default="mae_mean", help="Metric to sort by")
    parser.add_argument("--latex", action="store_true", default=True, help="Generate LaTeX table")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate benchmark comparison plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=" * 75)
    print("[Aggregate Benchmark Results Engine]")
    print("=" * 75)

    result_files = find_result_files(args.search_dirs)
    if not result_files:
        print("No *_results.json files found in search paths!")
        return

    print(f"Discovered {len(result_files)} unique model benchmark results.")

    rows = []
    for model_name, (fpath, data) in result_files.items():
        row = extract_model_summary(model_name, fpath, data)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(args.out_dir, "benchmark_summary.csv")
    export_cols = [c for c in df.columns if c not in ["step_48_mae", "color"]]
    df[export_cols].sort_values(by=args.sort_by, ascending=True).to_csv(csv_path, index=False)
    print(f"[Saved] CSV Summary: {csv_path}")

    # Generate Markdown Table
    md_content = generate_markdown_table(df, sort_by=args.sort_by)
    md_path = os.path.join("docs", "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(md_content)
    print(f"[Saved] Markdown Table: {md_path}")

    # Generate LaTeX Table
    if args.latex:
        tex_content = generate_latex_table(df, sort_by=args.sort_by)
        tex_path = os.path.join("docs", "benchmark_summary.tex")
        with open(tex_path, "w", encoding="utf-8") as fp:
            fp.write(tex_content)
        print(f"[Saved] LaTeX Table: {tex_path}")

    # Generate Plots
    if args.plots:
        print("\n[Plotting] Generating publication-grade comparison plots...")
        generate_plots(df, out_dir="plots")

    # Print Terminal Summary
    print("\n" + "=" * 75)
    print(f"[Top Models Ranked by {args.sort_by}]:")
    print("=" * 75)
    df_top = df.dropna(subset=[args.sort_by]).sort_values(by=args.sort_by, ascending=True).head(10)
    for r_idx, (_, r) in enumerate(df_top.iterrows(), 1):
        mae_str = f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}"
        rmse_str = f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
        print(f"  {r_idx:2d}. {r['model_name']:<28} | MAE: {mae_str:<17} | RMSE: {rmse_str:<17} | Family: {r['family']}")
    print("=" * 75)

if __name__ == '__main__':
    main()
