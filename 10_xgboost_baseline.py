#!/usr/bin/env python
# coding: utf-8

# # XGBoost Baseline (Direct Multi-Output: 48 separate models, one per horizon step)
# Tree-based ML baseline for comparison against Transformer-based deep learning
# models AND against LightGBM (00_lightgbm_baseline.py) - same feature set,
# same windowing, same evaluation, so the only variable between the two tree
# baselines is the boosting library itself.

import os
import sys
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

try:
    import torch
    xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    xgb_device = 'cpu'

try:
    from tqdm.auto import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable, *args, **kwargs):
            return iterable

# ---------------------------------------------------------------
# Data Loading & Preprocessing (same path auto-detect and split as other scripts)
# ---------------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # safety: enforce chronological order before time-based split
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = []
for col in df.columns:
    df[col] = df[col].astype('float32')
    if col != 'kWhDelivered':
        cols.append(col)

X = df[cols]
y = df['kWhDelivered']

print(f"Dataset Loaded successfully from {data_path}! Total Rows: {len(df)}, Features Count: {len(cols)}")

# Train/Val/Test Split (60% / 20% / 20%) - same split as other model scripts
train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]
X_test  = X[train_len + val_len :]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

# Feature Scaling (MinMaxScaler) - not strictly needed for tree models, kept for
# consistency with the LightGBM baseline (same feature space for fair comparison)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# Target NOT scaled - same rationale as the LightGBM baseline: tree models don't
# need target scaling, and keeping raw kW units avoids an extra inverse_transform
y_train_raw = y_train.values
y_val_raw   = y_val.values
y_test_raw  = y_test.values

# Compute Peak Load Threshold (Top 20% of TRAIN in actual kW) - same definition as other scripts
peak_threshold_kw = float(np.percentile(y_train, 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ---------------------------------------------------------------
# Config (mirrors LightGBM baseline)
# ---------------------------------------------------------------
LOOKBACK = 96      # 48 hours history (96 * 30 min)
HORIZON = 48       # 24 hours forecast (48 * 30 min)
output_json_filename = "10_xgboost_baseline_results.json"
import time
results_data = {
    "model_name": "10_xgboost_baseline",
    "total_parameters": HORIZON * 1000,
    "seeds": {},
    "summary": {}
}
all_seed_metrics = []
all_predictions = {}
best_overall_val_loss = float("inf")
best_seed_id = None
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

# ---------------------------------------------------------------
# Windowing: flatten [lookback, num_features] into a single feature vector per
# sample (tree models take tabular input, not sequences) - identical to LightGBM baseline
# ---------------------------------------------------------------
def create_windowed_tabular(X_data, y_data, lookback, horizon):
    """
    X_data: [N, num_features] scaled feature matrix
    y_data: [N] raw target array
    Returns:
        X_flat: [num_samples, lookback * num_features]  (flattened lookback window per sample)
        y_multi: [num_samples, horizon]                  (target for each of the horizon steps)
    """
    num_features = X_data.shape[1]
    num_samples = len(X_data) - lookback - horizon + 1
    X_flat = np.zeros((num_samples, lookback * num_features), dtype=np.float32)
    y_multi = np.zeros((num_samples, horizon), dtype=np.float32)
    for i in range(num_samples):
        X_flat[i] = X_data[i : i + lookback].reshape(-1)
        y_multi[i] = y_data[i + lookback : i + lookback + horizon]
    return X_flat, y_multi

print("Pre-building windowed tabular features...")
X_train_flat, y_train_multi = create_windowed_tabular(X_train_scaled, y_train_raw, LOOKBACK, HORIZON)
X_val_flat,   y_val_multi   = create_windowed_tabular(X_val_scaled,   y_val_raw,   LOOKBACK, HORIZON)
X_test_flat,  y_test_multi  = create_windowed_tabular(X_test_scaled,  y_test_raw,  LOOKBACK, HORIZON)

print(f"Train samples: {X_train_flat.shape}, Val samples: {X_val_flat.shape}, Test samples: {X_test_flat.shape}")

# ---------------------------------------------------------------
# Metrics function (identical definition to other model scripts, for fair comparison)
# ---------------------------------------------------------------
def compute_metrics(actual, predicted, peak_threshold):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    wape = (np.sum(np.abs(actual - predicted)) / np.sum(actual)) * 100

    non_zero_mask = actual > 0
    mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100 if non_zero_mask.any() else np.nan

    peak_mask = actual >= peak_threshold
    if peak_mask.any():
        mae_peak = mean_absolute_error(actual[peak_mask], predicted[peak_mask])
        wape_peak = (np.sum(np.abs(actual[peak_mask] - predicted[peak_mask])) / np.sum(actual[peak_mask])) * 100
    else:
        mae_peak, wape_peak = np.nan, np.nan

    bias = float(np.mean(predicted - actual))
    negative_pct = float(np.mean(predicted < 0) * 100)

    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape, mae_peak=mae_peak, wape_peak=wape_peak, bias=bias, negative_pct=negative_pct)

# ---------------------------------------------------------------
# Direct Multi-Output: train ONE XGBoost model PER horizon step (48 models total)
# Hyperparameters chosen to roughly mirror the LightGBM baseline's settings
# (n_estimators, learning_rate, subsampling) for a fair boosting-library comparison.
# ---------------------------------------------------------------
XGB_PARAMS = dict(
    objective='reg:squarederror',
    eval_metric='mae',
    n_estimators=1000,
    learning_rate=0.05808381432960484,
    max_depth=4,            # XGBoost uses max_depth rather than LightGBM's num_leaves
    min_child_weight=7.096220622166763,
    subsample=0.6648958679659673,
    colsample_bytree=0.5617104625598865,
    reg_alpha=0.0008553664409816912,
    reg_lambda=0.27861143653900455,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',     # fast histogram-based method, comparable to LightGBM's default
    device=xgb_device,
    verbosity=0,
)
EARLY_STOPPING_ROUNDS = 50

SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]

print(f"Starting Direct Multi-Output XGBoost ({HORIZON} models per seed) for {len(SEEDS)} seeds...")

for seed_idx, SEED in enumerate(SEEDS, 1):
    seed_start_time = time.time()
    print(f"\n=========================================================================")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"=========================================================================")

    seed_params = dict(XGB_PARAMS)
    seed_params['random_state'] = SEED

    y_pred_test = np.zeros((X_test_flat.shape[0], HORIZON), dtype=np.float64)

    step_pbar = tqdm(range(HORIZON), desc=f"Seed {SEED} - Training per-step models", leave=True)
    t0 = time.time()
    for step in step_pbar:
        model = xgb.XGBRegressor(**seed_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train_flat, y_train_multi[:, step],
            eval_set=[(X_val_flat, y_val_multi[:, step])],
            verbose=False,
        )
        y_pred_test[:, step] = model.predict(X_test_flat, iteration_range=(0, model.best_iteration + 1))
        step_pbar.set_postfix({'step': f"{step+1}/{HORIZON}"})

    elapsed = time.time() - t0
    print(f"Trained {HORIZON} models in {elapsed:.1f}s ({elapsed / HORIZON:.2f}s/step)")

    # Predictions and actuals are already in raw kW scale (no inverse_transform needed,
    # since target was not scaled for XGBoost)
    actual_by_step = {step: y_test_multi[:, step] for step in steps_to_eval}
    predictions_by_step = {step: y_pred_test[:, step] for step in steps_to_eval}

    # Build Markdown evaluation output text (same format as other model scripts)
    output_lines = []
    output_lines.append(f"## SEED {SEED}")
    output_lines.append("================ MODEL EVALUATION METRICS (per horizon step) ================")
    output_lines.append(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")
    output_lines.append("-------------------------------------------------------------------------------")

    for step in steps_to_eval:
        m = compute_metrics(actual_by_step[step], predictions_by_step[step], peak_threshold_kw)
        output_lines.append(f"\n[{step_labels[step]}]")
        output_lines.append(f"  Overall MAE   : {m['mae']:.4f} kW")
        output_lines.append(f"  Overall RMSE  : {m['rmse']:.4f} kW")
        output_lines.append(f"  Overall R²    : {m['r2']:.4f}")
        output_lines.append(f"  Overall MAPE  : {m['mape']:.2f}%")
        output_lines.append(f"  Overall WAPE  : {m['wape']:.2f}%")
        output_lines.append(f"  Peak Zone MAE : {m['mae_peak']:.4f} kW")
        output_lines.append(f"  Peak Zone WAPE: {m['wape_peak']:.2f}%")

    output_lines.append("=================================================================================\n")
    full_output_text = "\n".join(output_lines)
    print(full_output_text)

    overall_metrics = compute_metrics(y_test_multi.reshape(-1), y_pred_test.reshape(-1), peak_threshold_kw)
    seed_duration = round(time.time() - seed_start_time, 2)
    overall_metrics["training_time_seconds"] = seed_duration
    all_seed_metrics.append(overall_metrics)

    per_step_metrics = {}
    for step in steps_to_eval:
        m = compute_metrics(actual_by_step[step], predictions_by_step[step], peak_threshold_kw)
        per_step_metrics[step_labels[step]] = {k: (float(v) if not np.isnan(v) else None) for k, v in m.items()}

    # 48-step full horizon evaluation (24-hour error degradation)
    mae_48 = [float(mean_absolute_error(y_test_multi[:, s], y_pred_test[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_test_multi[:, s], y_pred_test[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_test.astype(np.float32)

    results_data["seeds"][str(SEED)] = {
        "training_time_seconds": seed_duration,
        "overall_metrics": {k: (float(v) if not np.isnan(v) else None) for k, v in overall_metrics.items()},
        "per_step_metrics": per_step_metrics,
        "step_48_metrics": {
            "mae": mae_48,
            "rmse": rmse_48
        }
    }
    with open(output_json_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Successfully saved SEED {SEED} results to {output_json_filename} (Runtime: {seed_duration}s)")
    gc.collect()

all_predictions["y_true"] = y_test_multi.astype(np.float32)
pred_stack = np.stack([all_predictions[f"seed_{s}"] for s in SEEDS], axis=0)
all_predictions["pred_mean"] = np.mean(pred_stack, axis=0).astype(np.float32)
all_predictions["pred_std"] = np.std(pred_stack, axis=0).astype(np.float32)
np.savez_compressed(f"10_xgboost_baseline_predictions.npz", **all_predictions)
print(f"Successfully saved all seed predictions to 10_xgboost_baseline_predictions.npz")

print(f"\n======================================================================")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — 10_xgboost_baseline")
print(f"======================================================================")
summary_dict = {}
metric_keys = ['mae', 'rmse', 'r2', 'wape', 'mape', 'bias', 'negative_pct', 'training_time_seconds']
for k in metric_keys:
    vals = [m[k] for m in all_seed_metrics if k in m and not np.isnan(m[k])]
    if vals:
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        print(f"  {k.upper():<22}: {mu:.4f} ± {sigma:.4f}")
        summary_dict[k] = {"mean": mu, "std": sigma}

all_mae_48 = [results_data["seeds"][str(s)]["step_48_metrics"]["mae"] for s in results_data["seeds"] if "step_48_metrics" in results_data["seeds"][str(s)]]
if all_mae_48:
    summary_dict["mean_mae_by_step_48"] = [float(v) for v in np.mean(all_mae_48, axis=0)]

results_data["config"] = {
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "batch_size": BATCH_SIZE if 'BATCH_SIZE' in globals() or 'BATCH_SIZE' in locals() else None,
    "seeds": SEEDS if 'SEEDS' in globals() or 'SEEDS' in locals() else None,
    "total_parameters": results_data.get("total_parameters", None)
}
results_data["summary"] = summary_dict
with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
print(f"Successfully saved final results to {output_json_filename}")
print(f"\nFinished running all {len(SEEDS)} SEEDs for 10_xgboost_baseline!")
