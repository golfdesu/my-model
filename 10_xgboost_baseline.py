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
data_path = 'acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = 'acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = '../preprocess/acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = '../preprocess/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = r'C:\Users\chaya\Documents\Program\Practice\preprocess\acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = r'C:\Users\chaya\Documents\Program\Practice\preprocess\acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
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
output_md_filename = "10_xgboost_baseline.md"
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

    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape, mae_peak=mae_peak, wape_peak=wape_peak)

# ---------------------------------------------------------------
# Direct Multi-Output: train ONE XGBoost model PER horizon step (48 models total)
# Hyperparameters chosen to roughly mirror the LightGBM baseline's settings
# (n_estimators, learning_rate, subsampling) for a fair boosting-library comparison.
# ---------------------------------------------------------------
XGB_PARAMS = dict(
    objective='reg:squarederror',
    eval_metric='mae',
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,            # XGBoost uses max_depth rather than LightGBM's num_leaves
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',     # fast histogram-based method, comparable to LightGBM's default
    verbosity=0,
)
EARLY_STOPPING_ROUNDS = 50

SEEDS = [164, 256, 355, 1234, 2026]

print(f"Starting Direct Multi-Output XGBoost ({HORIZON} models per seed) for {len(SEEDS)} seeds...")

for seed_idx, SEED in enumerate(SEEDS, 1):
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

    with open(output_md_filename, "a", encoding="utf-8") as f:
        f.write(full_output_text + "\n")

    print(f"Successfully saved SEED {SEED} metrics to {output_md_filename}")
    gc.collect()

print(f"\nFinished running all {len(SEEDS)} SEEDs for XGBoost Direct Multi-Output baseline!")
