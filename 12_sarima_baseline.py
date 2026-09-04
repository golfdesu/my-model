#!/usr/bin/env python
# coding: utf-8

# # SARIMA Baseline (Walk-Forward, Full Refit per Window)
# Statistical baseline for comparison against Transformer-based models.
# Univariate: uses only kWhDelivered history (no exogenous features), 
# matching the classic ARIMA/SARIMA setup.

import os
import sys
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# Safe tqdm import fallback
try:
    from tqdm.auto import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable, *args, **kwargs):
            return iterable

# ---------------------------------------------------------------
# Data Loading & Preprocessing (same path auto-detect as other scripts)
# ---------------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # safety: enforce chronological order before time-based split

y = df['kWhDelivered'].astype('float32')

print(f"Dataset Loaded successfully from {data_path}! Total Rows: {len(df)}")

# Train/Val/Test Split (60% / 20% / 20%) - same split as other model scripts
train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

# SARIMA baseline: train on TRAIN+VAL history (no gradient-based hyperparameter
# search here, so no need to hold out a separate validation set for early stopping)
y_history_full = pd.concat([y_train, y_val])

# Compute Peak Load Threshold (Top 20% of TRAIN in actual kW) - same definition as other scripts
peak_threshold_kw = float(np.percentile(y_train, 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ---------------------------------------------------------------
# Config (mirrors other model scripts)
# ---------------------------------------------------------------
LOOKBACK = 96     # not directly used by SARIMA (uses full history instead) but kept for reference
HORIZON = 48      # 24 hours forecast (48 * 30 min)
SEASONAL_PERIOD = 48   # 1 day = 48 half-hour steps (daily seasonality)

# SARIMA order - a reasonable default; can be tuned further with grid search / auto_arima if needed
ORDER = (0, 1, 1)
SEASONAL_ORDER = (1, 0, 1, SEASONAL_PERIOD)

output_json_filename = "12_sarima_baseline_results.json"
results_data = {
    "model_name": "12_sarima_baseline",
    "total_parameters": ORDER[0] + ORDER[2] + SEASONAL_ORDER[0] + SEASONAL_ORDER[2] + 1,
    "overall_metrics": {},
    "per_step_metrics": {}
}
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

# WARNING: Walk-forward with full refit at every window is very slow for SARIMA
# (each fit is O(n) or worse depending on n and order). Limit the number of test
# windows to keep runtime manageable; adjust MAX_TEST_WINDOWS as needed.
MAX_TEST_WINDOWS = 200  # set to None to run on the FULL test set (can take a very long time)
# NOTE on runtime: each SARIMA refit takes roughly 20-90+ seconds depending on history
# length and dataset size (measured ~60s/window on a small smoke-test dataset). A full
# walk-forward run over thousands of test windows can take many hours. Start with a
# small MAX_TEST_WINDOWS to estimate per-window time on YOUR dataset before committing
# to a full run, then scale up (or switch to the periodic-refit / single-fit variants
# discussed earlier) if the full walk-forward is too slow in practice.

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
# Walk-Forward SARIMA: refit at every test window, forecast HORIZON steps ahead
# ---------------------------------------------------------------
y_test_values = y_test.values
n_test_windows = len(y_test_values) - HORIZON + 1
if n_test_windows <= 0:
    raise ValueError("Test set too short for the configured HORIZON.")

if MAX_TEST_WINDOWS is not None:
    n_test_windows = min(n_test_windows, MAX_TEST_WINDOWS)

print(f"Starting Walk-Forward SARIMA{ORDER}x{SEASONAL_ORDER} over {n_test_windows} test windows...")
print("(Each window refits the model on all history up to that point - this may take a while.)")

# Full history array we'll extend as we walk forward through the test set
history = y_history_full.values.copy()

y_pred_all = np.zeros((n_test_windows, HORIZON), dtype=np.float64)
y_true_all = np.zeros((n_test_windows, HORIZON), dtype=np.float64)

t0 = time.time()
window_pbar = tqdm(range(n_test_windows), desc="SARIMA Walk-Forward", leave=True)
for w in window_pbar:
    # Fit SARIMA on all data observed so far (history up to this point)
    model = SARIMAX(
        history,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    # Forecast HORIZON steps ahead
    forecast = fitted.forecast(steps=HORIZON)
    y_pred_all[w, :] = forecast

    # Ground truth for this window
    y_true_all[w, :] = y_test_values[w : w + HORIZON]

    # Advance history by one step (walk-forward): append the next actual test observation
    history = np.append(history, y_test_values[w])

    if hasattr(window_pbar, 'set_postfix'):
        window_pbar.set_postfix({'window': f"{w+1}/{n_test_windows}"})

elapsed = time.time() - t0
print(f"\nWalk-forward SARIMA completed {n_test_windows} windows in {elapsed:.1f}s "
      f"({elapsed / n_test_windows:.2f}s/window)")

# ---------------------------------------------------------------
# Evaluation (same format as other model scripts)
# ---------------------------------------------------------------
actual_by_step = {step: y_true_all[:, step] for step in steps_to_eval}
predictions_by_step = {step: y_pred_all[:, step] for step in steps_to_eval}

output_lines = []
output_lines.append("## SARIMA Walk-Forward Baseline")
output_lines.append(f"Order={ORDER}, Seasonal Order={SEASONAL_ORDER}, Windows Evaluated={n_test_windows}")
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

overall_metrics = compute_metrics(y_true_all.reshape(-1), y_pred_all.reshape(-1), peak_threshold_kw)
overall_metrics["training_time_seconds"] = round(elapsed, 2)

per_step_metrics = {}
for step in steps_to_eval:
    m = compute_metrics(actual_by_step[step], predictions_by_step[step], peak_threshold_kw)
    per_step_metrics[step_labels[step]] = {k: (float(v) if not np.isnan(v) else None) for k, v in m.items()}

# 48-step full horizon evaluation (24-hour error degradation)
mae_48 = [float(mean_absolute_error(y_true_all[:, s], y_pred_all[:, s])) for s in range(HORIZON)]
rmse_48 = [float(np.sqrt(mean_squared_error(y_true_all[:, s], y_pred_all[:, s]))) for s in range(HORIZON)]

results_data["training_time_seconds"] = round(elapsed, 2)
results_data["overall_metrics"] = {k: (float(v) if not np.isnan(v) else None) for k, v in overall_metrics.items()}
results_data["per_step_metrics"] = per_step_metrics
results_data["step_48_metrics"] = {
    "mae": mae_48,
    "rmse": rmse_48
}
results_data["config"] = {
    "lookback": LOOKBACK if 'LOOKBACK' in globals() else 96,
    "horizon": HORIZON,
    "order": ORDER,
    "seasonal_order": SEASONAL_ORDER,
    "total_parameters": results_data.get("total_parameters", None)
}
with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
print(f"Successfully saved SARIMA baseline results to {output_json_filename}")

# Also save raw predictions for potential further analysis
np.savez(
    "00_sarima_baseline_predictions.npz",
    y_pred=y_pred_all,
    y_true=y_true_all,
)
print("Saved raw predictions to 00_sarima_baseline_predictions.npz")

np.savez_compressed(
    f"12_sarima_baseline_predictions.npz",
    y_true=y_true_all.astype(np.float32),
    pred_mean=y_pred_all.astype(np.float32)
)
print(f"Successfully saved SARIMA predictions to 12_sarima_baseline_predictions.npz")