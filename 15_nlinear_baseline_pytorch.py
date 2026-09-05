import json
#!/usr/bin/env python
# coding: utf-8

# NLinear (Zeng et al., AAAI 2023 - "Are Transformers Effective for Time Series Forecasting?")
# NLinear is the NORMALIZED variant of DLinear from the same paper.
# Key innovation: subtracts the last timestep value before linear projection
# (instance-level normalization = subtract sequence tail), then adds it back.
# Y = Linear(X - X[-1]) + X[-1]
# This directly addresses distribution shift / non-stationarity.
# Our dataset has pronounced non-stationarity (mean drifts ~61% over 22 months).
# Compare with 09_dlinear: NLinear should outperform DLinear on this non-stationary series.
# Uses UNIVARIATE input only (target history), matching original paper design.

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from tqdm.auto import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable, *args, **kwargs):
            return iterable

warnings.filterwarnings('ignore')

num_cpus = os.cpu_count() or 4
torch.set_num_threads(min(6, num_cpus))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("PyTorch Version:", torch.__version__)
print("Using Device:", device)

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# 1. Univariate Data Loading & Preprocessing
# ---------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # safety: enforce chronological order before time-based split

# Univariate only: target history alone as input (matching original NLinear/DLinear paper design)
y = df['kWhDelivered'].astype('float32')
print(f"Dataset Loaded from {data_path}! Total Rows: {len(df)}")

train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

peak_threshold_kw = float(np.percentile(y_train, 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------
def create_windowed_tensors_univariate(y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(y_data) - lookback - horizon + 1):
        X_seq.append(y_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))  # [N, lookback]
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))  # [N, horizon]
    return X_t, y_t, np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def compute_metrics(actual, predicted, peak_threshold):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    wape = (np.sum(np.abs(actual - predicted)) / np.sum(actual)) * 100
    non_zero = actual > 0
    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100 if non_zero.any() else np.nan
    peak = actual >= peak_threshold
    if peak.any():
        mae_peak  = mean_absolute_error(actual[peak], predicted[peak])
        wape_peak = (np.sum(np.abs(actual[peak] - predicted[peak])) / np.sum(actual[peak])) * 100
    else:
        mae_peak, wape_peak = np.nan, np.nan

    bias = float(np.mean(predicted - actual))
    negative_pct = float(np.mean(predicted < 0) * 100)

    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape, mae_peak=mae_peak, wape_peak=wape_peak, bias=bias, negative_pct=negative_pct)

# ---------------------------------------------------------
# 3. NLinear Architecture
# ---------------------------------------------------------
class NLinear(nn.Module):
    """
    NLinear (Zeng et al., AAAI 2023)
    Normalize-then-Linear: instance normalization by subtracting the last timestep value.

    Algorithm:
        last  = x[:, -1:]           # last observed value (distribution anchor)
        x_norm = x - last           # subtract to remove local distribution shift
        out   = Linear(x_norm)      # linear mapping of normalized sequence
        out   = out + last          # add back baseline (denormalize)

    This simple trick directly handles non-stationarity by operating on the
    difference from the most recent observation rather than on the raw level.
    """
    def __init__(self, lookback, horizon):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: [batch, lookback]  (univariate)
        last   = x[:, -1:]          # [batch, 1] — last observed value
        x_norm = x - last           # [batch, lookback] — remove distribution shift
        out    = self.linear(x_norm) # [batch, horizon]
        out    = out + last          # [batch, horizon] — add back baseline
        return out

# ---------------------------------------------------------
# 4. Config & Tensor Pre-build
# ---------------------------------------------------------
import time

LOOKBACK   = 96
HORIZON    = 48
BATCH_SIZE = 128
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
output_json_filename = "15_nlinear_baseline_pytorch_results.json"
results_data = {
    "model_name": "15_nlinear_baseline_pytorch",
    "seeds": {},
    "summary": {}
}
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print("Pre-building univariate sequence tensors...")
X_train_t, y_train_t, _, _ = create_windowed_tensors_univariate(y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t,   _, _ = create_windowed_tensors_univariate(y_val_scaled,   LOOKBACK, HORIZON)
X_test_t,  y_test_t,  X_test_seq, y_test_seq = create_windowed_tensors_univariate(y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

print(f"Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")
print(f"Starting {len(SEEDS)}-Seed Loop for NLinear...")

# ---------------------------------------------------------
# 5. Seed Training Loop
# ---------------------------------------------------------
all_seed_metrics = []
all_predictions = {}
best_overall_val_loss = float("inf")
best_seed_id = None
for seed_idx, SEED in enumerate(SEEDS, 1):
    seed_start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\n{'='*70}")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = NLinear(lookback=LOOKBACK, horizon=HORIZON).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results_data["total_parameters"] = total_params

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if seed_idx == 1:
        print(f"Model Parameters: {total_params:,}  (NLinear: minimal by design)")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007436705882390548, weight_decay=6.408391359387875e-05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    epochs           = 200
    patience         = 15
    best_val_loss    = float('inf')
    train_loss_history = []
    val_loss_history   = []
    best_epoch         = 1
    patience_counter = 0
    best_model_weights = None
    t0 = time.time()

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Seed {SEED} Training", leave=True)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bX.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        epoch_pbar.set_postfix({
            'train': f"{train_loss:.5f}", 'val': f"{val_loss:.5f}",
            'best':  f"{best_val_loss:.5f}", 'pat': f"{patience_counter}/{patience}"
        })

    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s | Best Val Loss: {best_val_loss:.6f}")

    model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    model.eval()
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for bX, by in test_loader:
            all_preds.append(model(bX.to(device)).cpu().numpy())
            all_targets.append(by.numpy())

    y_pred_scaled = np.concatenate(all_preds,   axis=0)
    y_true_scaled = np.concatenate(all_targets, axis=0)

    y_pred_kw = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, HORIZON)
    y_true_kw = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1, HORIZON)

    metrics = compute_metrics(y_true_kw.flatten(), y_pred_kw.flatten(), peak_threshold_kw)
    all_seed_metrics.append(metrics)

    print(f"\n--- Seed {SEED} Overall Test Metrics ---")
    print(f"  MAE:  {metrics['mae']:.4f} kWh")
    print(f"  RMSE: {metrics['rmse']:.4f} kWh")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  WAPE: {metrics['wape']:.2f}%")

    per_step_metrics = {}
    for step in steps_to_eval:
        step_metrics = compute_metrics(y_true_kw[:, step], y_pred_kw[:, step], peak_threshold_kw)
        label = step_labels[step]
        per_step_metrics[label] = {k: (float(v) if not np.isnan(v) else None) for k, v in step_metrics.items()}
        print(f"  [{label}] MAE: {step_metrics['mae']:.4f}, RMSE: {step_metrics['rmse']:.4f}, WAPE: {step_metrics['wape']:.2f}%")

    seed_duration = round(time.time() - seed_start_time, 2)
    metrics["training_time_seconds"] = seed_duration

    # 48-step full horizon evaluation (24-hour error degradation)
    mae_48 = [float(mean_absolute_error(y_true_kw[:, s], y_pred_kw[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_true_kw[:, s], y_pred_kw[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_kw.astype(np.float32)

    if best_val_loss < best_overall_val_loss and best_model_weights is not None:
        best_overall_val_loss = best_val_loss
        best_seed_id = SEED
        torch.save(best_model_weights, f"15_nlinear_baseline_pytorch_best.pt")
        results_data["best_seed"] = int(SEED)
        print(f"  [Checkpoint] New overall best model saved from SEED {SEED} (Val Loss: {best_val_loss:.6f}) -> 15_nlinear_baseline_pytorch_best.pt")

    results_data["seeds"][str(SEED)] = {
        "training_time_seconds": seed_duration,
        "peak_gpu_memory_mb": peak_vram_mb,
        "epochs": list(range(1, len(train_loss_history) + 1)),
        "train_loss": [float(v) for v in train_loss_history],
        "val_loss": [float(v) for v in val_loss_history],
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "overall_metrics": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics.items()},
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
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# 6. Final Summary
# ---------------------------------------------------------
all_predictions["y_true"] = y_true_kw.astype(np.float32)
pred_stack = np.stack([all_predictions[f"seed_{s}"] for s in SEEDS], axis=0)
all_predictions["pred_mean"] = np.mean(pred_stack, axis=0).astype(np.float32)
all_predictions["pred_std"] = np.std(pred_stack, axis=0).astype(np.float32)
np.savez_compressed(f"15_nlinear_baseline_pytorch_predictions.npz", **all_predictions)
print(f"Successfully saved all seed predictions to 15_nlinear_baseline_pytorch_predictions.npz")

print(f"\n======================================================================")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — 15_nlinear_baseline_pytorch")
print(f"======================================================================")
metric_keys = ['mae', 'rmse', 'r2', 'wape', 'mape', 'bias', 'negative_pct', 'training_time_seconds', 'peak_gpu_memory_mb']
summary_dict = {}
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
print(f"\nSaved results to {output_json_filename}")
