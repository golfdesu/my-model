import json
#!/usr/bin/env python
# coding: utf-8

# TimesNet (Wu et al., ICLR 2023)
# Key innovation: Transforms 1D time series into 2D representations based on
# dominant periods found via FFT, then applies 2D Inception convolution to
# capture both intra-period (within one cycle) and inter-period (across cycles) variations.
# Particularly suited for data with strong periodicity — our dataset: daily s=48 dominant.

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
import torch.nn.functional as F
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
if device.type == 'cuda':
    print("GPU Model:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print(f"CPU Multithreading Optimized with {num_cpus} threads")


# ---------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------
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
print(f"Dataset Loaded from {data_path}! Rows: {len(df)}, Features: {len(cols)}")

train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]
X_test  = X[train_len + val_len :]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

peak_threshold_kw = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------
def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
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
# 3. TimesNet Architecture
# ---------------------------------------------------------
class Inception_Block(nn.Module):
    """Multi-scale 2D convolution block (simplified Inception) for TimesNet."""
    def __init__(self, in_channels, out_channels, num_kernels=4):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_kernels):
            k = 2 * i + 1   # kernel sizes: 1, 3, 5, 7
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2))
        self.norm = nn.BatchNorm2d(out_channels)
        self.act  = nn.GELU()

    def forward(self, x):
        # x: [B, C, H, W]
        out = sum(conv(x) for conv in self.convs) / len(self.convs)
        return self.act(self.norm(out))

class TimesBlock(nn.Module):
    """
    Core TimesNet block.
    1. Use FFT to detect dominant periods in the input sequence.
    2. Reshape 1D sequence into 2D (rows=cycles, cols=period_length).
    3. Apply 2D Inception convolution to capture intra- and inter-period patterns.
    4. Reshape back to 1D and aggregate across all detected periods.
    """
    def __init__(self, seq_len, d_model, d_ff, top_k=3, num_kernels=4):
        super().__init__()
        self.seq_len  = seq_len
        self.top_k    = top_k
        self.conv1 = Inception_Block(d_model, d_ff,    num_kernels=num_kernels)
        self.conv2 = Inception_Block(d_ff,    d_model, num_kernels=num_kernels)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()

        # FFT-based period detection (official impl.: amplitude averaged over
        # batch & channels before selecting top-k frequencies)
        xf = torch.fft.rfft(x.mean(dim=-1), dim=1)   # [B, T//2+1]
        amp_all = xf.abs()                            # per-sample amplitudes [B, F]
        freq_scores = torch.cat(
            [torch.zeros_like(amp_all[:, :1]), amp_all[:, 1:]], dim=1
        ).mean(dim=0)                                 # drop DC (no in-place op) + batch mean -> [F]
        top_k_actual = min(self.top_k, freq_scores.size(0) - 1)
        _, top_freq_idx = torch.topk(freq_scores, top_k_actual)

        # Paper (ICLR 2023): ADAPTIVE aggregation - each period weighted by
        # softmax of its FFT amplitude, not a flat mean
        period_list = []
        weights_list = []
        for idx in top_freq_idx.tolist():
            p = T // max(int(idx), 1)
            if p >= 2:
                period_list.append(p)
                weights_list.append(amp_all[:, int(idx)].mean())  # scalar tensor (differentiable path kept simple)
        if not period_list:
            period_list = [48]                        # fallback to known daily period
            weights_list = None

        res_list = []
        for period in period_list:
            # Pad to make T divisible by period (replicate padding, official impl.)
            pad_len = (period - T % period) % period
            if pad_len > 0:
                xp = F.pad(x.transpose(1, 2), (0, pad_len), mode='replicate').transpose(1, 2)
            else:
                xp = x                                # [B, T+pad, C]
            Tp = T + pad_len

            # Reshape 1D -> 2D: [B, C, num_cycles, period]
            xp = xp.reshape(B, Tp // period, period, C).permute(0, 3, 1, 2)

            # 2D Inception convolution
            xp = self.conv2(self.conv1(xp))    # [B, C, num_cycles, period]

            # Reshape 2D -> 1D: [B, Tp, C]
            xp = xp.permute(0, 2, 3, 1).reshape(B, Tp, C)
            xp = xp[:, :T, :]                  # trim back to T
            res_list.append(xp)

        res_stack = torch.stack(res_list, dim=-1)     # [B, T, C, k]
        if weights_list is not None:
            w = torch.softmax(torch.stack(weights_list), dim=0)   # [k] adaptive weights
            res = (res_stack * w.view(1, 1, 1, -1)).sum(dim=-1)
        else:
            res = res_stack.mean(dim=-1)
        return self.norm(x + res)

class TimesNetModel(nn.Module):
    """
    TimesNet (Wu et al., ICLR 2023)
    Transforms temporal variation structure into 2D space via period-based reshaping,
    then applies vision-style 2D convolutions to learn multi-scale temporal patterns.
    """
    def __init__(self, lookback, num_features, horizon,
                 d_model=64, d_ff=128, num_layers=2, top_k=3, num_kernels=4, dropout_rate=0.1):
        super().__init__()
        # Input projection: multivariate features -> d_model channels
        self.proj_in = nn.Linear(num_features, d_model)
        self.drop    = nn.Dropout(dropout_rate)

        # Stack of TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(lookback, d_model, d_ff, top_k=top_k, num_kernels=num_kernels)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Official TimesNet forecasting head (thuml/Time-Series-Library):
        #   predict_linear projects along the TIME axis (keeps temporal order/resolution),
        #   then target_proj maps d_model -> 1 (target series). NO temporal pooling.
        self.predict_linear = nn.Linear(lookback, horizon)   # time-axis projection: [B, d, L] -> [B, d, H]
        self.target_proj = nn.Linear(d_model, 1)             # per-timestep projection to the target series

    def forward(self, x):
        # x: [batch, lookback, num_features]
        x = self.drop(self.proj_in(x))   # [batch, lookback, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)                                             # [B, L, d]
        x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)   # [B, H, d] — temporal projection
        out = self.target_proj(x).squeeze(-1)                        # [B, H]
        return out

# ---------------------------------------------------------
# 4. Config & Tensor Pre-build
# ---------------------------------------------------------
import time

LOOKBACK   = 96
HORIZON    = 48
BATCH_SIZE = 128
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
output_json_filename = "14_tfm_timesnet_pytorch_results.json"
results_data = {
    "model_name": "14_tfm_timesnet_pytorch",
    "seeds": {},
    "summary": {}
}
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print("Pre-building sequence tensors...")
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t,   _, _ = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)
X_test_t,  y_test_t,  X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

print(f"Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")
print(f"Starting {len(SEEDS)}-Seed Loop for TimesNet...")

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
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = TimesNetModel(
        lookback=LOOKBACK, num_features=X_train_scaled.shape[1], horizon=HORIZON,
        d_model=32, d_ff=128, num_layers=3, top_k=4, num_kernels=4, dropout_rate=0.2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if seed_idx == 1:
        print(f"Model Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0011311694756729275, weight_decay=0.000697139666328406)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device.type == 'cuda' else 0.0
    metrics["training_time_seconds"] = seed_duration
    metrics["peak_gpu_memory_mb"] = peak_vram_mb

    # 48-step full horizon evaluation (24-hour error degradation)
    mae_48 = [float(mean_absolute_error(y_true_kw[:, s], y_pred_kw[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_true_kw[:, s], y_pred_kw[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_kw.astype(np.float32)

    if best_val_loss < best_overall_val_loss and best_model_weights is not None:
        best_overall_val_loss = best_val_loss
        best_seed_id = SEED
        torch.save(best_model_weights, f"14_tfm_timesnet_pytorch_best.pt")
        results_data["best_seed"] = int(SEED)
        print(f"  [Checkpoint] New overall best model saved from SEED {SEED} (Val Loss: {best_val_loss:.6f}) -> 14_tfm_timesnet_pytorch_best.pt")

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
np.savez_compressed(f"14_tfm_timesnet_pytorch_predictions.npz", **all_predictions)
print(f"Successfully saved all seed predictions to 14_tfm_timesnet_pytorch_predictions.npz")

print(f"\n======================================================================")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — 14_tfm_timesnet_pytorch")
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
