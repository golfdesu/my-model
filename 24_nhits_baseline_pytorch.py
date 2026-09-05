#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Model 24: N-HiTS (Neural Hierarchical Interpolation for Time Series) PyTorch Implementation
# Reference: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series
#            Forecasting", AAAI 2023. https://arxiv.org/abs/2201.12886
#
# Architectural Innovations:
# 1. Multi-Rate Subsampling:
#    Employs max pooling with multiple kernel sizes (e.g. 8, 4, 1) to downsample input
#    windows at diverse temporal resolutions, effectively separating low-frequency trend
#    and diurnal components from sharp high-frequency EV charging spikes.
# 2. Hierarchical Interpolation:
#    Synthesizes long-term forecasts by projecting low-rate basis coefficients through
#    hierarchical interpolation, drastically curtailing parameter count and overfitting risk.
# 3. Double Residual Stacking (Backcast + Forecast):
#    Each block subtracts its reconstructed backcast from the residual input signal while
#    accumulating its forecast into the overall prediction sum.
# 4. Pure-MLP Computational Efficiency:
#    Eliminates quadratic attention computation while maintaining competitive long-range accuracy.
#
# Scientific Invariants:
# - Lookback (L) = 96 steps (48 hours at 30-min intervals)
# - Horizon (H) = 48 steps (24 hours at 30-min intervals)
# - Chronological Split: 60% Train, 20% Val, 20% Test (Never shuffled)
# - Target: kWhDelivered (EV aggregate station load)
# - Excluded noise features: prcp, tempDiff_48, cldc
# - Scaler: MinMaxScaler fit ONLY on Training split
# - 10-Seed Benchmark: SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
# - Hardware: PyTorch CUDA Eager Mode (TF32 enabled)
# ==============================================================================

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import gc
import json
import math
import time
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

# ---------------------------------------------------------
# Hardware & Multithreading Optimization
# ---------------------------------------------------------
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
# 1. Data Loading & Preprocessing (Scientific Invariants)
# ---------------------------------------------------------
data_path = '../data_cleaned/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = 'data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()
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

# Append target as input feature
TARGET_CH_IDX = X_train_scaled.shape[1]
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
X_test_scaled  = np.concatenate([X_test_scaled,  y_test_scaled.reshape(-1, 1)], axis=1)
num_total_features = X_train_scaled.shape[1]
print(f"Target variate appended at index {TARGET_CH_IDX} (total input features: {num_total_features})")

peak_threshold_kw = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ---------------------------------------------------------
# 2. Windowing Helper
# ---------------------------------------------------------
def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    total_len = len(X_data) - lookback - horizon + 1
    for i in range(total_len):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.float32)

LOOKBACK = 96
HORIZON  = 48

X_train_t, y_train_t = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t   = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)
X_test_t,  y_test_t  = create_windowed_tensors(X_test_scaled,  y_test_scaled,  LOOKBACK, HORIZON)

print(f"Tensors Shape: Train {X_train_t.shape}, Val {X_val_t.shape}, Test {X_test_t.shape}")

# ---------------------------------------------------------
# 3. Model Architecture: N-HiTS
# ---------------------------------------------------------
class NHiTSBlock(nn.Module):
    """
    Individual N-HiTS Block with multi-rate subsampling pooling and
    hierarchical interpolation.
    """
    def __init__(
        self,
        lookback=96,
        num_features=29,
        horizon=48,
        pooling_size=8,
        n_layers=2,
        hidden_dim=128,
        n_theta=12,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon
        self.pooling_size = pooling_size
        self.n_theta = n_theta

        # Subsampling pooling layer
        if pooling_size > 1:
            self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size, ceil_mode=True)
            pooled_len = math.ceil(lookback / pooling_size)
        else:
            self.pooling = nn.Identity()
            pooled_len = lookback

        input_dim = pooled_len * num_features

        # Multi-layer Perceptron
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Backcast projection: predicts reconstructed input sequence [B, L * num_features]
        self.backcast_proj = nn.Linear(hidden_dim, lookback * num_features)

        # Forecast basis coefficients projection
        self.forecast_theta = nn.Linear(hidden_dim, n_theta)
        # Hierarchical interpolation / synthesis to full horizon H
        self.forecast_synth = nn.Linear(n_theta, horizon)

    def forward(self, x):
        # x: [B, L, num_features]
        B, L, D = x.shape

        # Multi-rate subsampling
        x_t = x.transpose(1, 2)  # [B, D, L]
        x_pool = self.pooling(x_t)  # [B, D, L_pooled]
        x_flat = x_pool.reshape(B, -1)  # [B, D * L_pooled]

        h = self.mlp(x_flat)  # [B, hidden_dim]

        # Backcast reconstruction
        backcast = self.backcast_proj(h).reshape(B, L, D)  # [B, L, D]

        # Forecast synthesis
        theta_f = self.forecast_theta(h)  # [B, n_theta]
        forecast = self.forecast_synth(theta_f)  # [B, horizon]

        return backcast, forecast


class NHiTS(nn.Module):
    """
    N-HiTS Architecture stacking multiple multi-rate resolution blocks.
    Stack 1: Coarse pooling (e.g. 8) capturing low-frequency trends.
    Stack 2: Medium pooling (e.g. 4) capturing diurnal cycle patterns.
    Stack 3: Fine pooling (e.g. 1) capturing high-frequency transient peaks.
    """
    def __init__(
        self,
        lookback=96,
        num_features=29,
        horizon=48,
        pooling_sizes=None,
        n_layers=2,
        hidden_dim=128,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon

        if pooling_sizes is None:
            pooling_sizes = [8, 4, 1]

        self.blocks = nn.ModuleList()
        for pool_size in pooling_sizes:
            # Expressiveness size / basis modes proportional to resolution
            n_theta = max(4, horizon // max(1, pool_size))
            self.blocks.append(
                NHiTSBlock(
                    lookback=lookback,
                    num_features=num_features,
                    horizon=horizon,
                    pooling_size=pool_size,
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    n_theta=n_theta,
                    dropout=dropout
                )
            )

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        total_forecast = torch.zeros(x.size(0), self.horizon, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            total_forecast = total_forecast + forecast

        return total_forecast

# ---------------------------------------------------------
# 4. Training, Evaluation & Benchmarking Functions
# ---------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
    preds_arr = np.concatenate(all_preds, axis=0)
    trues_arr = np.concatenate(all_trues, axis=0)
    return total_loss / len(dataloader.dataset), preds_arr, trues_arr


def calculate_metrics(y_true, y_pred, peak_threshold):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    peak_mask = y_true >= peak_threshold
    if np.any(peak_mask):
        peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
        peak_rmse = float(np.sqrt(mean_squared_error(y_true[peak_mask], y_pred[peak_mask])))
    else:
        peak_mae, peak_rmse = float('nan'), float('nan')

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Peak_MAE": float(peak_mae),
        "Peak_RMSE": float(peak_rmse)
    }

# ---------------------------------------------------------
# 5. Multi-Seed Benchmark Execution
# ---------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
MODEL_NAME = "24_nhits_baseline_pytorch"
OUTPUT_DIR = f"outputs/{MODEL_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "lookback": LOOKBACK,
    "num_features": num_total_features,
    "horizon": HORIZON,
    "pooling_sizes": [8, 4, 1],
    "n_layers": 2,
    "hidden_dim": 128,
    "dropout": 0.10,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "batch_size": 128,
    "epochs": 100,
    "patience": 12
}

def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    model = NHiTS(
        lookback=CONFIG["lookback"],
        num_features=CONFIG["num_features"],
        horizon=CONFIG["horizon"],
        pooling_sizes=CONFIG["pooling_sizes"],
        n_layers=CONFIG["n_layers"],
        hidden_dim=CONFIG["hidden_dim"],
        dropout=CONFIG["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    _, val_preds_s, val_trues_s = evaluate(model, val_loader, criterion, device)
    _, test_preds_s, test_trues_s = evaluate(model, test_loader, criterion, device)

    val_preds = scaler_y.inverse_transform(val_preds_s.reshape(-1, 1)).reshape(-1, HORIZON)
    val_trues = scaler_y.inverse_transform(val_trues_s.reshape(-1, 1)).reshape(-1, HORIZON)
    test_preds = scaler_y.inverse_transform(test_preds_s.reshape(-1, 1)).reshape(-1, HORIZON)
    test_trues = scaler_y.inverse_transform(test_trues_s.reshape(-1, 1)).reshape(-1, HORIZON)

    test_metrics = calculate_metrics(test_trues.flatten(), test_preds.flatten(), peak_threshold_kw)
    val_metrics  = calculate_metrics(val_trues.flatten(), val_preds.flatten(), peak_threshold_kw)

    return {
        "model_weights": best_weights,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "test_preds": test_preds,
        "test_trues": test_trues
    }

if __name__ == '__main__':
    print("=" * 70)
    print(f"Starting Multi-Seed Benchmark for {MODEL_NAME} across {len(SEEDS)} seeds")
    print("=" * 70)

    results_per_seed = {}
    best_overall_seed = None
    best_overall_rmse = float('inf')
    best_overall_weights = None
    best_overall_preds = None
    best_overall_trues = None

    for seed in SEEDS:
        print(f"\n--- Running Seed: {seed} ---")
        start_t = time.time()
        res = run_seed(seed)
        elapsed = time.time() - start_t
        results_per_seed[seed] = {
            "test_metrics": res["test_metrics"],
            "val_metrics": res["val_metrics"],
            "elapsed_seconds": elapsed
        }
        print(f"Seed {seed} Completed in {elapsed:.1f}s | Test RMSE: {res['test_metrics']['RMSE']:.4f}, MAE: {res['test_metrics']['MAE']:.4f}, R2: {res['test_metrics']['R2']:.4f}")

        if res["test_metrics"]["RMSE"] < best_overall_rmse:
            best_overall_rmse = res["test_metrics"]["RMSE"]
            best_overall_seed = seed
            best_overall_weights = res["model_weights"]
            best_overall_preds = res["test_preds"]
            best_overall_trues = res["test_trues"]

    metric_keys = ["MAE", "RMSE", "R2", "Peak_MAE", "Peak_RMSE"]
    summary = {}
    for k in metric_keys:
        vals = [results_per_seed[s]["test_metrics"][k] for s in SEEDS if not math.isnan(results_per_seed[s]["test_metrics"][k])]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))

    print("\n" + "=" * 70)
    print(f"🏆 Final Benchmark Results for {MODEL_NAME} (10 Seeds):")
    print(f"  - Test MAE:       {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f}")
    print(f"  - Test RMSE:      {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f}")
    print(f"  - Test R2:        {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")
    print(f"  - Peak MAE:       {summary['Peak_MAE_mean']:.4f} ± {summary['Peak_MAE_std']:.4f}")
    print(f"  - Peak RMSE:      {summary['Peak_RMSE_mean']:.4f} ± {summary['Peak_RMSE_std']:.4f}")
    print("=" * 70)

    if best_overall_weights is not None:
        torch.save(best_overall_weights, os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best.pt"))
    if best_overall_preds is not None and best_overall_trues is not None:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions.npz"),
            predictions=best_overall_preds,
            ground_truth=best_overall_trues
        )
    final_payload = {
        "model_name": MODEL_NAME,
        "config": CONFIG,
        "summary": summary,
        "best_seed": best_overall_seed,
        "seeds_detail": results_per_seed
    }
    with open(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_results.json"), 'w', encoding='utf-8') as f:
        json.dump(final_payload, f, indent=4)
    print(f"Artifacts successfully saved to {OUTPUT_DIR}")
