#!/usr/bin/env python
# coding: utf-8

# iTransformer (Liu et al., ICLR 2024 Spotlight)
# Key innovation: Inverts temporal and variate dimensions.
# - Standard Transformer: attention over time steps (tokens = time steps)
# - iTransformer: attention over variates/features (tokens = each feature's full time series)
# This captures multivariate correlations directly.
# Especially suited for datasets with many informative features (our dataset: 30 features).

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import subprocess
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
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print(f"CPU Multithreading Optimized with {num_cpus} threads")

vcvars_path = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if os.path.exists(vcvars_path):
    try:
        msvc_env = subprocess.check_output(f'cmd.exe /c ""{vcvars_path}" && set"', text=True)
        for line in msvc_env.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
    except Exception:
        pass

# ---------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------
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

# iTransformer (paper, ICLR 2024): each variate is a token and the shared head forecasts
# EVERY variate's own future; the target's forecast is read from the target variate token.
# The target series must therefore be one of the input variates -> append scaled y.
TARGET_CH_IDX = X_train_scaled.shape[1]  # index of the appended target variate
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
X_test_scaled  = np.concatenate([X_test_scaled,  y_test_scaled.reshape(-1, 1)], axis=1)
print(f"Target variate appended at index {TARGET_CH_IDX} (total variates: {X_train_scaled.shape[1]})")

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
    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape, mae_peak=mae_peak, wape_peak=wape_peak)

# ---------------------------------------------------------
# 3. iTransformer Architecture
# ---------------------------------------------------------
class iTransformerModel(nn.Module):
    """
    iTransformer (Liu et al., ICLR 2024 Spotlight)
    Inverts the role of time and variate dimensions in Transformer.
    Each token = one input feature's full time series projected to d_model.
    Attention is computed across variates (features), not time steps.
    Captures multivariate correlations that standard Transformers miss.
    """
    def __init__(self, lookback, num_features, horizon,
                 d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.num_features = num_features
        self.horizon = horizon

        # Project each variate's time series [lookback] -> [d_model]
        # Each feature becomes one token; its embedding encodes its full historical trajectory
        self.variate_proj = nn.Linear(lookback, d_model)
        self.drop_in = nn.Dropout(dropout_rate)

        # Standard Transformer encoder — but applied over the feature/variate dimension.
        # Official iTransformer (THUML): post-norm layers + GELU activation
        # + a FINAL LayerNorm after the whole stack (Encoder(norm_layer=LayerNorm)).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout_rate, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model)   # official: final LayerNorm over variate tokens
        )

        # Project each variate's encoded representation -> forecast horizon
        self.output_proj = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: [batch, lookback, num_features] - last variate is the appended target series

        # Invert: treat each variate as a token
        x = x.transpose(1, 2)                          # [batch, num_features, lookback]

        # Embed each variate's time series to d_model
        x = self.drop_in(self.variate_proj(x))         # [batch, num_features, d_model]

        # Attention over the variate (feature) dimension
        x = self.encoder(x)                            # [batch, num_features, d_model]

        # Project each variate's representation to the forecast horizon
        x = self.output_proj(x)                        # [batch, num_features, horizon]

        # Paper-faithful: read the forecast directly from the target variate token
        x = x[:, TARGET_CH_IDX, :]                     # [batch, horizon]
        return x

# ---------------------------------------------------------
# 4. Config & Tensor Pre-build
# ---------------------------------------------------------
import time

LOOKBACK   = 96
HORIZON    = 48
BATCH_SIZE = 256
SEEDS      = [164, 256, 355, 1234, 2026]
output_md_filename = "13_tfm_itfm_pytorch.md"
steps_to_eval = [0, 5, 11, 47]
step_labels   = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print("Pre-building sequence tensors...")
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t,   _, _ = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)
X_test_t,  y_test_t,  X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

print(f"Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")
print(f"Starting {len(SEEDS)}-Seed Loop for iTransformer...")

# ---------------------------------------------------------
# 5. Seed Training Loop
# ---------------------------------------------------------
all_seed_metrics = []
md_lines = [
    f"# iTransformer (ICLR 2024) — ACN Caltech EV Charging Forecast\n",
    f"LOOKBACK={LOOKBACK}, HORIZON={HORIZON}, BATCH_SIZE={BATCH_SIZE}, SEEDS={SEEDS}\n\n"
]

for seed_idx, SEED in enumerate(SEEDS, 1):
    print(f"\n{'='*70}")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = iTransformerModel(
        lookback=LOOKBACK, num_features=X_train_scaled.shape[1], horizon=HORIZON,
        d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if seed_idx == 1:
        print(f"Model Parameters: {total_params:,}")
        md_lines.append(f"**Model Parameters:** {total_params:,}\n\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    epochs          = 100
    patience        = 15
    best_val_loss   = float('inf')
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

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
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

    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    # Test evaluation
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

    # Overall metrics
    metrics = compute_metrics(y_true_kw.flatten(), y_pred_kw.flatten(), peak_threshold_kw)
    all_seed_metrics.append(metrics)

    print(f"\n--- Seed {SEED} Overall Test Metrics ---")
    print(f"  MAE:  {metrics['mae']:.4f} kWh")
    print(f"  RMSE: {metrics['rmse']:.4f} kWh")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  WAPE: {metrics['wape']:.2f}%")

    md_lines.append(f"\n## Seed {SEED}\n")
    md_lines.append(f"| Metric | Value |\n|---|---|\n")
    md_lines.append(f"| MAE | {metrics['mae']:.4f} kWh |\n")
    md_lines.append(f"| RMSE | {metrics['rmse']:.4f} kWh |\n")
    md_lines.append(f"| R² | {metrics['r2']:.4f} |\n")
    md_lines.append(f"| WAPE | {metrics['wape']:.2f}% |\n")
    md_lines.append(f"| MAPE | {metrics['mape']:.2f}% |\n")
    md_lines.append(f"| MAE Peak | {metrics['mae_peak']:.4f} kWh |\n")
    md_lines.append(f"| WAPE Peak | {metrics['wape_peak']:.2f}% |\n")

    # Per-step metrics
    md_lines.append(f"\n### Per-Step Metrics\n| Step | MAE | RMSE | WAPE |\n|---|---|---|---|\n")
    for step in steps_to_eval:
        step_metrics = compute_metrics(y_true_kw[:, step], y_pred_kw[:, step], peak_threshold_kw)
        label = step_labels[step]
        md_lines.append(f"| {label} | {step_metrics['mae']:.4f} | {step_metrics['rmse']:.4f} | {step_metrics['wape']:.2f}% |\n")

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# 6. Final Summary
# ---------------------------------------------------------
print(f"\n{'='*70}")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — iTransformer")
print(f"{'='*70}")
metric_keys = ['mae', 'rmse', 'r2', 'wape', 'mape']
summary_lines = ["\n## Final Summary (Mean ± Std across Seeds)\n",
                 "| Metric | Mean | Std |\n|---|---|---|\n"]
for k in metric_keys:
    vals = [m[k] for m in all_seed_metrics if not np.isnan(m[k])]
    mu, sigma = np.mean(vals), np.std(vals)
    print(f"  {k.upper():<8}: {mu:.4f} ± {sigma:.4f}")
    summary_lines.append(f"| {k.upper()} | {mu:.4f} | {sigma:.4f} |\n")

md_lines += summary_lines

with open(output_md_filename, 'w', encoding='utf-8') as f:
    f.writelines(md_lines)
print(f"\nSaved results to {output_md_filename}")
