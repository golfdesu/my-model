#!/usr/bin/env python
# coding: utf-8

# # S4D Baseline (Diagonal Structured State Space Sequence Model)
# Time-series forecasting model based on S4D (Gu et al., ICLR 2022).
# Simplifies HiPPO continuous state spaces into diagonal transition matrices Lambda,
# evaluated through an exact Cauchy kernel and circular FFT convolution in O(L log L) time.

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import subprocess
import gc
import math
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

# Safe tqdm import fallback
try:
    from tqdm.auto import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable, *args, **kwargs):
            return iterable

warnings.filterwarnings('ignore')

# CPU Multithreading Speed Optimization
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

# Load and clean dataset (Local path auto-detect)
data_path = 'acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = 'acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = '../preprocess/acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = '../preprocess/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = '../data_cleaned/acn_caltech_ready2.csv'
if not os.path.exists(data_path):
    data_path = r'C:\Users\chaya\Documents\Program\Practice\preprocess\acn_caltech_ready.csv'
if not os.path.exists(data_path):
    data_path = r'C:\Users\chaya\Documents\Program\Practice\preprocess\acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()

# Drop unneeded columns
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = []
for col in df.columns:
    df[col] = df[col].astype('float32')
    if col != 'kWhDelivered':
        cols.append(col)

X = df[cols]
y = df['kWhDelivered']

print(f"Dataset Loaded successfully from {data_path}! Total Rows: {len(df)}, Features Count: {len(cols)}")

# Train/Val/Test Split (60% / 20% / 20%)
train_len = int(len(df) * 0.6)
val_len = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]
X_test  = X[train_len + val_len :]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

# Feature Scaling (MinMaxScaler fitted ONLY on train split)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# Target Scaling (MinMaxScaler for y fitted ONLY on train split)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Compute Peak Load Threshold (Top 20% of TRAIN in actual kW)
peak_threshold_kw = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# Helper 1: PyTorch DataLoader Creator
def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t, np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

# Helper 2: S4D Diagonal State Space Kernel
class S4DKernel(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.Lambda_real = nn.Parameter(-0.5 * torch.ones(d_model, d_state))
        self.Lambda_imag = nn.Parameter(math.pi * torch.arange(d_state).unsqueeze(0).repeat(d_model, 1))

        self.C = nn.Parameter(torch.randn(d_model, d_state, 2) * (0.5 / math.sqrt(d_state)))

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        C = torch.complex(self.C[..., 0], self.C[..., 1])

        dtA = dt.unsqueeze(-1) * Lambda
        t = torch.arange(L, device=dt.device)
        V = torch.exp(dtA.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0))
        K = 2 * torch.einsum('hn,hnl->hl', C, V).real
        return K

class S4DBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model, d_state=d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model, d_model)  # Corrected dimension
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch, L, d_model = x.shape
        res = x

        K = self.kernel(L)
        x_tr = x.transpose(1, 2)
        fft_len = L + L
        x_fft = torch.fft.rfft(x_tr, n=fft_len, dim=-1)
        k_fft = torch.fft.rfft(K, n=fft_len, dim=-1)
        y = torch.fft.irfft(x_fft * k_fft, n=fft_len, dim=-1)[..., :L]
        y = (y + x_tr * self.D.unsqueeze(-1)).transpose(1, 2)

        u, gate = self.linear1(y).chunk(2, dim=-1)
        out = self.linear2(F.gelu(u) * torch.sigmoid(gate))
        return self.norm(res + self.drop(out))

class S4DModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, d_state=64, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([
            S4DBlock(d_model=d_model, d_state=d_state, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        self.head_fc1 = nn.Linear(d_model * 2, 128)
        self.head_drop1 = nn.Dropout(dropout_rate)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_proj(x)
        for block in self.blocks:
            x = block(x)

        last_feat = x[:, -1, :]
        avg_feat = torch.mean(x, dim=1)
        ctx = torch.cat([last_feat, avg_feat], dim=-1)

        h = self.relu(self.head_fc1(ctx))
        h = self.head_drop1(h)
        h = self.relu(self.head_fc2(h))
        h = self.head_drop2(h)
        out = self.out_proj(h)
        return out

# Helper: Metrics Evaluator Function
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

import time
# Config Parameters
LOOKBACK = 96      # 48 hours history (96 * 30 min)
HORIZON = 48       # 24 hours forecast (48 * 30 min)
BATCH_SIZE = 256
SEEDS = [164, 256, 355, 1234, 2026]
output_md_filename = "20_s4d_baseline_pytorch.md"
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print('Pre-building sequence tensors...')
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t, _, _     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)
X_test_t, y_test_t, X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

with open(output_md_filename, "w", encoding="utf-8") as f:
    f.write(f"# S4D Baseline PyTorch Benchmark Results\n\n- Lookback: {LOOKBACK}\n- Horizon: {HORIZON}\n- Batch Size: {BATCH_SIZE}\n- Seeds: {SEEDS}\n\n")

print(f"Starting Automated {len(SEEDS)}-Seed Loop for {output_md_filename} in PyTorch...")

all_seed_metrics = []

for seed_idx, SEED in enumerate(SEEDS, 1):
    print(f"\n=========================================================================")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"=========================================================================")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = S4DModel(lookback=LOOKBACK, num_features=X_train_scaled.shape[1], horizon=HORIZON, d_model=64, d_state=64, num_layers=2).to(device)


    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Seed {SEED} Training", leave=True)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        epoch_pbar.set_postfix({'train_loss': f"{train_loss:.5f}", 'val_loss': f"{val_loss:.5f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_pbar.write(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    model.eval()
    y_pred_list = []
    with torch.inference_mode():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            out = model(batch_X)
            y_pred_list.append(out.cpu().numpy())

    y_pred_scaled = np.vstack(y_pred_list)

    y_test_seq_unscaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    y_pred_unscaled     = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    actual_by_step = {step: y_test_seq_unscaled[:, step] for step in steps_to_eval}
    predictions_by_step = {step: y_pred_unscaled[:, step] for step in steps_to_eval}

    overall_m = compute_metrics(y_test_seq_unscaled.flatten(), y_pred_unscaled.flatten(), peak_threshold_kw)
    all_seed_metrics.append(overall_m)

    output_lines = []
    output_lines.append(f"## SEED {SEED}")
    output_lines.append("================ MODEL EVALUATION METRICS (per horizon step) ================")
    output_lines.append(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")
    output_lines.append("-------------------------------------------------------------------------------")
    output_lines.append(f"Overall MAE: {overall_m['mae']:.4f} kW | RMSE: {overall_m['rmse']:.4f} kW | R2: {overall_m['r2']:.4f} | WAPE: {overall_m['wape']:.2f}%\n")

    for step in steps_to_eval:
        m = compute_metrics(actual_by_step[step], predictions_by_step[step], peak_threshold_kw)
        output_lines.append(f"[{step_labels[step]}]")
        output_lines.append(f"  MAE   : {m['mae']:.4f} kW")
        output_lines.append(f"  RMSE  : {m['rmse']:.4f} kW")
        output_lines.append(f"  R²    : {m['r2']:.4f}")
        output_lines.append(f"  MAPE  : {m['mape']:.2f}%")
        output_lines.append(f"  WAPE  : {m['wape']:.2f}%")
        output_lines.append(f"  Peak Zone MAE : {m['mae_peak']:.4f} kW")
        output_lines.append(f"  Peak Zone WAPE: {m['wape_peak']:.2f}%\n")

    full_output_text = "\n".join(output_lines)
    print(full_output_text)

    with open(output_md_filename, "a", encoding="utf-8") as f:
        f.write(full_output_text + "\n")

    print(f"Successfully saved SEED {SEED} metrics to {output_md_filename}")
    gc.collect()

print(f"\n{'='*70}")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — S4D")
print(f"{'='*70}")
summary_lines = ["\n## Final Summary (Mean ± Std across Seeds)\n\n",
                 "| Metric | Mean | Std |\n|---|---|---|\n"]
for k in ['mae', 'rmse', 'r2', 'wape', 'mape']:
    vals = [m[k] for m in all_seed_metrics if not np.isnan(m[k])]
    mu, sigma = np.mean(vals), np.std(vals)
    print(f"  {k.upper():<8}: {mu:.4f} ± {sigma:.4f}")
    summary_lines.append(f"| {k.upper()} | {mu:.4f} | {sigma:.4f} |\n")

with open(output_md_filename, "a", encoding="utf-8") as f:
    f.writelines(summary_lines)

print(f"\nFinished running all {len(SEEDS)} SEEDs for {output_md_filename}!")
