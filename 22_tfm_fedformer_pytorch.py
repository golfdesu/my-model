#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Model 22: FEDformer (Frequency Enhanced Decomposed Transformer) PyTorch Implementation
# Reference: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for
#            Long-term Series Forecasting", ICML 2022. https://arxiv.org/abs/2201.12740
#
# Architectural Innovations:
# 1. Seasonal-Trend Decomposition Architecture:
#    Integrates moving average pooling to disentangle long-term trend-cyclical components
#    from high-frequency seasonal variations at every layer.
# 2. Frequency Enhanced Block with Fourier Transform (FEA-f):
#    Transforms time representations to the frequency domain via Real FFT (rfft),
#    selects dominant frequency modes, and performs compact matrix multiplications in
#    the Fourier domain with O(L) linear computational complexity.
# 3. Frequency-Domain Cross-Attention:
#    Cross-attends frequency components between encoder history and decoder queries to
#    capture global periodic patterns across the entire sequence.
# 4. Multi-Step Direct Prediction:
#    Combines extrapolated trend projections with frequency-enhanced seasonal projections
#    to predict the aggregate EV charging load for H=48 steps (24 hours).
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
# 3. Model Architecture: FEDformer
# ---------------------------------------------------------
class SeriesDecomp(nn.Module):
    """
    Moving average series decomposition block (Wu et al. / Zhou et al.)
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, D]
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front = x[:, 0:1, :].repeat(1, pad_front, 1)
        end   = x[:, -1:, :].repeat(1, pad_end, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.transpose(1, 2)  # [B, D, L_padded]
        trend = self.avg(x_pad).transpose(1, 2)  # [B, L, D]
        seasonal = x - trend
        return seasonal, trend


class FourierBlock(nn.Module):
    """
    Frequency Enhanced Block with Fourier Transform (FEB-f, Zhou et al., ICML 2022).
    Multi-head frequency representation learning with complex linear transformation.
    """
    def __init__(self, d_model, n_heads=4, modes=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_k = d_model // n_heads
        self.modes = modes

        self.weights_real = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        x_heads = x.view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        x_fft = torch.fft.rfft(x_heads, dim=2)
        L_freq = x_fft.shape[2]
        modes_eff = min(self.modes, L_freq)

        x_mode = x_fft[:, :, :modes_eff, :]
        w_real = self.weights_real[:, :modes_eff, :, :]
        w_imag = self.weights_imag[:, :modes_eff, :, :]

        xr = x_mode.real
        xi = x_mode.imag

        out_r = torch.einsum('bhmd,hmde->bhme', xr, w_real) - torch.einsum('bhmd,hmde->bhme', xi, w_imag)
        out_i = torch.einsum('bhmd,hmde->bhme', xr, w_imag) + torch.einsum('bhmd,hmde->bhme', xi, w_real)
        out_mode = torch.complex(out_r, out_i)

        out_fft = torch.zeros(B, self.n_heads, L_freq, self.d_k, device=x.device, dtype=torch.cfloat)
        out_fft[:, :, :modes_eff, :] = out_mode

        out_time = torch.fft.irfft(out_fft, n=L, dim=2)
        out_time = out_time.permute(0, 2, 1, 3).contiguous().view(B, L, D)

        return self.out_proj(out_time)


class FourierCrossAttention(nn.Module):
    """
    Fourier Cross Attention (FEA-f, Zhou et al., ICML 2022).
    Cross-attention in frequency domain between decoder queries and encoder keys/values.
    """
    def __init__(self, d_model, n_heads=4, modes=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_k = d_model // n_heads
        self.modes = modes

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.weights_real = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, modes, self.d_k, self.d_k) * 0.02
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, cross):
        B, L_q, D = q.shape
        _, L_k, _ = cross.shape

        q_proj = self.q_proj(q).view(B, L_q, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k_proj = self.k_proj(cross).view(B, L_k, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        q_fft = torch.fft.rfft(q_proj, dim=2)
        k_fft = torch.fft.rfft(k_proj, dim=2)

        modes_eff = min(self.modes, q_fft.shape[2], k_fft.shape[2])
        q_mode = q_fft[:, :, :modes_eff, :]
        k_mode = k_fft[:, :, :modes_eff, :]

        cross_freq = (q_mode * torch.conj(k_mode)) / math.sqrt(self.d_k)
        xr = cross_freq.real
        xi = cross_freq.imag

        w_real = self.weights_real[:, :modes_eff, :, :]
        w_imag = self.weights_imag[:, :modes_eff, :, :]

        out_r = torch.einsum('bhmd,hmde->bhme', xr, w_real) - torch.einsum('bhmd,hmde->bhme', xi, w_imag)
        out_i = torch.einsum('bhmd,hmde->bhme', xr, w_imag) + torch.einsum('bhmd,hmde->bhme', xi, w_real)
        out_mode = torch.complex(out_r, out_i)

        out_fft = torch.zeros(B, self.n_heads, q_fft.shape[2], self.d_k, device=q.device, dtype=torch.cfloat)
        out_fft[:, :, :modes_eff, :] = out_mode

        out_time = torch.fft.irfft(out_fft, n=L_q, dim=2)
        out_time = out_time.permute(0, 2, 1, 3).contiguous().view(B, L_q, D)

        return self.out_proj(out_time)


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, modes=16, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        self.self_attn = FourierBlock(d_model=d_model, n_heads=n_heads, modes=modes)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_f = self.self_attn(x)
        x = x + self.dropout(x_f)
        x, _ = self.decomp1(x)

        x_ff = self.mlp(x)
        x = x + self.dropout(x_ff)
        x, _ = self.decomp2(x)
        return x


class FEDformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, modes=16, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        self.self_attn = FourierBlock(d_model=d_model, n_heads=n_heads, modes=modes)
        self.cross_attn = FourierCrossAttention(d_model=d_model, n_heads=n_heads, modes=modes)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.decomp3 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, seasonal, cross, trend_part):
        res = self.self_attn(seasonal)
        seasonal, trend1 = self.decomp1(seasonal + self.dropout(res))
        trend_part = trend_part + trend1

        res = self.cross_attn(seasonal, cross)
        seasonal, trend2 = self.decomp2(seasonal + self.dropout(res))
        trend_part = trend_part + trend2

        res = self.mlp(seasonal)
        seasonal, trend3 = self.decomp3(seasonal + self.dropout(res))
        trend_part = trend_part + trend3

        return seasonal, trend_part


class FEDformer(nn.Module):
    """
    Authentic FEDformer Architecture (Zhou et al., ICML 2022).
    Full Encoder-Decoder with Frequency Enhanced Block (FEB-f),
    Fourier Cross Attention (FEA-f), and Progressive Trend Accumulation.
    """
    def __init__(
        self,
        lookback=96,
        num_features=30,
        horizon=48,
        d_model=64,
        n_heads=4,
        modes=16,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_ff=128,
        dropout=0.1,
        kernel_size=25
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.horizon = horizon
        self.label_len = lookback // 2
        self.d_model = d_model

        self.decomp_init = SeriesDecomp(kernel_size)
        self.enc_embedding = nn.Linear(num_features, d_model)
        self.dec_embedding = nn.Linear(num_features, d_model)
        self.trend_embedding = nn.Linear(num_features, d_model)

        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                modes=modes,
                d_ff=d_ff,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            FEDformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                modes=modes,
                d_ff=d_ff,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for _ in range(num_decoder_layers)
        ])

        self.seasonal_proj = nn.Linear(d_model, 1)
        self.trend_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]

        x_seasonal, x_trend = self.decomp_init(x)

        enc_in = self.enc_embedding(x_seasonal)
        enc_out = enc_in
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out)

        zeros_seasonal = torch.zeros(B, self.horizon, self.num_features, device=x.device, dtype=x.dtype)
        seasonal_dec_in = torch.cat([x_seasonal[:, -self.label_len:, :], zeros_seasonal], dim=1)

        mean_trend = x_trend.mean(dim=1, keepdim=True).repeat(1, self.horizon, 1)
        trend_dec_in = torch.cat([x_trend[:, -self.label_len:, :], mean_trend], dim=1)

        dec_seasonal = self.dec_embedding(seasonal_dec_in)
        dec_trend = self.trend_embedding(trend_dec_in)

        for dec_layer in self.decoder_layers:
            dec_seasonal, dec_trend = dec_layer(dec_seasonal, enc_out, dec_trend)

        seasonal_pred = dec_seasonal[:, -self.horizon:, :]
        trend_pred = dec_trend[:, -self.horizon:, :]

        out_seasonal = self.seasonal_proj(seasonal_pred).squeeze(-1)
        out_trend = self.trend_proj(trend_pred).squeeze(-1)

        out = out_seasonal + out_trend
        return out


FEDformerModel = FEDformer

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


def compute_metrics(actual, predicted, peak_threshold):
    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2   = float(r2_score(actual, predicted))
    wape = float((np.sum(np.abs(actual - predicted)) / np.sum(actual)) * 100)
    non_zero = actual > 0
    mape = float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100) if non_zero.any() else np.nan
    peak = actual >= peak_threshold
    if peak.any():
        mae_peak  = float(mean_absolute_error(actual[peak], predicted[peak]))
        wape_peak = float((np.sum(np.abs(actual[peak] - predicted[peak])) / np.sum(actual[peak])) * 100)
    else:
        mae_peak, wape_peak = np.nan, np.nan

    bias = float(np.mean(predicted - actual))
    negative_pct = float(np.mean(predicted < 0) * 100)

    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape,
                mae_peak=mae_peak, wape_peak=wape_peak, bias=bias, negative_pct=negative_pct)

# ---------------------------------------------------------
# 5. Multi-Seed Benchmark Execution
# ---------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
MODEL_NAME = "22_tfm_fedformer_pytorch"
OUTPUT_DIR = f"outputs/{MODEL_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default Architecture Configurations
CONFIG = {
    "lookback": LOOKBACK,
    "num_features": num_total_features,
    "horizon": HORIZON,
    "d_model": 32,
    "n_heads": 4,
    "modes": 16,
    "num_encoder_layers": 2,
    "num_decoder_layers": 1,
    "d_ff": 64,
    "dropout": 0.2,
    "kernel_size": 49,
    "learning_rate": 0.0017800057427018164,
    "weight_decay": 1.13508354920271e-05,
    "batch_size": 256,
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

    model = FEDformer(
        lookback=CONFIG["lookback"],
        num_features=CONFIG["num_features"],
        horizon=CONFIG["horizon"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        modes=CONFIG["modes"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
        kernel_size=CONFIG["kernel_size"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    best_epoch = 1

    for epoch in range(CONFIG["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
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

    test_metrics = compute_metrics(test_trues.flatten(), test_preds.flatten(), peak_threshold_kw)
    val_metrics  = compute_metrics(val_trues.flatten(), val_preds.flatten(), peak_threshold_kw)

    steps_to_eval = [0, 5, 11, 47]
    step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}
    per_step_metrics = {}
    for step in steps_to_eval:
        step_m = compute_metrics(test_trues[:, step], test_preds[:, step], peak_threshold_kw)
        per_step_metrics[step_labels[step]] = {k: (float(v) if not np.isnan(v) else None) for k, v in step_m.items()}

    mae_48 = [float(mean_absolute_error(test_trues[:, s], test_preds[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(test_trues[:, s], test_preds[:, s]))) for s in range(HORIZON)]

    return {
        "model_weights": best_weights,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "test_preds": test_preds,
        "test_trues": test_trues,
        "epochs": list(range(1, len(train_loss_history) + 1)),
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "per_step_metrics": per_step_metrics,
        "step_48_metrics": {
            "mae": mae_48,
            "rmse": rmse_48
        },
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

if __name__ == '__main__':
    print("=" * 70)
    print(f"Starting Multi-Seed Benchmark for {MODEL_NAME} across {len(SEEDS)} seeds")
    print("=" * 70)

    output_json_filename = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_results.json")
    output_pt_filename   = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best.pt")
    output_npz_filename  = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_predictions.npz")
    root_json_filename   = f"{MODEL_NAME}_results.json"
    root_pt_filename     = f"{MODEL_NAME}_best.pt"
    root_npz_filename    = f"{MODEL_NAME}_predictions.npz"

    results_data = {
        "model_name": MODEL_NAME,
        "seeds": {},
        "summary": {}
    }
    all_seed_metrics = []
    all_predictions = {}
    best_overall_val_loss = float('inf')
    best_seed_id = None
    best_overall_weights = None
    total_model_parameters = None

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n=========================================================================")
        print(f"RUNNING SEED {seed} ({seed_idx}/{len(SEEDS)}) — {MODEL_NAME}")
        print(f"=========================================================================")
        start_t = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        res = run_seed(seed)
        elapsed = round(time.time() - start_t, 2)
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device.type == 'cuda' else 0.0

        if total_model_parameters is None:
            total_model_parameters = res["total_params"]
            results_data["total_parameters"] = total_model_parameters
            print(f"Model Trainable Parameters: {total_model_parameters:,}")

        metrics = res["test_metrics"]
        metrics["training_time_seconds"] = elapsed
        metrics["peak_gpu_memory_mb"] = peak_vram_mb
        all_seed_metrics.append(metrics)

        all_predictions[f"seed_{seed}"] = res["test_preds"].astype(np.float32)

        print(f"Seed {seed} Completed in {elapsed:.1f}s | Val Loss: {res['best_val_loss']:.5f} | Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}, WAPE: {metrics['wape']:.2f}%")

        if res["best_val_loss"] < best_overall_val_loss and res["model_weights"] is not None:
            best_overall_val_loss = res["best_val_loss"]
            best_seed_id = seed
            best_overall_weights = res["model_weights"]
            torch.save(best_overall_weights, output_pt_filename)
            torch.save(best_overall_weights, root_pt_filename)
            results_data["best_seed"] = int(seed)
            print(f"  [Checkpoint] New overall best model saved from SEED {seed} (Val Loss: {best_overall_val_loss:.6f}) -> {output_pt_filename}")

        results_data["seeds"][str(seed)] = {
            "training_time_seconds": elapsed,
            "peak_gpu_memory_mb": peak_vram_mb,
            "epochs": res["epochs"],
            "train_loss": res["train_loss"],
            "val_loss": res["val_loss"],
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "overall_metrics": {k: (float(v) if not np.isnan(v) else None) for k, v in metrics.items()},
            "per_step_metrics": res["per_step_metrics"],
            "step_48_metrics": res["step_48_metrics"]
        }

        # Save incremental results
        with open(output_json_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
        with open(root_json_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)

    # Final Ensemble & Summary
    test_trues = res["test_trues"]
    all_predictions["y_true"] = test_trues.astype(np.float32)
    pred_stack = np.stack([all_predictions[f"seed_{s}"] for s in SEEDS], axis=0)
    all_predictions["pred_mean"] = np.mean(pred_stack, axis=0).astype(np.float32)
    all_predictions["pred_std"]  = np.std(pred_stack,  axis=0).astype(np.float32)

    np.savez_compressed(output_npz_filename, **all_predictions)
    np.savez_compressed(root_npz_filename, **all_predictions)
    print(f"Successfully saved all seed predictions to {output_npz_filename} and {root_npz_filename}")

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — {MODEL_NAME}")
    print("=" * 70)
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

    results_data["config"] = CONFIG
    results_data["summary"] = summary_dict

    with open(output_json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    with open(root_json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nArtifacts successfully saved to {OUTPUT_DIR} and current directory")
    print(f"Finished running all {len(SEEDS)} SEEDs for {MODEL_NAME}!")
