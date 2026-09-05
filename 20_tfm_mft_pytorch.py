#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Model 20: Multi-scale Fusion Transformer (MFT) PyTorch Implementation
# Reference: Liu et al., "Multi-scale fusion transformer for EV charging station load prediction",
#            Nature Scientific Reports (2026) 16:8609. https://doi.org/10.1038/s41598-026-38562-z
#
# Architectural Innovations:
# 1. 3M (Multi-scale Modeling Mechanism):
#    Predefined scale masks constrain individual attention heads to model temporal
#    dependencies at distinct granularities in parallel (fine-to-coarse strides: m_n^{a,b} = 0 if (a-b)%n == 0 else -inf).
# 2. FAM (Feature-correlation Analysis Module):
#    Computes static base prior weights from Pearson Correlation Coefficients (PCC)
#    between external covariates and target EV load on the training set (strictly zero test leakage).
# 3. MFM (Multi-variable Fusion Module):
#    Dynamic sample-level cross-attention feature reweighting with target load as Query
#    and external features as Key/Value, combining FAM prior weights with dynamic attention scores.
# 4. Hybrid Decoder:
#    Fuses multi-scale temporal representation R and external fusion representation E
#    into an LSTM recurrent decoder with Dual Feature Aggregation (Last Step + Global Pooling).
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
    # Fallback to local path if running from different subdirectory
    data_path = 'data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # Chronological order enforcement

# Drop proven noise features (Invariant #2)
df = df.drop(columns=['prcp', 'tempDiff_48', 'cldc'], errors='ignore')

cols = []
for col in df.columns:
    df[col] = df[col].astype('float32')
    if col != 'kWhDelivered':
        cols.append(col)

X = df[cols]
y = df['kWhDelivered']

print(f"Dataset Loaded from {data_path}! Rows: {len(df)}, External Features: {len(cols)}")

# Chronological Split: 60% Train, 20% Val, 20% Test
train_len = int(len(df) * 0.6)
val_len   = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]
X_test  = X[train_len + val_len :]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

# Feature Scaling: Fit strictly on Train
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# Target Scaling: Fit strictly on Train
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Append target load as the last column so sequence windows contain both past load and covariates
TARGET_CH_IDX = X_train_scaled.shape[1]  # index 28 (total channels: 29)
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
X_test_scaled  = np.concatenate([X_test_scaled,  y_test_scaled.reshape(-1, 1)], axis=1)
print(f"Target load channel appended at index {TARGET_CH_IDX} (Total Channels: {X_train_scaled.shape[1]})")

# Peak Load Threshold (Top 20% of Train kWhDelivered)
peak_threshold_kw = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ---------------------------------------------------------
# 2. Feature-correlation Analysis Module (FAM) Base Weights
# ---------------------------------------------------------
def compute_fam_base_weights(X_train_ext, y_train_arr):
    """
    Computes static base prior weights w_i from Pearson Correlation Coefficients (PCC)
    between each external feature and target load strictly on the training set (Eq. 5 & 6, Liu et al. 2026).
    """
    num_ext = X_train_ext.shape[1]
    corrs = np.zeros(num_ext, dtype=np.float32)
    y_mean = float(np.mean(y_train_arr))
    y_std = float(np.std(y_train_arr)) + 1e-8

    for i in range(num_ext):
        x_i = X_train_ext[:, i]
        x_mean = float(np.mean(x_i))
        x_std = float(np.std(x_i)) + 1e-8
        cov = float(np.mean((x_i - x_mean) * (y_train_arr - y_mean)))
        corrs[i] = cov / (x_std * y_std)

    # Softmax over PCC values as defined in Eq. (6)
    corrs_exp = np.exp(corrs - np.max(corrs))
    w = corrs_exp / np.sum(corrs_exp)
    return torch.tensor(w, dtype=torch.float32)

# Compute FAM base weights from training data only
fam_base_weights = compute_fam_base_weights(X_train_scaled[:, :TARGET_CH_IDX], y_train_scaled)
print(f"FAM base weights initialized for {len(fam_base_weights)} external features. Min: {fam_base_weights.min():.4f}, Max: {fam_base_weights.max():.4f}")

# ---------------------------------------------------------
# 3. Windowing & Metric Helpers
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
# 4. Multi-scale Fusion Transformer (MFT) Architecture
# ---------------------------------------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Encoding (Vaswani et al., 2017; Eq. 2 in Liu et al., 2026)."""
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, seq_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ScaleMaskedAttention(nn.Module):
    """
    Scale-Masked Multi-Head Attention (3M Module, Eq. 3 & 4 in Liu et al., 2026).
    Predefined scale masks constrain each head n in {1, ..., N} to temporal stride n:
    m_n^{a,b} = 0 if (a - b) % n == 0 else -1e9.
    """
    def __init__(self, d_model, num_heads, seq_len, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Precompute scale masks for all heads: [1, num_heads, seq_len, seq_len]
        # Head index h in 0..num_heads-1 corresponds to scale stride s = h + 1
        diff = (torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)).abs()
        masks = torch.full((1, num_heads, seq_len, seq_len), -1e9, dtype=torch.float32)
        for h in range(num_heads):
            stride = h + 1
            masks[0, h][diff % stride == 0] = 0.0
        self.register_buffer('scale_masks', masks)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)       # [B, num_heads, L, L]
        scores = scores + self.scale_masks[:, :, :L, :L]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)                                            # [B, num_heads, L, d_k]
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)   # [B, L, d_model]
        return self.out_proj(context)

class MFTEncoderLayer(nn.Module):
    """Single 3M Transformer Encoder Layer with Scale-Masked Attention + FFN."""
    def __init__(self, d_model, num_heads, seq_len, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attn = ScaleMaskedAttention(d_model, num_heads, seq_len, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.attn(x)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

class MultiVariableFusionModule(nn.Module):
    """
    MFM (Multi-variable Fusion Module, Eq. 7, 8, 9 in Liu et al., 2026).
    Integrates FAM static base weights with dynamic cross-attention scores where
    historical load acts as Query and external features act as Key and Value.
    """
    def __init__(self, num_ext_features, d_model, base_weights=None, dropout_rate=0.1):
        super().__init__()
        self.num_ext_features = num_ext_features
        self.d_model = d_model

        if base_weights is None:
            base_weights = torch.ones(num_ext_features, dtype=torch.float32) / max(num_ext_features, 1)
        elif not isinstance(base_weights, torch.Tensor):
            base_weights = torch.tensor(base_weights, dtype=torch.float32)
        self.register_buffer('base_weights', base_weights)

        self.load_q_proj = nn.Linear(1, d_model)
        self.load_pool   = nn.Linear(d_model, d_model)

        # External feature Key and Value temporal projections
        self.feat_k_proj = nn.Linear(1, d_model)
        self.feat_v_proj = nn.Linear(1, d_model)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, load_seq, ext_features):
        """
        load_seq: [B, L, 1]
        ext_features: [B, L, F]
        Returns E: [B, L, d_model]
        """
        B, L, F = ext_features.shape

        # 1. Load Query representation
        l_emb = self.load_q_proj(load_seq)                             # [B, L, d_model]
        l_q = self.load_pool(l_emb.mean(dim=1))                        # [B, d_model]

        # 2. External Features Key & Value representations
        ext_perm = ext_features.permute(0, 2, 1).unsqueeze(-1)         # [B, F, L, 1]
        feat_k = self.feat_k_proj(ext_perm)                            # [B, F, L, d_model]
        feat_v = self.feat_v_proj(ext_perm)                            # [B, F, L, d_model]
        feat_k_pooled = feat_k.mean(dim=2)                             # [B, F, d_model]

        # 3. Dynamic Attention Weights (Cross-attention between load query and feature keys)
        scores = torch.bmm(l_q.unsqueeze(1), feat_k_pooled.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model) # [B, F]
        sigma_w = F.softmax(scores, dim=-1)                            # [B, F]

        # 4. Total Dynamic Weight = FAM base weight + MFM attention weight (Eq. 8)
        if self.base_weights.shape[0] == F:
            base_w = self.base_weights
        else:
            base_w = torch.ones(F, device=ext_features.device, dtype=ext_features.dtype) / max(F, 1)
        w_tilde = base_w.unsqueeze(0) + sigma_w                        # [B, F]

        # 5. Weighted aggregation of external feature representations (Eq. 9)
        w_expanded = w_tilde.unsqueeze(-1).unsqueeze(-1)               # [B, F, 1, 1]
        E = torch.sum(w_expanded * feat_v, dim=1)                      # [B, L, d_model]
        E = self.leaky_relu(E)
        E = self.dropout(E)
        return E

class MFTModel(nn.Module):
    """
    Complete Multi-scale Fusion Transformer (MFT).
    Combines 3M Multi-scale Modeling, FAM base weights, MFM dynamic fusion,
    and a recurrent LSTM Decoder with Dual Feature Aggregation.
    """
    def __init__(self, lookback=96, num_features=29, horizon=48,
                 target_idx=None, base_weights=None,
                 d_model=64, num_heads=4, d_ff=128, num_layers=2,
                 decoder_hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model

        if target_idx is None:
            target_idx = num_features - 1
        self.target_idx = target_idx

        num_ext_features = max(num_features - 1, 1)

        if base_weights is None:
            base_weights = torch.ones(num_ext_features, dtype=torch.float32) / max(num_ext_features, 1)
        elif not isinstance(base_weights, torch.Tensor):
            base_weights = torch.tensor(base_weights, dtype=torch.float32)

        # 1. 3M Multi-scale Modeling Mechanism (Target Load Pipeline)
        self.load_emb = nn.Linear(1, d_model)
        self.pos_emb  = SinusoidalPositionalEmbedding(lookback, d_model)
        self.encoder_layers = nn.ModuleList([
            MFTEncoderLayer(d_model=d_model, num_heads=num_heads, seq_len=lookback,
                            d_ff=d_ff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])

        # 2. MFM Multi-variable Fusion Module (External Features Pipeline)
        self.mfm = MultiVariableFusionModule(
            num_ext_features=num_ext_features,
            d_model=d_model,
            base_weights=base_weights,
            dropout_rate=dropout_rate
        )

        # 3. Hybrid Recurrent Decoder (Concat R and E: 2 * d_model)
        fused_dim = 2 * d_model
        self.decoder_lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 4. Dual Feature Aggregation Prediction Head
        self.head_fc1 = nn.Linear(decoder_hidden_dim * 2, 128)
        self.head_drop1 = nn.Dropout(dropout_rate)
        self.head_fc2 = nn.Linear(128, 64)
        self.head_drop2 = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(64, horizon)

    def forward(self, x):
        # x: [B, lookback, num_features]
        # Separate target load series and external feature covariates
        load_seq = x[:, :, self.target_idx : self.target_idx + 1]       # [B, L, 1]
        ext_features = torch.cat([
            x[:, :, :self.target_idx],
            x[:, :, self.target_idx + 1:]
        ], dim=-1)                                                      # [B, L, F]

        # 1. 3M Multi-scale Temporal Representation (R)
        h = self.pos_emb(self.load_emb(load_seq))                       # [B, L, d_model]
        for layer in self.encoder_layers:
            h = layer(h)
        R = h                                                           # [B, L, d_model]

        # 2. MFM External Feature Fusion Representation (E)
        E = self.mfm(load_seq, ext_features)                           # [B, L, d_model]

        # 3. Fuse R and E along feature dimension: [B, L, 2 * d_model]
        fused = torch.cat([R, E], dim=-1)

        # 4. Decoder LSTM
        lstm_out, _ = self.decoder_lstm(fused)                          # [B, L, decoder_hidden_dim]

        # 5. Dual Feature Aggregation (Last Step + Global Average Pooling)
        last_step = lstm_out[:, -1, :]
        global_avg = torch.mean(lstm_out, dim=1)
        ctx = torch.cat([last_step, global_avg], dim=-1)                # [B, 2 * decoder_hidden_dim]

        # 6. Forecasting Head
        head = F.relu(self.head_fc1(ctx))
        head = self.head_drop1(head)
        head = F.relu(self.head_fc2(head))
        head = self.head_drop2(head)
        out = self.out_proj(head)                                       # [B, horizon]
        return out

# ---------------------------------------------------------
# 5. Experiment Configuration & Pre-building Tensors
# ---------------------------------------------------------
LOOKBACK   = 96
HORIZON    = 48
BATCH_SIZE = 128
SEEDS      = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]

MODEL_NAME = "20_tfm_mft_pytorch"
output_dir = os.path.join("outputs", MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)

output_json_filename = os.path.join(output_dir, f"{MODEL_NAME}_results.json")
output_pt_filename   = os.path.join(output_dir, f"{MODEL_NAME}_best.pt")
output_npz_filename  = os.path.join(output_dir, f"{MODEL_NAME}_predictions.npz")

root_json_filename   = f"{MODEL_NAME}_results.json"

results_data = {
    "model_name": MODEL_NAME,
    "seeds": {},
    "summary": {}
}

steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print("\nPre-building sequence tensors...")
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t,   y_val_t,   _, _ = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)
X_test_t,  y_test_t,  X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

print(f"Train Tensors: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")
print(f"Starting {len(SEEDS)}-Seed Benchmark Loop for Multi-scale Fusion Transformer (MFT)...")

# ---------------------------------------------------------
# 6. Multi-Seed Training & Benchmark Execution Loop
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
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)}) — {MODEL_NAME}")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    model = MFTModel(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
        target_idx=TARGET_CH_IDX,
        base_weights=fam_base_weights,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        decoder_hidden_dim=64,
        dropout_rate=0.10
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if seed_idx == 1:
        print(f"MFT Model Trainable Parameters: {total_params:,}")
        results_data["total_parameters"] = total_params

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    epochs = 200
    patience = 15
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history   = []
    best_epoch = 1
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
            best_epoch       = epoch
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
    if best_model_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    # Test set evaluation
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

    # Overall metrics across horizon
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

    mae_48 = [float(mean_absolute_error(y_true_kw[:, s], y_pred_kw[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_true_kw[:, s], y_pred_kw[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_kw.astype(np.float32)

    # Save best overall checkpoint across all seeds
    if best_val_loss < best_overall_val_loss and best_model_weights is not None:
        best_overall_val_loss = best_val_loss
        best_seed_id = SEED
        torch.save(best_model_weights, output_pt_filename)
        torch.save(best_model_weights, f"{MODEL_NAME}_best.pt")
        results_data["best_seed"] = int(SEED)
        print(f"  [Checkpoint] New overall best model saved from SEED {SEED} (Val Loss: {best_val_loss:.6f}) -> {output_pt_filename}")

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

    # Save incremental JSON after each seed
    with open(output_json_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    with open(root_json_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Successfully saved SEED {SEED} results to {output_json_filename} (Runtime: {seed_duration}s)")

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# 7. Final Summary & Predictions Serialization
# ---------------------------------------------------------
all_predictions["y_true"] = y_true_kw.astype(np.float32)
available_seeds = [s for s in SEEDS if f"seed_{s}" in all_predictions]
if available_seeds:
    pred_stack = np.stack([all_predictions[f"seed_{s}"] for s in available_seeds], axis=0)
    all_predictions["pred_mean"] = np.mean(pred_stack, axis=0).astype(np.float32)
    all_predictions["pred_std"]  = np.std(pred_stack,  axis=0).astype(np.float32)
else:
    all_predictions["pred_mean"] = np.zeros_like(y_true_kw, dtype=np.float32)
    all_predictions["pred_std"]  = np.zeros_like(y_true_kw, dtype=np.float32)

np.savez_compressed(output_npz_filename, **all_predictions)
np.savez_compressed(f"{MODEL_NAME}_predictions.npz", **all_predictions)
print(f"Successfully saved all seed predictions to {output_npz_filename} and {MODEL_NAME}_predictions.npz")

print(f"\n{'='*70}")
print(f"FINAL SUMMARY ACROSS {len(available_seeds)} SEEDS — {MODEL_NAME}")
print(f"{'='*70}")
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
    "model": "MFT (Multi-scale Fusion Transformer)",
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "batch_size": BATCH_SIZE,
    "seeds": SEEDS,
    "d_model": 64,
    "num_heads": 4,
    "d_ff": 128,
    "num_layers": 2,
    "decoder_hidden_dim": 64,
    "dropout_rate": 0.10,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "total_parameters": results_data.get("total_parameters", None)
}
results_data["summary"] = summary_dict

with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
with open(root_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)

print(f"\nSuccessfully saved final results to {output_json_filename} and {root_json_filename}")
