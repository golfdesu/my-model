#!/usr/bin/env python
# coding: utf-8

# # 00_tfm_custom_jpn_no_weather.py
# Model 00 (Inverted Variate-Centric Custom Transformer + Orthogonal Regularization)
# Target Dataset: ACN-JPN / JPL without Weather (acn_jpl_ready.csv)
# 10-Seed Multi-Seed Production Benchmark ([42, 123, 456, ..., 9999])
# Parameters: JPN Full HPO Selected Configuration (Trial 26)

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import gc
import json
import time
import shutil
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
torch.set_num_threads(min(16, num_cpus))
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

# ==============================================================================
# 1. Dataset Loading & Preprocessing (JPN without Weather)
# ==============================================================================
data_path = '../data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = 'data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = '../data_cleaned/acn_jpn_ready.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # safety: enforce chronological order before time-based split

# Weather ablation: drop all weather variables while retaining load and calendar features
weather_cols = ['temp', 'rhum', 'prcp', 'wspd', 'pres', 'cldc', 'apparent_temp', 'tempDiff_48', 'tempMean_48']
df = df.drop(columns=weather_cols, errors='ignore')

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
val_len   = int(len(df) * 0.2)

X_train = X[:train_len]
X_val   = X[train_len : train_len + val_len]
X_test  = X[train_len + val_len :]

y_train = y[:train_len]
y_val   = y[train_len : train_len + val_len]
y_test  = y[train_len + val_len :]

# Feature Scaling (MinMaxScaler on Train only)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# Target Scaling (MinMaxScaler for y on Train only)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Inverted Transformer Tokenization: Target variate appended as an input variate token
TARGET_CH_IDX = X_train_scaled.shape[1]
X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
X_test_scaled  = np.concatenate([X_test_scaled,  y_test_scaled.reshape(-1, 1)], axis=1)
print(f"Target variate appended at index {TARGET_CH_IDX} (Total Variate Tokens: {X_train_scaled.shape[1]})")

# EEO Feature Partition: Without weather, all variates are endogenous
exo_indices = []
endo_indices = list(range(len(cols))) + [TARGET_CH_IDX]
print(f"  -> EEO Subspace Partition: {len(endo_indices)} Endogenous Variates, 0 Exogenous Weather Variates (No-Weather Ablation)")

# Compute Peak Load Threshold (Top 20% of TRAIN in actual kW)
peak_threshold_kw = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))
print(f"Peak Load Threshold (Top 20% of TRAIN): {peak_threshold_kw:.4f} kW")

# ==============================================================================
# 2. Sequence Windowing Helpers
# ==============================================================================
def create_windowed_tensors(X_data, y_data, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X_seq.append(X_data[i : i + lookback])
        y_seq.append(y_data[i + lookback : i + lookback + horizon])
    X_t = torch.tensor(np.array(X_seq, dtype=np.float32))
    y_t = torch.tensor(np.array(y_seq, dtype=np.float32))
    return X_t, y_t, np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

# ==============================================================================
# 3. Model Architecture (Inverted Variate-Centric Transformer)
# ==============================================================================
class PositionalEmbedding(nn.Module):
    """Sinusoidal Positional Encoding (Vaswani et al., NIPS 2017, Sec 3.5)"""
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019)"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FastMHA(nn.Module):
    """
    Multi-Head Attention using PyTorch 2.0+ scaled_dot_product_attention.
    Separated Q, K, V projections to prevent Adam momentum coupling across orthogonal subspaces.
    """
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, query, key=None, value=None, is_causal=False):
        B, S_q, _ = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S_k = key.shape[1]

        q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)

        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop_p, is_causal=is_causal)
        out = attn_out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        return self.out_proj(out)


class InvertedTransformerLayer(nn.Module):
    """Pre-LN RMSNorm Inverted Self-Attention Layer across Variate Tokens"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = FastMHA(d_model, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm)
        x = x + self.drop1(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x


class InvertedCustomTransformer(nn.Module):
    """
    Model 00 (Proposed Custom Architecture - Direction 1):
    Inverted Variate-Centric Transformer with Endogenous-Exogenous Subspace Disentanglement
    and Vectorized Batched Orthogonal Regularization.
    """
    def __init__(self, lookback, num_variates, horizon, target_ch_idx,
                 d_model=128, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.15,
                 endo_indices=None, exo_indices=None):
        super().__init__()
        self.lookback = lookback
        self.num_variates = num_variates
        self.horizon = horizon
        self.target_ch_idx = target_ch_idx
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.use_disentangled_proj = (endo_indices is not None and exo_indices is not None and len(exo_indices) > 0)
        self.endo_indices = endo_indices
        self.exo_indices = exo_indices

        if self.use_disentangled_proj:
            self.endo_proj = nn.Linear(lookback, d_model)
            self.exo_proj  = nn.Linear(lookback, d_model)
        else:
            self.variate_proj = nn.Linear(lookback, d_model)

        self.pos_emb = PositionalEmbedding(num_variates, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([
            InvertedTransformerLayer(d_model, num_heads, d_ff, dropout=dropout_rate)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        self.readout_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, horizon)
        )

        self.all_attn_layers = [layer.self_attn for layer in self.layers]

    def forward(self, x):
        x_inv = x.transpose(1, 2)
        B, N, L = x_inv.shape

        if self.use_disentangled_proj:
            tokens = torch.zeros(B, N, self.d_model, device=x.device, dtype=x.dtype)
            tokens[:, self.endo_indices, :] = self.endo_proj(x_inv[:, self.endo_indices, :])
            tokens[:, self.exo_indices, :]  = self.exo_proj(x_inv[:, self.exo_indices, :])
        else:
            tokens = self.variate_proj(x_inv)

        tokens = self.pos_emb(tokens)
        tokens = self.dropout(tokens)

        for layer in self.layers:
            tokens = layer(tokens)

        tokens = self.final_norm(tokens)

        target_token = tokens[:, self.target_ch_idx, :]
        global_context = tokens.mean(dim=1)
        combined = torch.cat([target_token, global_context], dim=-1)
        out = self.readout_head(combined)
        return out

    def compute_vectorized_orthogonal_penalty(self, strength=1e-5, inter_head_strength=1e-5, eeo_strength=1e-5):
        ref_device = self.endo_proj.weight.device if self.use_disentangled_proj else self.variate_proj.weight.device
        if strength <= 0.0 and inter_head_strength <= 0.0 and eeo_strength <= 0.0:
            return torch.tensor(0.0, device=ref_device)

        q_weights = [layer.q_proj.weight for layer in self.all_attn_layers]
        k_weights = [layer.k_proj.weight for layer in self.all_attn_layers]
        v_weights = [layer.v_proj.weight for layer in self.all_attn_layers]
        out_weights = [layer.out_proj.weight for layer in self.all_attn_layers]

        stacked_qk = torch.stack(q_weights + k_weights, dim=0)
        stacked_v_out = torch.stack(v_weights + out_weights, dim=0)
        all_weights = torch.cat([stacked_qk, stacked_v_out], dim=0)

        penalty = torch.tensor(0.0, device=all_weights.device)

        # 1. Batched Intra-Matrix Isometry
        if strength > 0.0:
            wt_w = torch.bmm(all_weights.transpose(1, 2), all_weights)
            eye = torch.eye(self.d_model, device=all_weights.device).unsqueeze(0)
            penalty = penalty + strength * torch.sum((wt_w - eye) ** 2)

        # 2. Batched Inter-Head Diversity
        num_qk = stacked_qk.size(0)
        if inter_head_strength > 0.0 and self.num_heads > 1:
            w_heads_flat = stacked_qk.view(num_qk, self.num_heads, -1)
            head_norms = torch.norm(w_heads_flat, dim=-1, keepdim=True) + 1e-8
            norm_gram = torch.bmm(w_heads_flat, w_heads_flat.transpose(1, 2)) / (head_norms * head_norms.transpose(1, 2))
            off_diag = norm_gram - torch.eye(self.num_heads, device=all_weights.device).unsqueeze(0)
            inter_loss = torch.sum(off_diag ** 2) / (self.num_heads * (self.num_heads - 1))
            penalty = penalty + inter_head_strength * inter_loss

        # 3. Batched EEO Cross-Subspace Orthogonality
        if eeo_strength > 0.0 and self.use_disentangled_proj:
            w_endo = F.normalize(self.endo_proj.weight, p=2, dim=-1)
            w_exo  = F.normalize(self.exo_proj.weight, p=2, dim=-1)
            cross_cos = torch.mm(w_endo, w_exo.t())
            eeo_loss = torch.sum(cross_cos ** 2) / (self.d_model * self.d_model)
            penalty = penalty + eeo_strength * eeo_loss

        return penalty


# ==============================================================================
# 4. Metrics Evaluator Helper
# ==============================================================================
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

    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape,
                mae_peak=mae_peak, wape_peak=wape_peak, bias=bias, negative_pct=negative_pct)


# ==============================================================================
# 5. Configuration & Multi-Seed Benchmark
# ==============================================================================
LOOKBACK = 96
HORIZON  = 48
BATCH_SIZE = 64
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]

# Selected Hyperparameters from JPN Full HPO (Trial 26)
D_MODEL             = 128
NUM_HEADS           = 4
D_FF                = 256
NUM_LAYERS          = 2
DROPOUT_RATE        = 0.15
LEARNING_RATE       = 0.0015619559771917156
WEIGHT_DECAY        = 3.378567993925436e-06
PATIENCE            = 15
LR_SCHEDULER_PATIENCE = 5

# Custom Regularization Hyperparameters (EEO is 0.0 because no weather features)
ATTN_ORTHOGONAL_REG       = 0.002356491752796512
INTER_HEAD_ORTHOGONAL_REG = 8.65923378099159e-05
EEO_ORTHOGONAL_REG        = 0.0

OUTPUT_STEM = "00_tfm_custom_pytorch"
output_json_filename = f"{OUTPUT_STEM}_results.json"
best_model_filename  = f"{OUTPUT_STEM}_best.pt"
predictions_filename = f"{OUTPUT_STEM}_predictions.npz"
output_archive_dir   = "outputs/acn_jpn_no_weather/00_tfm_custom_pytorch"

results_data = {
    "model_name": "00_tfm_custom_pytorch",
    "architecture_paradigm": "inverted_variate_centric_transformer",
    "base_model": "07_tfm_itfm_pytorch",
    "version": "v7_direction1",
    "active_custom_features": [
        "inverted_variate_tokenization",
        "pre_ln_rmsnorm_backbone",
        "fast_sdpa_fused_attention",
        "independent_qkv_subspace_projections",
        "vectorized_batched_orthogonal_regularization",
        "dual_context_readout_head",
        "zero_copy_gpu_resident_tensors"
    ],
    "attn_orthogonal_reg_strength": ATTN_ORTHOGONAL_REG,
    "inter_head_orthogonal_reg_strength": INTER_HEAD_ORTHOGONAL_REG,
    "eeo_orthogonal_reg_strength": EEO_ORTHOGONAL_REG,
    "num_endo_features": len(endo_indices),
    "num_exo_features": 0,
    "dataset": "acn_jpl_ready",
    "with_weather": False,
    "seeds": {},
    "summary": {}
}

all_seed_metrics = []
all_predictions = {}
best_overall_val_loss = float("inf")
best_seed_id = None
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print('Pre-building sequence tensors...')
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t, _, _     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)
X_test_t, y_test_t, X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

X_train_dev = X_train_t.to(device)
y_train_dev = y_train_t.to(device)
X_val_dev   = X_val_t.to(device)
y_val_dev   = y_val_t.to(device)
X_test_dev  = X_test_t.to(device)

n_train = X_train_dev.size(0)
n_val   = X_val_dev.size(0)
n_batches_train = n_train // BATCH_SIZE

print(f"Pre-loaded tensors to {device}! Memory Footprint: Train={X_train_dev.element_size() * X_train_dev.nelement() / (1024**2):.1f}MB")
print(f"Starting Automated {len(SEEDS)}-Seed Loop for Model 00 (JPN No Weather)...")

for seed_idx, SEED in enumerate(SEEDS, 1):
    seed_start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\n=========================================================================")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"=========================================================================")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    model = InvertedCustomTransformer(
        lookback=LOOKBACK,
        num_variates=X_train_scaled.shape[1],
        horizon=HORIZON,
        target_ch_idx=TARGET_CH_IDX,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        endo_indices=endo_indices,
        exo_indices=exo_indices
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results_data["total_parameters"] = total_params
    if seed_idx == 1:
        print(f"Model Parameters: {total_params:,}")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_weights = None
    train_loss_history, val_loss_history = [], []

    epoch_pbar = tqdm(range(1, 101), desc=f"Seed {SEED}", unit="epoch", leave=False)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        perm = torch.randperm(n_train, device=device)
        for b_i in range(n_batches_train):
            idx = perm[b_i * BATCH_SIZE : (b_i + 1) * BATCH_SIZE]
            batch_X = X_train_dev[idx]
            batch_y = y_train_dev[idx]

            optimizer.zero_grad(set_to_none=True)
            out = model(batch_X)

            mse_loss = criterion(out, batch_y)
            ortho_loss = model.compute_vectorized_orthogonal_penalty(
                strength=ATTN_ORTHOGONAL_REG,
                inter_head_strength=INTER_HEAD_ORTHOGONAL_REG,
                eeo_strength=EEO_ORTHOGONAL_REG
            )
            loss = mse_loss + ortho_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += mse_loss.item() * BATCH_SIZE

        train_loss /= (n_batches_train * BATCH_SIZE)

        # Validation
        model.eval()
        val_loss = 0.0
        eval_batch_size = 1024
        with torch.inference_mode():
            for v_i in range(0, n_val, eval_batch_size):
                batch_X = X_val_dev[v_i : v_i + eval_batch_size]
                batch_y = y_val_dev[v_i : v_i + eval_batch_size]
                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= n_val
        scheduler.step(val_loss)

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        epoch_pbar.set_postfix({'train_loss': f"{train_loss:.5f}", 'val_loss': f"{val_loss:.5f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Evaluation on Test Split
    if best_model_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
    model.eval()

    test_preds = []
    with torch.inference_mode():
        for t_i in range(0, X_test_dev.size(0), 1024):
            batch_X = X_test_dev[t_i : t_i + 1024]
            out = model(batch_X)
            test_preds.append(out.cpu())
    y_pred_scaled = torch.cat(test_preds, dim=0).numpy()

    # Inverse transform
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, HORIZON)
    y_test_seq_unscaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(-1, HORIZON)

    overall_metrics = compute_metrics(y_test_seq_unscaled.reshape(-1), y_pred_unscaled.reshape(-1), peak_threshold_kw)
    seed_duration = round(time.time() - seed_start_time, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device.type == 'cuda' else 0.0
    overall_metrics["training_time_seconds"] = seed_duration
    overall_metrics["peak_gpu_memory_mb"] = peak_vram_mb
    all_seed_metrics.append(overall_metrics)

    per_step_metrics = {}
    for step in steps_to_eval:
        m = compute_metrics(y_test_seq_unscaled[:, step], y_pred_unscaled[:, step], peak_threshold_kw)
        per_step_metrics[step_labels[step]] = {k: (float(v) if not np.isnan(v) else None) for k, v in m.items()}

    mae_48 = [float(mean_absolute_error(y_test_seq_unscaled[:, s], y_pred_unscaled[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_test_seq_unscaled[:, s], y_pred_unscaled[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_unscaled.astype(np.float32)

    if best_val_loss < best_overall_val_loss and best_model_weights is not None:
        best_overall_val_loss = best_val_loss
        best_seed_id = SEED
        torch.save(best_model_weights, best_model_filename)
        results_data["best_seed"] = int(SEED)
        print(f"  [Checkpoint] New overall best model saved from SEED {SEED} (Val Loss: {best_val_loss:.6f}) -> {best_model_filename}")

    results_data["seeds"][str(SEED)] = {
        "training_time_seconds": seed_duration,
        "peak_gpu_memory_mb": peak_vram_mb,
        "epochs": list(range(1, len(train_loss_history) + 1)),
        "train_loss": [float(v) for v in train_loss_history],
        "val_loss": [float(v) for v in val_loss_history],
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "overall_metrics": {k: (float(v) if not np.isnan(v) else None) for k, v in overall_metrics.items()},
        "per_step_metrics": per_step_metrics,
        "step_48_metrics": {
            "mae": mae_48,
            "rmse": rmse_48
        }
    }
    with open(output_json_filename, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved SEED {SEED} results to {output_json_filename} (MAE: {overall_metrics['mae']:.4f}, Runtime: {seed_duration}s)")
    gc.collect()

all_predictions["y_true"] = y_test_seq_unscaled.astype(np.float32)
pred_stack = np.stack([all_predictions[f"seed_{s}"] for s in SEEDS], axis=0)
all_predictions["pred_mean"] = np.mean(pred_stack, axis=0).astype(np.float32)
all_predictions["pred_std"] = np.std(pred_stack, axis=0).astype(np.float32)
np.savez_compressed(predictions_filename, **all_predictions)
print(f"Saved all seed predictions to {predictions_filename}")

# Final Summary Across 10 Seeds
print(f"\n======================================================================")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — 00_tfm_custom_pytorch (JPN No Weather)")
print(f"======================================================================")
summary_dict = {}
metric_keys = ['mae', 'rmse', 'r2', 'wape', 'mape', 'bias', 'negative_pct', 'training_time_seconds', 'peak_gpu_memory_mb']
for k in metric_keys:
    vals = [m[k] for m in all_seed_metrics if k in m and not np.isnan(m[k])]
    if vals:
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        print(f"  {k.upper():<22}: {mu:.4f} ± {sigma:.4f}")
        summary_dict[k] = {"mean": mu, "std": sigma}

results_data["summary"] = summary_dict
with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)

# Automatically archive to outputs directory
os.makedirs(output_archive_dir, exist_ok=True)
shutil.copyfile(output_json_filename, os.path.join(output_archive_dir, output_json_filename))
if os.path.exists(best_model_filename):
    shutil.copyfile(best_model_filename, os.path.join(output_archive_dir, best_model_filename))
if os.path.exists(predictions_filename):
    shutil.copyfile(predictions_filename, os.path.join(output_archive_dir, predictions_filename))
print(f"\nAll artifacts successfully mirrored to: {output_archive_dir}/")
