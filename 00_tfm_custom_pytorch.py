#!/usr/bin/env python
# coding: utf-8

# # 00_tfm_custom_pytorch.py
# Model 00 (Proposed): Inverted Variate-Centric Custom Transformer in PyTorch (L=96, H=48)
# Key Architectural Mechanisms:
# - Inverted Variate Tokenization: Each variate time series is projected into a d_model token
# - Endogenous-Exogenous Subspace Disentanglement: Dual input projection layers (endo_proj & exo_proj)
# - Vectorized Batched Orthogonal Regularization:
#     * Intra-Matrix Isometry: Gram matrix isometric penalty on Q, K, V, Out attention projections
#     * Inter-Head Diversity: Normalized Gram matrix off-diagonal penalty across attention heads
#     * EEO Cross-Subspace Orthogonality: Normalized cosine similarity penalty between endo & exo weights
# - Pre-LN RMSNorm + Hardware-Accelerated FastSDPA (Native PyTorch Scaled Dot-Product Attention)
# - Dual-Context Readout Projection Head: Target variate token concatenated with global context mean
# - Master Production Benchmark Script (Caltech ACN with Weather, 10-Seed Deterministic Protocol)

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
torch.set_num_threads(6)
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
# 1. Dataset Loading & Preprocessing (Caltech with Weather)
# ==============================================================================
data_path = '../data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # safety: enforce chronological order before time-based split

# Drop unneeded noise columns (Paper Invariants: prcp, tempDiff_48, cldc)
# Keeping weather features intact (temp, rhum, wspd, pres, apparent_temp, tempMean_48)
drop_noise_cols = ['prcp', 'tempDiff_48', 'cldc']
df = df.drop(columns=drop_noise_cols, errors='ignore')

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

# EEO Feature Partition: Endogenous (Load + Calendar + Target) vs Exogenous (Weather Physics)
exo_col_names = ['temp', 'rhum', 'wspd', 'pres', 'apparent_temp', 'tempMean_48']
exo_indices = [i for i, c in enumerate(cols) if c in exo_col_names]
endo_indices = [i for i, c in enumerate(cols) if c not in exo_col_names] + [TARGET_CH_IDX]
print(f"  -> EEO Subspace Partition: {len(endo_indices)} Endogenous Variates, {len(exo_indices)} Exogenous Weather Variates")

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
# 3. Model Architecture (Inverted Variate-Centric Transformer with Dual-Context Head)
# ==============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019; MLE Foundations Topic 150).
    Replaces standard LayerNorm to eliminate mean shift computation, stabilizing gradient propagation
    and preserving isometric scaling across residual connections:
    RMSNorm(x) = (x / RMS(x)) * gamma, where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FastMHA(nn.Module):
    """
    Hardware-Accelerated Scaled Dot-Product Multi-Head Attention (FastMHA)
    (MLE Foundations Topic 151; FlashAttention & Online Softmax Math).
    Replaces standard un-fused MultiheadAttention with PyTorch native F.scaled_dot_product_attention.
    Executes directly in on-chip SRAM via fused FlashAttention-2 / cuDNN kernels on NVIDIA H100
    without materializing intermediate N x N attention matrices in high-bandwidth memory (HBM).
    Maintains independent Q, K, V linear projections to prevent Adam momentum coupling across orthogonal subspaces.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout

    def forward(self, query, key, value, is_causal=False):
        B, Sq, _ = query.shape
        _, Sk, _ = key.shape

        q = self.q_proj(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        drop_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).reshape(B, Sq, self.embed_dim)
        return self.out_proj(attn_out)


class InvertedCustomTransformer(nn.Module):
    """
    Model 00 Inverted Variate-Centric Transformer with EEO Disentangled Embedding
    and Multi-Head Cross-Variate Orthogonal Regularization (v7 Direction 1 Engine).

    Key Architectural Mechanisms:
    1. Inverted Variate Tokenization:
       - Transposes input [batch, lookback, num_variates] to [batch, num_variates, lookback].
       - Each variate time series is treated as an independent token (Vaswani NLP token analogue).
       - Eliminates the zero-placeholder decoder query failure mode of standard Seq2Seq.
    2. Disentangled EEO Variate Projection:
       - Endogenous tokens (Load history, lag features, calendar features, target load) -> endo_proj: Linear(lookback, d_model)
       - Exogenous tokens (Ambient weather physics: temp, rhum, wspd, pres, apparent_temp, tempMean) -> exo_proj: Linear(lookback, d_model)
       - Enforces distinct inductive representations between operational charging dynamics and external meteorological physics.
    3. Pre-LN RMSNorm + FastMHA Cross-Variate Encoder:
       - Computes all-to-all cross-variate attention across feature tokens (29 x 29 matrix, lightweight & stable).
       - Uses RMSNorm for scale invariance and stable condition numbers without mean-shift overhead.
       - Independent Q, K, V projections to avoid Adam optimizer momentum coupling across orthogonal subspaces.
    4. Vectorized Cross-Variate Orthogonal Regularization:
       - Intra-Matrix Isometry: W^T W approx I on Q, K, V, Out projections.
       - Inter-Head Diversity: Forces heads to attend to distinct variate relationships (Autocorrelation, Calendar, Weather).
       - EEO Cross-Subspace Orthogonality: Penalizes cross-correlation between endo_proj and exo_proj weight spaces.
    5. Dual-Context Target Readout Head:
       - Combines target variate token representation with global average pooled context across all variates.
       - Direct multi-step projection to forecast horizon H = 48 steps simultaneously.
    """
    def __init__(self, lookback, num_features=None, num_variates=None, horizon=48, target_ch_idx=None,
                 d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout_rate=0.1,
                 endo_indices=None, exo_indices=None):
        super().__init__()
        self.lookback = lookback
        self.num_variates = num_variates if num_variates is not None else num_features
        self.horizon = horizon
        self.target_ch_idx = target_ch_idx if target_ch_idx is not None else (self.num_variates - 1)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.endo_indices = endo_indices
        self.exo_indices = exo_indices

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Disentangled EEO Variate Projections
        if endo_indices is not None and exo_indices is not None:
            self.use_disentangled_proj = True
            self.endo_proj = nn.Linear(lookback, d_model)
            self.exo_proj  = nn.Linear(lookback, d_model)
        else:
            self.use_disentangled_proj = False
            self.variate_proj = nn.Linear(lookback, d_model)

        self.drop_in = nn.Dropout(dropout_rate)

        # Cross-Variate Transformer Layers (Pre-LN with RMSNorm)
        self.enc_attn = nn.ModuleList([
            FastMHA(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.enc_norm1 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.enc_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_layers)
        ])
        self.enc_norm2 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.enc_final_norm = RMSNorm(d_model)
        self.drop_enc = nn.Dropout(dropout_rate)

        # Dual-Context Readout Head: Target Token + Global Cross-Variate Mean Context
        self.out_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, horizon)
        )

        # Pre-cache attention layer references for zero-overhead vectorized penalty computation
        self.all_attn_layers = list(self.enc_attn)

    def forward(self, x):
        # x: [batch, lookback, num_variates]
        B = x.size(0)
        x_inv = x.transpose(1, 2)  # [batch, num_variates, lookback]

        # Disentangled EEO Variate Projection
        if self.use_disentangled_proj:
            tokens = torch.empty(B, self.num_variates, self.d_model, device=x.device)
            tokens[:, self.endo_indices, :] = self.endo_proj(x_inv[:, self.endo_indices, :])
            tokens[:, self.exo_indices, :]  = self.exo_proj(x_inv[:, self.exo_indices, :])
        else:
            tokens = self.variate_proj(x_inv)

        tokens = self.drop_in(tokens)

        # Cross-Variate Transformer Attention Layers
        for i in range(self.num_layers):
            normed = self.enc_norm1[i](tokens)
            attn_out = self.enc_attn[i](normed, normed, normed, is_causal=False)
            tokens = tokens + self.drop_enc(attn_out)

            normed = self.enc_norm2[i](tokens)
            ffn_out = self.enc_ffn[i](normed)
            tokens = tokens + self.drop_enc(ffn_out)

        tokens = self.enc_final_norm(tokens)

        # Dual-Context Target Readout Head
        target_token = tokens[:, self.target_ch_idx, :]          # [batch, d_model]
        global_mean  = torch.mean(tokens, dim=1)                  # [batch, d_model]
        context = torch.cat([target_token, global_mean], dim=-1) # [batch, 2 * d_model]
        out = self.out_head(context)                              # [batch, horizon]
        return out

    def compute_vectorized_orthogonal_penalty(self, strength=1e-5, inter_head_strength=1e-5, eeo_strength=1e-5):
        """
        Vectorized Batched Orthogonal Regularization across Attention Subspaces
        and Disentangled EEO Feature Projection Subspaces.
        """
        ref_device = self.endo_proj.weight.device if self.use_disentangled_proj else self.variate_proj.weight.device
        if strength <= 0.0 and inter_head_strength <= 0.0 and eeo_strength <= 0.0:
            return torch.tensor(0.0, device=ref_device)

        q_weights = [layer.q_proj.weight for layer in self.all_attn_layers]
        k_weights = [layer.k_proj.weight for layer in self.all_attn_layers]
        v_weights = [layer.v_proj.weight for layer in self.all_attn_layers]
        out_weights = [layer.out_proj.weight for layer in self.all_attn_layers]

        stacked_qk = torch.stack(q_weights + k_weights, dim=0)        # [2 * num_layers, d_model, d_model]
        stacked_v_out = torch.stack(v_weights + out_weights, dim=0)   # [2 * num_layers, d_model, d_model]
        all_weights = torch.cat([stacked_qk, stacked_v_out], dim=0)   # [4 * num_layers, d_model, d_model]

        penalty = torch.tensor(0.0, device=all_weights.device)

        # 1. Batched Intra-Matrix Isometry (across all Q, K, V, Out weight matrices)
        if strength > 0.0:
            wt_w = torch.bmm(all_weights.transpose(1, 2), all_weights)
            eye = torch.eye(self.d_model, device=all_weights.device).unsqueeze(0)
            penalty = penalty + strength * torch.sum((wt_w - eye) ** 2)

        # 2. Batched Inter-Head Diversity (across Q and K matrices)
        num_qk = stacked_qk.size(0)
        if inter_head_strength > 0.0 and self.num_heads > 1:
            w_heads_flat = stacked_qk.view(num_qk, self.num_heads, -1)
            head_norms = torch.norm(w_heads_flat, dim=-1, keepdim=True) + 1e-8
            norm_gram = torch.bmm(w_heads_flat, w_heads_flat.transpose(1, 2)) / (head_norms * head_norms.transpose(1, 2))
            off_diag = norm_gram - torch.eye(self.num_heads, device=all_weights.device).unsqueeze(0)
            inter_loss = torch.sum(off_diag ** 2) / (self.num_heads * (self.num_heads - 1))
            penalty = penalty + inter_head_strength * inter_loss

        # 3. Batched EEO Cross-Subspace Orthogonality (Endogenous vs Exogenous Projection Weights)
        if eeo_strength > 0.0 and self.use_disentangled_proj:
            w_endo = F.normalize(self.endo_proj.weight, p=2, dim=-1)  # [d_model, lookback]
            w_exo  = F.normalize(self.exo_proj.weight, p=2, dim=-1)   # [d_model, lookback]
            cross_cos = torch.mm(w_endo, w_exo.t())                   # [d_model, d_model]
            eeo_loss = torch.sum(cross_cos ** 2) / (self.d_model * self.d_model)
            penalty = penalty + eeo_strength * eeo_loss

        return penalty


# Alias for backward compatibility
EncoderDecoderTransformer = InvertedCustomTransformer


# ==============================================================================
# 4. Custom Feature: Attention Orthogonal Regularization Helper
# ==============================================================================
def compute_orthogonal_penalty(model, strength=1e-5, inter_head_strength=1e-5, eeo_strength=1e-5, num_heads=8):
    if hasattr(model, 'compute_vectorized_orthogonal_penalty'):
        return model.compute_vectorized_orthogonal_penalty(strength, inter_head_strength, eeo_strength)
    return torch.tensor(0.0, device=device)


# ==============================================================================
# 5. Metrics Evaluator Helper
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
# 6. Configuration & Multi-Seed Benchmark
# ==============================================================================
LOOKBACK = 96      # 48 hours history (96 * 30 min)
HORIZON  = 48      # 24 hours forecast (48 * 30 min)
BATCH_SIZE = 64    # Selected by Full HPO (Trial 14)
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]

# Hyperparameters (Selected by 50-Trial Optuna TPE Full HPO on Caltech)
D_MODEL             = 128
NUM_HEADS           = 8
D_FF                = 512
NUM_LAYERS          = 3
DROPOUT_RATE        = 0.1
LEARNING_RATE       = 0.00038608860950560737
WEIGHT_DECAY        = 1.000134592489716e-05
PATIENCE            = 15
LR_SCHEDULER_PATIENCE = 5

# Custom Regularization Hyperparameters (Intra-Matrix + Inter-Head Diversity + EEO Cross-Subspace)
ATTN_ORTHOGONAL_REG       = 6.842882094148905e-05
INTER_HEAD_ORTHOGONAL_REG = 0.0002480116648234053
EEO_ORTHOGONAL_REG        = 1.1175530342864682e-05

output_json_filename = "00_tfm_custom_pytorch_results.json"
results_data = {
    "model_name": "00_tfm_custom_pytorch",
    "architecture_paradigm": "inverted_variate_centric_transformer",
    "base_model": "07_tfm_itfm_pytorch",
    "version": "v7_direction1",
    "active_custom_features": [
        "inverted_variate_tokenization",
        "disentangled_eeo_variate_projections",
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
    "num_exo_features": len(exo_indices),
    "dataset": "acn_caltech_ready2",
    "with_weather": True,
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

# Pre-load tensors to target compute device (GPU VRAM / CPU) to eliminate per-batch transfer latency
X_train_dev = X_train_t.to(device)
y_train_dev = y_train_t.to(device)
X_val_dev   = X_val_t.to(device)
y_val_dev   = y_val_t.to(device)
X_test_dev  = X_test_t.to(device)

n_train = X_train_dev.size(0)
n_val   = X_val_dev.size(0)
n_test  = X_test_dev.size(0)
n_batches_train = n_train // BATCH_SIZE

print(f"Starting Automated {len(SEEDS)}-Seed Loop for 00_tfm_custom_pytorch in PyTorch...")

for seed_idx, SEED in enumerate(SEEDS, 1):
    seed_start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\n=========================================================================")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"=========================================================================")

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results_data["total_parameters"] = total_params

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-5)

    epochs = 200
    patience = PATIENCE
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history   = []
    best_epoch         = 1
    patience_counter   = 0
    best_model_weights = None

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Seed {SEED} Training", leave=True)
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

            # Primary MSE Loss + [ACTIVE CUSTOM FEATURE] Vectorized Batched Orthogonal Regularization
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
            if patience_counter >= patience:
                epoch_pbar.write(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Inference on Test Set
    model.eval()
    y_pred_list = []
    eval_batch_size = 1024
    with torch.inference_mode():
        for t_i in range(0, n_test, eval_batch_size):
            batch_X = X_test_dev[t_i : t_i + eval_batch_size]
            out = model(batch_X)
            y_pred_list.append(out.cpu().numpy())

    y_pred_scaled = np.vstack(y_pred_list)

    # Inverse transform predictions and actual values back to kW scale
    y_test_seq_unscaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    y_pred_unscaled     = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    # Extract step vectors
    actual_by_step = {step: y_test_seq_unscaled[:, step] for step in steps_to_eval}
    predictions_by_step = {step: y_pred_unscaled[:, step] for step in steps_to_eval}

    # Print evaluation metrics
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

    overall_metrics = compute_metrics(y_test_seq_unscaled.reshape(-1), y_pred_unscaled.reshape(-1), peak_threshold_kw)
    seed_duration = round(time.time() - seed_start_time, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device.type == 'cuda' else 0.0
    overall_metrics["training_time_seconds"] = seed_duration
    overall_metrics["peak_gpu_memory_mb"] = peak_vram_mb
    all_seed_metrics.append(overall_metrics)

    per_step_metrics = {}
    for step in steps_to_eval:
        m = compute_metrics(actual_by_step[step], predictions_by_step[step], peak_threshold_kw)
        per_step_metrics[step_labels[step]] = {k: (float(v) if not np.isnan(v) else None) for k, v in m.items()}

    mae_48 = [float(mean_absolute_error(y_test_seq_unscaled[:, s], y_pred_unscaled[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_test_seq_unscaled[:, s], y_pred_unscaled[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_unscaled.astype(np.float32)

    if best_val_loss < best_overall_val_loss and best_model_weights is not None:
        best_overall_val_loss = best_val_loss
        best_seed_id = SEED
        torch.save(best_model_weights, "00_tfm_custom_pytorch_best.pt")
        results_data["best_seed"] = int(SEED)
        print(f"  [Checkpoint] New overall best model saved from SEED {SEED} (Val Loss: {best_val_loss:.6f}) -> 00_tfm_custom_pytorch_best.pt")

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
    print(f"Successfully saved SEED {SEED} results to {output_json_filename} (Runtime: {seed_duration}s)")
    gc.collect()

all_predictions["y_true"] = y_test_seq_unscaled.astype(np.float32)
pred_stack = np.stack([all_predictions[f"seed_{s}"] for s in SEEDS], axis=0)
all_predictions["pred_mean"] = np.mean(pred_stack, axis=0).astype(np.float32)
all_predictions["pred_std"] = np.std(pred_stack, axis=0).astype(np.float32)
np.savez_compressed("00_tfm_custom_pytorch_predictions.npz", **all_predictions)
print(f"Successfully saved all seed predictions to 00_tfm_custom_pytorch_predictions.npz")

print(f"\n======================================================================")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — 00_tfm_custom_pytorch")
print(f"======================================================================")
summary_dict = {}
metric_keys = ['mae', 'rmse', 'r2', 'wape', 'mape', 'bias', 'negative_pct', 'training_time_seconds', 'peak_gpu_memory_mb']
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
    "batch_size": BATCH_SIZE,
    "seeds": SEEDS,
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "d_ff": D_FF,
    "num_layers": NUM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "weight_decay": WEIGHT_DECAY,
    "attn_orthogonal_reg": ATTN_ORTHOGONAL_REG,
    "inter_head_orthogonal_reg": INTER_HEAD_ORTHOGONAL_REG,
    "eeo_orthogonal_reg": EEO_ORTHOGONAL_REG,
    "learning_rate": LEARNING_RATE,
    "total_parameters": results_data.get("total_parameters", None)
}
results_data["summary"] = summary_dict
with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
print(f"Successfully saved final results to {output_json_filename}")

# Automatically archive artifacts to outputs/acn_caltech/00_tfm_custom_pytorch
import shutil
output_target_dir = os.path.join("outputs", "acn_caltech", "00_tfm_custom_pytorch")
os.makedirs(output_target_dir, exist_ok=True)
for fname in [output_json_filename, "00_tfm_custom_pytorch_best.pt", "00_tfm_custom_pytorch_predictions.npz"]:
    if os.path.exists(fname):
        shutil.copy(fname, os.path.join(output_target_dir, fname))
print(f"Successfully archived all artifacts to {output_target_dir}/")
print(f"\nFinished running all {len(SEEDS)} SEEDs in PyTorch!")
