#!/usr/bin/env python
# coding: utf-8

# # 00_tfm_custom_pytorch.py
# Custom Encoder-Decoder Transformer in PyTorch for EV Charging Load Forecasting (L=96, H=48)
# Base Architecture: 03_tfm_encdec_pytorch.py (Full Seq2Seq Transformer - Vaswani et al., NIPS 2017)
#
# Customization State:
# - Architecture Paradigm: Full Encoder-Decoder Seq2Seq with Causal Masked Self-Attention and Cross-Attention
# - Active Custom Feature: Attention Weight Orthogonal Regularization across Encoder, Decoder, and Cross-Attention
# - See docs/00_custom_features_log.md for full customization backlog and activation roadmap.

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
# 1. Dataset Loading & Preprocessing (JPL with Weather)
# ==============================================================================
data_path = '../data_cleaned/acn_jpl_ready.csv'

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

# EEO Feature Partition: Endogenous (Load + Calendar) vs Exogenous (Weather Physics)
exo_col_names = ['temp', 'rhum', 'wspd', 'pres', 'apparent_temp', 'tempMean_48']
exo_indices = [i for i, c in enumerate(cols) if c in exo_col_names]
endo_indices = [i for i, c in enumerate(cols) if c not in exo_col_names]

X = df[cols]
y = df['kWhDelivered']

print(f"Dataset Loaded successfully from {data_path}! Total Rows: {len(df)}, Features Count: {len(cols)}")
print(f"  -> EEO Subspace Partition: {len(endo_indices)} Endogenous Features, {len(exo_indices)} Exogenous Weather Features")

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
# 3. Model Architecture (Vaswani et al. 2017 Backbone with Dual-Context Head)
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
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, seq_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


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


class EncoderDecoderTransformer(nn.Module):
    """
    Custom Pre-LN Encoder-Decoder Seq2Seq Transformer with RMSNorm and
    Disentangled Exogenous-Endogenous Orthogonal Attention (Model 00 v5 Fused Fast Engine).
    Features:
    - Dual-Stream Feature Partitioning: Separates Endogenous features (load lags + calendar)
      from Exogenous features (ambient weather physics).
    - Dual-Subspace Projection: Dedicated linear projections (proj_endo, proj_exo) concatenated
      into a partitioned latent representation [d_model // 2, d_model // 2].
    - Pre-LN Residual Highway: Normalization precedes multi-head attention and FFN,
      guaranteeing an unimpeded gradient flow without vanishing gradients.
    - RMSNorm: Eliminates mean shift computation to enforce scale invariance and stabilize attention condition numbers.
    - Fused FastMHA: Fused Scaled Dot-Product Attention (SDPA) with hardware SRAM tiling (Topic 151).
    - Vectorized Batched Orthogonal Regularization: Closed-form batched matrix multiplication (torch.bmm)
      across all attention projections simultaneously (Topic 15 & 21).
    - Output Head: Token-wise Linear projection (d_model -> 1) squeezed to [batch, horizon].
    """
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2,
                 dropout_rate=0.05, endo_indices=None, exo_indices=None):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.endo_indices = endo_indices
        self.exo_indices = exo_indices

        # Dual-Stream Feature Projection (Endogenous vs Exogenous)
        if endo_indices is not None and exo_indices is not None:
            self.use_disentangled_proj = True
            d_endo = d_model // 2
            d_exo = d_model - d_endo
            self.endo_proj = nn.Linear(len(endo_indices), d_endo)
            self.exo_proj = nn.Linear(len(exo_indices), d_exo)
        else:
            self.use_disentangled_proj = False
            self.enc_proj = nn.Linear(num_features, d_model)

        self.pos_emb_enc = PositionalEmbedding(lookback, d_model)
        self.drop_enc = nn.Dropout(dropout_rate)

        self.enc_attn = nn.ModuleList([
            FastMHA(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.enc_norm1 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.enc_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_layers)
        ])
        self.enc_norm2 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.enc_final_norm = RMSNorm(d_model)

        # Decoder
        self.pos_emb_dec = PositionalEmbedding(horizon, d_model)
        self.drop_dec = nn.Dropout(dropout_rate)

        self.dec_attn = nn.ModuleList([
            FastMHA(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.dec_norm1 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])

        self.cross_attn = nn.ModuleList([
            FastMHA(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.dec_norm2 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])

        self.ffn_dec = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_layers)
        ])
        self.dec_norm3 = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        self.dec_final_norm = RMSNorm(d_model)

        # Token-wise linear projection head
        self.out_head = nn.Linear(d_model, 1)

        # Pre-cache attention layer references for zero-overhead vectorized penalty computation
        self.all_attn_layers = list(self.enc_attn) + list(self.dec_attn) + list(self.cross_attn)

    def forward(self, x):
        # x: [batch, lookback, num_features]
        batch_size = x.size(0)

        # Encoder (Pre-LN with RMSNorm and Disentangled EEO Projection)
        if self.use_disentangled_proj:
            x_endo = x[:, :, self.endo_indices]
            x_exo = x[:, :, self.exo_indices]
            enc_feat = torch.cat([self.endo_proj(x_endo), self.exo_proj(x_exo)], dim=-1)
        else:
            enc_feat = self.enc_proj(x)

        enc = self.drop_enc(self.pos_emb_enc(enc_feat))
        for i in range(self.num_layers):
            normed = self.enc_norm1[i](enc)
            attn_out = self.enc_attn[i](normed, normed, normed, is_causal=False)
            enc = enc + self.drop_enc(attn_out)

            normed = self.enc_norm2[i](enc)
            ffn_out = self.enc_ffn[i](normed)
            enc = enc + self.drop_enc(ffn_out)

        enc_out = self.enc_final_norm(enc)

        # Decoder initial context (zero placeholder query tokens for forecast horizon)
        dec_in = torch.zeros(batch_size, self.horizon, self.d_model, device=x.device)
        dec = self.drop_dec(self.pos_emb_dec(dec_in))

        for i in range(self.num_layers):
            normed = self.dec_norm1[i](dec)
            # Fused SDPA causal masking (is_causal=True) executes directly in SRAM without materializing mask tensor
            dec_attn_out = self.dec_attn[i](normed, normed, normed, is_causal=True)
            dec = dec + self.drop_dec(dec_attn_out)

            normed = self.dec_norm2[i](dec)
            cross_attn_out = self.cross_attn[i](query=normed, key=enc_out, value=enc_out, is_causal=False)
            dec = dec + self.drop_dec(cross_attn_out)

            normed = self.dec_norm3[i](dec)
            ffn_out = self.ffn_dec[i](normed)
            dec = dec + self.drop_dec(ffn_out)

        dec = self.dec_final_norm(dec)
        out = self.out_head(dec).squeeze(-1)  # [batch, horizon]
        return out

    def compute_vectorized_orthogonal_penalty(self, strength=1e-5, inter_head_strength=1e-5, eeo_strength=1e-5):
        """
        [ACTIVE CUSTOM FEATURE: Vectorized Batched Orthogonal Regularization (Model 00 v5 Engine)]
        Replaces Python named_parameters loop and micro-kernel launches with 3 batched GEMM operations (torch.bmm):
        1. Batched Intra-Matrix Isometry: all_weights.transpose(1, 2) @ all_weights -> 1 kernel
        2. Batched Inter-Head Diversity: w_heads_flat @ w_heads_flat.transpose(1, 2) -> 1 kernel
        3. Batched EEO Cross-Subspace Orthogonality: w_endo @ w_exo.transpose(1, 2) -> 1 kernel
        """
        if strength <= 0.0 and inter_head_strength <= 0.0 and eeo_strength <= 0.0:
            return torch.tensor(0.0, device=self.out_head.weight.device)

        q_weights = [layer.q_proj.weight for layer in self.all_attn_layers]
        k_weights = [layer.k_proj.weight for layer in self.all_attn_layers]
        v_weights = [layer.v_proj.weight for layer in self.all_attn_layers]
        out_weights = [layer.out_proj.weight for layer in self.all_attn_layers]

        stacked_qk = torch.stack(q_weights + k_weights, dim=0) # [12, d_model, d_model]
        stacked_v_out = torch.stack(v_weights + out_weights, dim=0) # [12, d_model, d_model]
        all_weights = torch.cat([stacked_qk, stacked_v_out], dim=0) # [24, d_model, d_model]

        penalty = torch.tensor(0.0, device=all_weights.device)

        # 1. Batched Intra-Matrix Isometry (across all 24 weight matrices)
        if strength > 0.0:
            wt_w = torch.bmm(all_weights.transpose(1, 2), all_weights)
            eye = torch.eye(self.d_model, device=all_weights.device).unsqueeze(0)
            penalty = penalty + strength * torch.sum((wt_w - eye) ** 2)

        # 2. Batched Inter-Head Diversity (across all 12 Q and K matrices)
        num_qk = stacked_qk.size(0)
        if inter_head_strength > 0.0 and self.num_heads > 1:
            w_heads_flat = stacked_qk.view(num_qk, self.num_heads, -1)
            head_norms = torch.norm(w_heads_flat, dim=-1, keepdim=True) + 1e-8
            norm_gram = torch.bmm(w_heads_flat, w_heads_flat.transpose(1, 2)) / torch.bmm(head_norms, head_norms.transpose(1, 2))
            off_diag = norm_gram - torch.eye(self.num_heads, device=all_weights.device).unsqueeze(0)
            inter_loss = torch.sum(off_diag ** 2) / (num_qk * self.num_heads * (self.num_heads - 1))
            penalty = penalty + inter_head_strength * inter_loss

        # 3. Batched EEO Cross-Subspace Orthogonality (across all 12 Q and K matrices)
        if eeo_strength > 0.0 and self.num_heads >= 4:
            mid = self.num_heads // 2
            w_heads = stacked_qk.view(num_qk, self.num_heads, self.head_dim, self.d_model)
            w_endo = w_heads[:, :mid].reshape(num_qk, mid * self.head_dim, self.d_model)
            w_exo  = w_heads[:, mid:].reshape(num_qk, mid * self.head_dim, self.d_model)
            w_endo_norm = torch.norm(w_endo, dim=-1, keepdim=True) + 1e-8
            w_exo_norm  = torch.norm(w_exo, dim=-1, keepdim=True) + 1e-8
            cross_cos = torch.bmm(w_endo, w_exo.transpose(1, 2)) / torch.bmm(w_endo_norm, w_exo_norm.transpose(1, 2))
            eeo_loss = torch.mean(cross_cos ** 2)
            penalty = penalty + eeo_strength * eeo_loss

        return penalty


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
BATCH_SIZE = 128    # Seq2Seq optimal batch size for JPL
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]

# Hyperparameters (Matching Seq2Seq optimal baseline 03 on JPL with weather)
D_MODEL             = 128
NUM_HEADS           = 8
D_FF                = 512
NUM_LAYERS          = 2
DROPOUT_RATE        = 0.1
LEARNING_RATE       = 0.0006097839109531517
WEIGHT_DECAY        = 3.972110727381911e-06
PATIENCE            = 15
LR_SCHEDULER_PATIENCE = 5

# Custom Regularization Hyperparameters (Intra-Matrix + Inter-Head Diversity + EEO Cross-Subspace)
ATTN_ORTHOGONAL_REG = 4.207053950287936e-06
INTER_HEAD_ORTHOGONAL_REG = 3.1489116479568635e-05
EEO_ORTHOGONAL_REG = 3.1489116479568635e-05

output_json_filename = "00_tfm_custom_pytorch_results.json"
results_data = {
    "model_name": "00_tfm_custom_pytorch",
    "architecture_paradigm": "encoder_decoder_seq2seq_fast_fused_eeo",
    "base_model": "03_tfm_encdec_pytorch",
    "version": "v5",
    "active_custom_features": [
        "encoder_decoder_cross_attention",
        "attention_orthogonal_regularization",
        "inter_head_orthogonal_regularization",
        "pre_ln_rmsnorm_backbone",
        "disentangled_eeo_attention",
        "fast_sdpa_fused_attention",
        "vectorized_batched_orthogonal_regularization",
        "zero_copy_gpu_resident_tensors"
    ],
    "attn_orthogonal_reg_strength": ATTN_ORTHOGONAL_REG,
    "inter_head_orthogonal_reg_strength": INTER_HEAD_ORTHOGONAL_REG,
    "eeo_orthogonal_reg_strength": EEO_ORTHOGONAL_REG,
    "num_endo_features": len(endo_indices),
    "num_exo_features": len(exo_indices),
    "dataset": "acn_jpl_ready",
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

    model = EncoderDecoderTransformer(
        lookback=LOOKBACK,
        num_features=X_train_scaled.shape[1],
        horizon=HORIZON,
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
            optimizer.step()
            train_loss += mse_loss.item() * BATCH_SIZE

        train_loss /= (n_batches_train * BATCH_SIZE)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for v_i in range(0, n_val, BATCH_SIZE):
                batch_X = X_val_dev[v_i : v_i + BATCH_SIZE]
                batch_y = y_val_dev[v_i : v_i + BATCH_SIZE]
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
    with torch.inference_mode():
        for t_i in range(0, n_test, BATCH_SIZE):
            batch_X = X_test_dev[t_i : t_i + BATCH_SIZE]
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

# Automatically archive artifacts to outputs/acn_jpn/00_v5
import shutil
output_v5_dir = os.path.join("outputs", "acn_jpn", "00_v5")
os.makedirs(output_v5_dir, exist_ok=True)
for fname in [output_json_filename, "00_tfm_custom_pytorch_best.pt", "00_tfm_custom_pytorch_predictions.npz"]:
    if os.path.exists(fname):
        shutil.copy(fname, os.path.join(output_v5_dir, fname))
print(f"Successfully archived all artifacts to {output_v5_dir}/")
print(f"\nFinished running all {len(SEEDS)} SEEDs in PyTorch!")
