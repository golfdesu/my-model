#!/usr/bin/env python
# coding: utf-8

"""
2x2 Factorial Ablation Study Engine for Model 00 (ACN JPL V3 & Caltech V3)
Investigates the individual and interaction effects of:
  1. EEO (Disentangled Endogenous / Exogenous Variate Projections)
  2. Orthogonal Regularization (Intra-Matrix Isometry + Inter-Head Diversity + EEO Cross-Subspace)

Variants:
  - V1 (Full Model 00): EEO = ON,  Ortho = ON  (Loaded from existing 00 results)
  - V2 (No EEO)        : EEO = OFF, Ortho = ON  (Evaluated across 10 seeds)
  - V3 (No Ortho)      : EEO = ON,  Ortho = OFF (Evaluated across 10 seeds)
  - V4 (Backbone Only) : EEO = OFF, Ortho = OFF (Evaluated across 10 seeds)
"""

import sys
import os
import argparse
import time
import json
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
    def tqdm(iterable, *args, **kwargs):
        return iterable

warnings.filterwarnings('ignore')

# Hardware optimization
num_cpus = os.cpu_count() or 4
torch.set_num_threads(min(6, num_cpus))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# Model Architecture (Identical to 00_tfm_custom_pytorch.py)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FastMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

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
    def __init__(self, lookback, num_variates, horizon=48, target_ch_idx=None,
                 d_model=128, num_heads=8, d_ff=512, num_layers=2, dropout_rate=0.2,
                 endo_indices=None, exo_indices=None):
        super().__init__()
        self.lookback = lookback
        self.num_variates = num_variates
        self.horizon = horizon
        self.target_ch_idx = target_ch_idx if target_ch_idx is not None else (num_variates - 1)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.endo_indices = endo_indices
        self.exo_indices = exo_indices

        # Disentangled EEO Variate Projections vs Single Variate Projection
        if endo_indices is not None and exo_indices is not None:
            self.use_disentangled_proj = True
            self.endo_proj = nn.Linear(lookback, d_model)
            self.exo_proj  = nn.Linear(lookback, d_model)
        else:
            self.use_disentangled_proj = False
            self.variate_proj = nn.Linear(lookback, d_model)

        self.drop_in = nn.Dropout(dropout_rate)

        # Cross-Variate Transformer Layers
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

        # Dual-Context Target Readout Head
        self.out_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, horizon)
        )

        self.all_attn_layers = list(self.enc_attn)

    def forward(self, x):
        B = x.size(0)
        x_inv = x.transpose(1, 2)  # [batch, num_variates, lookback]

        if self.use_disentangled_proj:
            tokens = torch.empty(B, self.num_variates, self.d_model, device=x.device)
            tokens[:, self.endo_indices, :] = self.endo_proj(x_inv[:, self.endo_indices, :])
            tokens[:, self.exo_indices, :]  = self.exo_proj(x_inv[:, self.exo_indices, :])
        else:
            tokens = self.variate_proj(x_inv)

        tokens = self.drop_in(tokens)

        for i in range(self.num_layers):
            normed = self.enc_norm1[i](tokens)
            attn_out = self.enc_attn[i](normed, normed, normed, is_causal=False)
            tokens = tokens + self.drop_enc(attn_out)

            normed = self.enc_norm2[i](tokens)
            ffn_out = self.enc_ffn[i](normed)
            tokens = tokens + self.drop_enc(ffn_out)

        tokens = self.enc_final_norm(tokens)

        target_token = tokens[:, self.target_ch_idx, :]
        global_mean  = torch.mean(tokens, dim=1)
        context = torch.cat([target_token, global_mean], dim=-1)
        return self.out_head(context)

    def compute_vectorized_orthogonal_penalty(self, strength=0.0, inter_head_strength=0.0, eeo_strength=0.0):
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

        # 1. Intra-Matrix Isometry
        if strength > 0.0:
            wt_w = torch.bmm(all_weights.transpose(1, 2), all_weights)
            eye = torch.eye(self.d_model, device=all_weights.device).unsqueeze(0)
            penalty = penalty + strength * torch.sum((wt_w - eye) ** 2)

        # 2. Inter-Head Diversity
        num_qk = stacked_qk.size(0)
        if inter_head_strength > 0.0 and self.num_heads > 1:
            w_heads_flat = stacked_qk.view(num_qk, self.num_heads, -1)
            head_norms = torch.norm(w_heads_flat, dim=-1, keepdim=True) + 1e-8
            norm_gram = torch.bmm(w_heads_flat, w_heads_flat.transpose(1, 2)) / (head_norms * head_norms.transpose(1, 2))
            off_diag = norm_gram - torch.eye(self.num_heads, device=all_weights.device).unsqueeze(0)
            inter_loss = torch.sum(off_diag ** 2) / (self.num_heads * (self.num_heads - 1))
            penalty = penalty + inter_head_strength * inter_loss

        # 3. EEO Cross-Subspace Orthogonality
        if eeo_strength > 0.0 and self.use_disentangled_proj:
            w_endo = F.normalize(self.endo_proj.weight, p=2, dim=-1)
            w_exo  = F.normalize(self.exo_proj.weight, p=2, dim=-1)
            cross_cos = torch.mm(w_endo, w_exo.t())
            eeo_loss = torch.sum(cross_cos ** 2) / (self.d_model * self.d_model)
            penalty = penalty + eeo_strength * eeo_loss

        return penalty


# ==============================================================================
# Helper Functions
# ==============================================================================
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
    mape = float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100) if non_zero.any() else None
    peak = actual >= peak_threshold
    if peak.any():
        mae_peak  = float(mean_absolute_error(actual[peak], predicted[peak]))
        wape_peak = float((np.sum(np.abs(actual[peak] - predicted[peak])) / np.sum(actual[peak])) * 100)
    else:
        mae_peak, wape_peak = None, None

    bias = float(np.mean(predicted - actual))
    negative_pct = float(np.mean(predicted < 0) * 100)
    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape,
                mae_peak=mae_peak, wape_peak=wape_peak, bias=bias, negative_pct=negative_pct)


# ==============================================================================
# Dataset & Hyperparameter Configurations
# ==============================================================================
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
LOOKBACK = 96
HORIZON = 48

DATASET_CONFIGS = {
    'jpl': {
        'name_tag': 'acn_jpl_v3',
        'display_name': 'ACN JPL V3',
        'csv_candidates': [
            '../data_cleaned/acn_jpl_ready_v3.csv',
            'data_cleaned/acn_jpl_ready_v3.csv',
            '../../data_cleaned/acn_jpl_ready_v3.csv',
            'acn_jpl_ready_v3.csv',
            '../data_cleaned/acn_jpn_ready_v3.csv',
            'data_cleaned/acn_jpn_ready_v3.csv'
        ],
        'v1_candidates': [
            'outputs/alr_fix/acn_jpn_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            'outputs/alr_fix/acn_jpl_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            'outputs/acn_jpn_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            'outputs/acn_jpl_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            'outputs/not_fix/acn_jpn_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            '00_tfm_custom_pytorch_results.json'
        ],
        'm07_candidates': [
            'outputs/alr_fix/acn_jpn_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            'outputs/alr_fix/acn_jpl_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            'outputs/acn_jpn_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            'outputs/acn_jpl_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            'outputs/not_fix/acn_jpn_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            '07_tfm_itfm_pytorch_results.json'
        ],
        'd_model': 128,
        'num_heads': 8,
        'd_ff': 512,
        'num_layers': 2,
        'dropout_rate': 0.1,
        'learning_rate': 0.0001493931976721778,
        'weight_decay': 8.492935741365948e-06,
        'batch_size': 32,
        'patience': 15,
        'lr_scheduler_patience': 5,
        'attn_reg_default': 0.00010565582330168113,
        'inter_reg_default': 2.7131713354741595e-05,
        'eeo_reg_default': 4.344012725210976e-06
    },
    'caltech': {
        'name_tag': 'acn_caltech_v3',
        'display_name': 'Caltech V3',
        'csv_candidates': [
            '../data_cleaned/acn_caltech_ready_v3.csv',
            'data_cleaned/acn_caltech_ready_v3.csv',
            '../../data_cleaned/acn_caltech_ready_v3.csv',
            'acn_caltech_ready_v3.csv',
            '../data_cleaned/acn_caltech_ready2.csv',
            'data_cleaned/acn_caltech_ready2.csv'
        ],
        'v1_candidates': [
            'outputs/alr_fix/acn_caltech_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            'outputs/acn_caltech_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            'outputs/not_fix/acn_caltech_v3/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json',
            '00_tfm_custom_pytorch_results.json'
        ],
        'm07_candidates': [
            'outputs/alr_fix/acn_caltech_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            'outputs/acn_caltech_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            'outputs/not_fix/acn_caltech_v3/07_tfm_itfm_pytorch/07_tfm_itfm_pytorch_results.json',
            '07_tfm_itfm_pytorch_results.json'
        ],
        'd_model': 128,
        'num_heads': 8,
        'd_ff': 512,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 0.0004299749266334553,
        'weight_decay': 2.000844389639091e-06,
        'batch_size': 32,
        'patience': 15,
        'lr_scheduler_patience': 5,
        'attn_reg_default': 0.0031134843833110284,
        'inter_reg_default': 3.099246938221801e-05,
        'eeo_reg_default': 1.2016244471675806e-05
    }
}


def load_dataset(dataset_cfg):
    csv_candidates = dataset_cfg['csv_candidates']
    data_path = None
    for cand in csv_candidates:
        if os.path.exists(cand):
            data_path = cand
            break

    if data_path is None:
        raise FileNotFoundError(f"Could not locate dataset for {dataset_cfg['display_name']} from candidates: {csv_candidates}")

    df = pd.read_csv(data_path)
    df['connectionTime'] = pd.to_datetime(df['connectionTime'])
    df = df.set_index('connectionTime').sort_index()

    # Drop unneeded noise columns (Paper Invariants)
    drop_noise_cols = ['prcp', 'tempDiff_48', 'cldc']
    df = df.drop(columns=drop_noise_cols, errors='ignore')

    cols = [col for col in df.columns if col != 'kWhDelivered']
    for col in df.columns:
        df[col] = df[col].astype('float32')

    X = df[cols]
    y = df['kWhDelivered']

    train_len = int(len(df) * 0.6)
    val_len   = int(len(df) * 0.2)

    X_train, y_train = X[:train_len], y[:train_len]
    X_val,   y_val   = X[train_len : train_len + val_len], y[train_len : train_len + val_len]
    X_test,  y_test  = X[train_len + val_len :], y[train_len + val_len :]

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    target_idx = X_train_scaled.shape[1]
    X_train_scaled = np.concatenate([X_train_scaled, y_train_scaled.reshape(-1, 1)], axis=1)
    X_val_scaled   = np.concatenate([X_val_scaled,   y_val_scaled.reshape(-1, 1)], axis=1)
    X_test_scaled  = np.concatenate([X_test_scaled,  y_test_scaled.reshape(-1, 1)], axis=1)

    exo_col_names = ['temp', 'rhum', 'wspd', 'pres', 'apparent_temp', 'tempMean_48']
    exo_indices = [i for i, c in enumerate(cols) if c in exo_col_names]
    endo_indices = [i for i, c in enumerate(cols) if c not in exo_col_names] + [target_idx]

    peak_threshold = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))

    print(f"Loaded {dataset_cfg['display_name']} from: {data_path}")
    print(f"  Rows: {len(df)} (Train: {train_len}, Val: {val_len}, Test: {len(df) - train_len - val_len})")
    print(f"  Variate Tokens: {X_train_scaled.shape[1]} ({len(endo_indices)} Endogenous, {len(exo_indices)} Exogenous)")
    print(f"  Peak Load Threshold: {peak_threshold:.4f} kW")

    return {
        'X_train_scaled': X_train_scaled, 'y_train_scaled': y_train_scaled,
        'X_val_scaled': X_val_scaled,     'y_val_scaled': y_val_scaled,
        'X_test_scaled': X_test_scaled,   'y_test_scaled': y_test_scaled,
        'target_idx': target_idx,
        'endo_indices': endo_indices, 'exo_indices': exo_indices,
        'scaler_y': scaler_y, 'peak_threshold': peak_threshold,
        'data_path': data_path
    }


def get_variant_configs(dataset_cfg):
    attn_reg = dataset_cfg['attn_reg_default']
    inter_reg = dataset_cfg['inter_reg_default']
    eeo_reg = dataset_cfg['eeo_reg_default']

    return {
        'v1_full': {
            'name': 'V1: Full Model 00 (EEO + Ortho)',
            'use_eeo': True,
            'attn_reg': attn_reg,
            'inter_reg': inter_reg,
            'eeo_reg': eeo_reg
        },
        'v2_no_eeo': {
            'name': 'V2: Ablate EEO (No EEO, Ortho ON)',
            'use_eeo': False,
            'attn_reg': attn_reg,
            'inter_reg': inter_reg,
            'eeo_reg': 0.0
        },
        'v3_no_ortho': {
            'name': 'V3: Ablate Ortho (EEO ON, No Ortho)',
            'use_eeo': True,
            'attn_reg': 0.0,
            'inter_reg': 0.0,
            'eeo_reg': 0.0
        },
        'v4_backbone': {
            'name': 'V4: Backbone Only (No EEO, No Ortho)',
            'use_eeo': False,
            'attn_reg': 0.0,
            'inter_reg': 0.0,
            'eeo_reg': 0.0
        }
    }


def run_single_variant(var_key, data_bundle, out_dir, dataset_cfg):
    variant_configs = get_variant_configs(dataset_cfg)
    cfg = variant_configs[var_key]
    print(f"\n{'='*75}")
    print(f">>> STARTING ABLATION VARIANT: {cfg['name']}")
    print(f"    Target Dataset        : {dataset_cfg['display_name']}")
    print(f"    EEO Disentangled Proj : {cfg['use_eeo']}")
    print(f"    Attn Ortho Reg        : {cfg['attn_reg']:.6e}")
    print(f"    Inter-Head Ortho Reg  : {cfg['inter_reg']:.6e}")
    print(f"    EEO Ortho Reg         : {cfg['eeo_reg']:.6e}")
    print(f"    Learning Rate         : {dataset_cfg['learning_rate']:.6e}")
    print(f"    Dropout Rate          : {dataset_cfg['dropout_rate']}")
    print(f"    Batch Size            : {dataset_cfg['batch_size']}")
    print(f"{'='*75}")

    X_train_scaled = data_bundle['X_train_scaled']
    y_train_scaled = data_bundle['y_train_scaled']
    X_val_scaled   = data_bundle['X_val_scaled']
    y_val_scaled   = data_bundle['y_val_scaled']
    X_test_scaled  = data_bundle['X_test_scaled']
    y_test_scaled  = data_bundle['y_test_scaled']
    target_idx     = data_bundle['target_idx']
    scaler_y       = data_bundle['scaler_y']
    peak_threshold = data_bundle['peak_threshold']
    num_variates   = X_train_scaled.shape[1]

    endo_idx = data_bundle['endo_indices'] if cfg['use_eeo'] else None
    exo_idx  = data_bundle['exo_indices']  if cfg['use_eeo'] else None

    X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
    X_val_t,   y_val_t,   _, _ = create_windowed_tensors(X_val_scaled,   y_val_scaled,   LOOKBACK, HORIZON)
    X_test_t,  y_test_t,  _, _ = create_windowed_tensors(X_test_scaled,  y_test_scaled,  LOOKBACK, HORIZON)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    all_seed_metrics = []
    all_predictions = {}
    results_data = {
        'variant_key': var_key,
        'variant_name': cfg['name'],
        'dataset': dataset_cfg['name_tag'],
        'config': {
            'use_eeo': cfg['use_eeo'],
            'attn_reg': cfg['attn_reg'],
            'inter_reg': cfg['inter_reg'],
            'eeo_reg': cfg['eeo_reg'],
            'learning_rate': dataset_cfg['learning_rate'],
            'dropout_rate': dataset_cfg['dropout_rate'],
            'batch_size': dataset_cfg['batch_size'],
            'weight_decay': dataset_cfg['weight_decay'],
            'd_model': dataset_cfg['d_model'],
            'num_heads': dataset_cfg['num_heads'],
            'd_ff': dataset_cfg['d_ff'],
            'num_layers': dataset_cfg['num_layers']
        },
        'seeds': {},
        'summary': {}
    }

    batch_size = dataset_cfg['batch_size']
    learning_rate = dataset_cfg['learning_rate']
    weight_decay = dataset_cfg['weight_decay']
    d_model = dataset_cfg['d_model']
    num_heads = dataset_cfg['num_heads']
    d_ff = dataset_cfg['d_ff']
    num_layers = dataset_cfg['num_layers']
    dropout_rate = dataset_cfg['dropout_rate']
    patience = dataset_cfg['patience']
    lr_scheduler_patience = dataset_cfg['lr_scheduler_patience']

    t_variant_start = time.time()
    for seed_idx, seed in enumerate(SEEDS, 1):
        t_seed_start = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, pin_memory=(device.type == 'cuda'))
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

        model = InvertedCustomTransformer(
            lookback=LOOKBACK, num_variates=num_variates, horizon=HORIZON,
            target_ch_idx=target_idx, d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, num_layers=num_layers, dropout_rate=dropout_rate,
            endo_indices=endo_idx, exo_indices=exo_idx
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience, min_lr=1e-5)

        epochs = 200
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bX, by in train_loader:
                bX, by = bX.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(bX)
                forecast_loss = criterion(pred, by)

                reg_loss = model.compute_vectorized_orthogonal_penalty(
                    strength=cfg['attn_reg'],
                    inter_head_strength=cfg['inter_reg'],
                    eeo_strength=cfg['eeo_reg']
                )
                loss = forecast_loss + reg_loss
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
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Test evaluation with best weights
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model.eval()

        all_preds, all_targets = [], []
        with torch.inference_mode():
            for bX, by in test_loader:
                all_preds.append(model(bX.to(device)).cpu().numpy())
                all_targets.append(by.numpy())

        y_pred_scaled = np.concatenate(all_preds, axis=0)
        y_true_scaled = np.concatenate(all_targets, axis=0)
        y_pred_kw = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, HORIZON)
        y_true_kw = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1, HORIZON)

        metrics = compute_metrics(y_true_kw.flatten(), y_pred_kw.flatten(), peak_threshold)
        seed_duration = round(time.time() - t_seed_start, 2)
        metrics['training_time_seconds'] = seed_duration
        all_seed_metrics.append(metrics)
        all_predictions[f'seed_{seed}'] = y_pred_kw.astype(np.float32)

        results_data['seeds'][str(seed)] = {
            'best_val_loss': float(best_val_loss),
            'training_time_seconds': seed_duration,
            'metrics': metrics
        }
        print(f"  [{var_key.upper()}] Seed {seed:4d} ({seed_idx:2d}/{len(SEEDS)}) | Val Loss: {best_val_loss:.6f} | MAE: {metrics['mae']:.4f} kW | RMSE: {metrics['rmse']:.4f} kW ({seed_duration}s)")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summarize variant
    summary_dict = {}
    for k in ['mae', 'rmse', 'r2', 'wape', 'mae_peak', 'bias', 'training_time_seconds']:
        vals = [float(m[k]) for m in all_seed_metrics if k in m and m[k] is not None and not np.isnan(m[k])]
        if vals:
            summary_dict[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    results_data['summary'] = summary_dict

    # Save variant JSON and predictions NPZ
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{var_key}_results.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    all_predictions['y_true'] = y_true_kw.astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, f"{var_key}_predictions.npz"), **all_predictions)

    elapsed_var = round(time.time() - t_variant_start, 2)
    print(f"\n>>> [{var_key.upper()} COMPLETED in {elapsed_var}s]")
    print(f"    Mean MAE  : {summary_dict['mae']['mean']:.4f} ± {summary_dict['mae']['std']:.4f} kW")
    print(f"    Mean RMSE : {summary_dict['rmse']['mean']:.4f} ± {summary_dict['rmse']['std']:.4f} kW")
    print(f"    Mean R²   : {summary_dict['r2']['mean']:.4f} ± {summary_dict['r2']['std']:.4f}")
    return results_data


# ==============================================================================
# Master Aggregation & Factorial Analysis
# ==============================================================================
def compile_ablation_report(out_dir, dataset_cfg):
    print(f"\n{'='*75}")
    print(f">>> COMPILING 2x2 FACTORIAL ABLATION REPORT FOR {dataset_cfg['display_name'].upper()}")
    print(f"{'='*75}")

    # 1. Load V1 from existing results
    v1_candidates = dataset_cfg.get('v1_candidates', []) + [
        os.path.join(out_dir, 'v1_full_results.json'),
        os.path.join(out_dir, '00_tfm_custom_pytorch_results.json')
    ]

    target_kw = 'caltech' if 'caltech' in dataset_cfg['name_tag'].lower() else 'jp'
    v1_data = None
    for cand in v1_candidates:
        if os.path.exists(cand):
            try:
                with open(cand, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    cand_ds = str(d.get('dataset', '')).lower()
                    if cand_ds and target_kw not in cand_ds:
                        continue
                    if 'summary' in d and 'mae' in d['summary']:
                        v1_data = d
                        print(f"Loaded V1 (Full Model 00) from {cand}")
                        break
            except Exception:
                pass

    variants = {}
    if v1_data is not None:
        v1_summary = dict(v1_data['summary'])
        if 'mae_peak' not in v1_summary and 'seeds' in v1_data:
            peaks = [
                s['overall_metrics']['mae_peak']
                for s in v1_data['seeds'].values()
                if isinstance(s, dict) and 'overall_metrics' in s and 'mae_peak' in s['overall_metrics'] and s['overall_metrics']['mae_peak'] is not None
            ]
            if peaks:
                v1_summary['mae_peak'] = {'mean': float(np.mean(peaks)), 'std': float(np.std(peaks))}
        variants['V1 (Full: EEO + Ortho)'] = v1_summary

    for var_key, var_label in [
        ('v2_no_eeo', 'V2 (Ablate EEO: No EEO, Ortho ON)'),
        ('v3_no_ortho', 'V3 (Ablate Ortho: EEO ON, No Ortho)'),
        ('v4_backbone', 'V4 (Backbone: No EEO, No Ortho)')
    ]:
        p = os.path.join(out_dir, f"{var_key}_results.json")
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
                variants[var_label] = d['summary']
        else:
            print(f"Notice: {var_key}_results.json not found yet in {out_dir}")

    # Check for Model 07 reference
    m07_candidates = dataset_cfg.get('m07_candidates', []) + [
        os.path.join(out_dir, '07_tfm_itfm_pytorch_results.json')
    ]

    for cand in m07_candidates:
        if os.path.exists(cand):
            try:
                with open(cand, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    cand_ds = str(d.get('dataset', '')).lower()
                    if cand_ds and target_kw not in cand_ds:
                        continue
                    if 'summary' in d and 'mae' in d['summary']:
                        m07_summary = dict(d['summary'])
                        if 'mae_peak' not in m07_summary and 'seeds' in d:
                            peaks = [
                                s['overall_metrics']['mae_peak']
                                for s in d['seeds'].values()
                                if isinstance(s, dict) and 'overall_metrics' in s and 'mae_peak' in s['overall_metrics'] and s['overall_metrics']['mae_peak'] is not None
                            ]
                            if peaks:
                                m07_summary['mae_peak'] = {'mean': float(np.mean(peaks)), 'std': float(np.std(peaks))}
                        variants['Ref: Model 07 (iTransformer)'] = m07_summary
                        print(f"Loaded Ref Model 07 from {cand}")
                        break
            except Exception:
                pass

    # Markdown Table Generation
    lines = []
    lines.append(f"# Model 00 Ablation Study: EEO vs Orthogonal Regularization ({dataset_cfg['display_name']})")
    lines.append("")
    lines.append("## 1. 2x2 Factorial Performance Matrix (10 Seeds Mean ± Std)")
    lines.append("")
    lines.append("| Model Variant | EEO Proj | Ortho Reg | MAE (kW) | RMSE (kW) | R² | Peak MAE (kW) | Runtime (s) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for name, s in variants.items():
        eeo_flag = "✅" if "EEO ON" in name or "Full" in name else ("❌" if "No EEO" in name or "Backbone" in name else "-")
        ortho_flag = "✅" if "Ortho ON" in name or "Full" in name else ("❌" if "No Ortho" in name or "Backbone" in name else "-")
        mae_str = f"{s['mae']['mean']:.4f} ± {s['mae']['std']:.4f}"
        rmse_str = f"{s['rmse']['mean']:.4f} ± {s['rmse']['std']:.4f}"
        r2_str = f"{s['r2']['mean']:.4f}"
        peak_str = f"{s.get('mae_peak', {}).get('mean', np.nan):.4f}" if 'mae_peak' in s else "N/A"
        time_str = f"{s.get('training_time_seconds', {}).get('mean', np.nan):.1f}s" if 'training_time_seconds' in s else "N/A"
        lines.append(f"| **{name}** | {eeo_flag} | {ortho_flag} | {mae_str} | {rmse_str} | {r2_str} | {peak_str} | {time_str} |")

    lines.append("")
    lines.append("## 2. Factorial Effect Decomposition (Impact Analysis)")
    lines.append("")

    v1 = variants.get('V1 (Full: EEO + Ortho)')
    v2 = variants.get('V2 (Ablate EEO: No EEO, Ortho ON)')
    v3 = variants.get('V3 (Ablate Ortho: EEO ON, No Ortho)')
    v4 = variants.get('V4 (Backbone: No EEO, No Ortho)')

    if v1 and v2 and v3 and v4:
        mae_v1 = v1['mae']['mean']
        mae_v2 = v2['mae']['mean']
        mae_v3 = v3['mae']['mean']
        mae_v4 = v4['mae']['mean']

        # Delta when removing EEO (V1 vs V2, and V3 vs V4)
        eeo_effect_with_ortho = mae_v2 - mae_v1
        eeo_effect_no_ortho   = mae_v4 - mae_v3
        mean_eeo_impact       = (eeo_effect_with_ortho + eeo_effect_no_ortho) / 2.0

        # Delta when removing Ortho (V1 vs V3, and V2 vs V4)
        ortho_effect_with_eeo = mae_v3 - mae_v1
        ortho_effect_no_eeo   = mae_v4 - mae_v2
        mean_ortho_impact     = (ortho_effect_with_eeo + ortho_effect_no_eeo) / 2.0

        # Overall improvement from Backbone V4 to Full V1
        total_synergy = mae_v4 - mae_v1

        lines.append(f"- **Effect of Removing EEO**: MAE increases by {eeo_effect_with_ortho:+.4f} kW (with Ortho) and {eeo_effect_no_ortho:+.4f} kW (without Ortho). Average Impact: **{mean_eeo_impact:+.4f} kW**")
        lines.append(f"- **Effect of Removing Orthogonal Reg**: MAE increases by {ortho_effect_with_eeo:+.4f} kW (with EEO) and {ortho_effect_no_eeo:+.4f} kW (without EEO). Average Impact: **{mean_ortho_impact:+.4f} kW**")
        lines.append(f"- **Full Synergy Gain (V4 -> V1)**: Total MAE improvement of **{total_synergy:+.4f} kW**")
        lines.append("")
        if abs(mean_ortho_impact) > abs(mean_eeo_impact):
            dom = "**Orthogonal Regularization**"
            sec = "**EEO Projections**"
            ratio = abs(mean_ortho_impact) / (abs(mean_eeo_impact) + 1e-8)
        else:
            dom = "**EEO Projections**"
            sec = "**Orthogonal Regularization**"
            ratio = abs(mean_eeo_impact) / (abs(mean_ortho_impact) + 1e-8)

        lines.append(f"> **Scientific Finding**: {dom} exhibits a stronger individual effect on forecasting accuracy compared to {sec} (Relative impact ratio: ~{ratio:.2f}x).")

    report_md = "\n".join(lines)
    report_path = os.path.join(out_dir, "ablation_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nReport written to: {report_path}")
    print("\n" + report_md)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    default_dataset = 'caltech' if 'caltech' in os.environ.get('DATASET_TARGET', 'acn_jpl_ready_v3').lower() else 'jpl'

    parser = argparse.ArgumentParser(description="Run Ablation Study for Model 00 (EEO vs Ortho)")
    parser.add_argument('--dataset', default=default_dataset, choices=['jpl', 'caltech'],
                        help=f"Target dataset configuration (default: {default_dataset})")
    parser.add_argument('--variants', nargs='+', default=['v2_no_eeo', 'v3_no_ortho', 'v4_backbone'],
                        choices=['v1_full', 'v2_no_eeo', 'v3_no_ortho', 'v4_backbone', 'all_missing', 'all'],
                        help="Which variants to execute (default: v2, v3, v4; skipping v1 because results already exist)")
    parser.add_argument('--output-dir', default='.',
                        help="Output directory for ablation results (default: root .)")
    args = parser.parse_args()

    dataset_cfg = DATASET_CONFIGS[args.dataset]

    selected_variants = args.variants
    if 'all_missing' in selected_variants:
        selected_variants = ['v2_no_eeo', 'v3_no_ortho', 'v4_backbone']
    elif 'all' in selected_variants:
        selected_variants = ['v1_full', 'v2_no_eeo', 'v3_no_ortho', 'v4_backbone']

    os.makedirs(args.output_dir, exist_ok=True)
    data_bundle = load_dataset(dataset_cfg)

    for v in selected_variants:
        run_single_variant(v, data_bundle, args.output_dir, dataset_cfg)

    compile_ablation_report(args.output_dir, dataset_cfg)


if __name__ == '__main__':
    main()
