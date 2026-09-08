#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Multi-Seed Benchmark for Model 25: TiDE (Time-series Dense Encoder)
# Reference: Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder",
#            TMLR 2023. https://arxiv.org/abs/2304.08424 (Google Research)
#
# Standardized Benchmark Protocol:
# - Lookback (L) = 96, Horizon (H) = 48
# - Chronological Split: 60% Train, 20% Val, 20% Test
# - 10 Deterministic Seeds: [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
# - Target: kWhDelivered (EV aggregate station load)
# ==============================================================================

import os
import sys
import gc
import json
import math
import time
import random
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

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

num_cpus = os.cpu_count() or 4
torch.set_num_threads(min(6, num_cpus))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
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
data_path = 'data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = '../data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = '../../data_cleaned/acn_jpl_ready.csv'
if not os.path.exists(data_path):
    data_path = 'acn_jpl_ready.csv'

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

peak_threshold_kw = float(np.percentile(df['kWhDelivered'].iloc[:train_len], 80))

# ---------------------------------------------------------
# 2. Windowing Function
# ---------------------------------------------------------
def create_windowed_tensors(X_data, y_data, lookback=96, horizon=48):
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

# ---------------------------------------------------------
# 3. Model Architecture: TiDE
# ---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        h = self.fc2(self.drop(self.relu(self.fc1(x))))
        return self.ln(h + res)


class TiDE(nn.Module):
    def __init__(
        self,
        lookback=96,
        num_features=30,
        horizon=48,
        target_idx=29,
        d_feat=16,
        d_hidden=256,
        d_dec=64,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.target_idx = target_idx
        self.num_covariates = num_features - 1

        self.feature_proj = nn.Sequential(
            nn.Linear(self.num_covariates, d_feat),
            nn.ReLU()
        )

        encoder_input_dim = lookback * (1 + d_feat)
        enc_layers = []
        in_d = encoder_input_dim
        for _ in range(num_encoder_layers):
            enc_layers.append(ResBlock(in_d, d_hidden, d_hidden, dropout=dropout))
            in_d = d_hidden
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for _ in range(num_decoder_layers):
            dec_layers.append(ResBlock(d_hidden, d_hidden, d_hidden, dropout=dropout))
        self.decoder = nn.Sequential(*dec_layers)
        self.decoder_proj = nn.Linear(d_hidden, horizon * d_dec)
        self.d_dec = d_dec

        self.temporal_head = ResBlock(d_dec, 1, d_dec, dropout=dropout)
        self.global_skip = nn.Linear(lookback, horizon)

    def forward(self, x):
        B, L, _ = x.shape
        cov_idx = [i for i in range(x.shape[-1]) if i != self.target_idx]
        covariates = x[:, :, cov_idx]
        past_target = x[:, :, self.target_idx]

        proj_cov = self.feature_proj(covariates)
        combined = torch.cat([past_target.unsqueeze(-1), proj_cov], dim=-1)
        flat_input = combined.reshape(B, -1)

        e = self.encoder(flat_input)
        d = self.decoder(e)
        g = self.decoder_proj(d)
        g = g.reshape(B, self.horizon, self.d_dec)

        dense_out = self.temporal_head(g).squeeze(-1)
        skip_out = self.global_skip(past_target)

        return dense_out + skip_out

TiDEModel = TiDE

# ---------------------------------------------------------
# 4. Training, Evaluation & Metrics
# ---------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []
    with torch.inference_mode():
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
MODEL_NAME = "25_tide_baseline_pytorch"
OUTPUT_DIR = f"outputs/{MODEL_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "lookback": LOOKBACK,
    "num_features": num_total_features,
    "horizon": HORIZON,
    "target_idx": TARGET_CH_IDX,
    "d_feat": 16,
    "d_hidden": 256,
    "d_dec": 64,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.1,
    "learning_rate": 0.00136080647750805,
    "weight_decay": 0.0002576417442523316,
    "batch_size": 128,
    "epochs": 200,
    "patience": 15}

def run_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    model = TiDE(
        lookback=CONFIG["lookback"],
        num_features=CONFIG["num_features"],
        horizon=CONFIG["horizon"],
        target_idx=CONFIG["target_idx"],
        d_feat=CONFIG["d_feat"],
        d_hidden=CONFIG["d_hidden"],
        d_dec=CONFIG["d_dec"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        dropout=CONFIG["dropout"]
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

    epoch_pbar = tqdm(range(1, CONFIG["epochs"] + 1), desc=f"Seed {seed} Training", leave=True)
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        epoch_pbar.set_postfix({'train_loss': f"{train_loss:.5f}", 'val_loss': f"{val_loss:.5f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                epoch_pbar.write(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
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

        with open(output_json_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
        with open(root_json_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)

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
