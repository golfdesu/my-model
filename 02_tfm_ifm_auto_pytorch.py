import sys
import os
import sys
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
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load and clean dataset (Local path auto-detect for VS Code)
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

# Feature Scaling (MinMaxScaler)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

# Target Scaling (MinMaxScaler for y)
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

# Helper: Positional Embedding Layer in PyTorch
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, d_model)
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        return x + self.pos_emb(positions)

# Helper: PyTorch Gaussian Noise Layer
class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.01):
        super().__init__()
        self.stddev = stddev
    def forward(self, x):
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

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

# Helper: Distillation Layer (Informer Core Component)
class DistillLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x_tr = x.transpose(1, 2)
        c = self.act(self.conv(x_tr))
        c = self.norm(c.transpose(1, 2)).transpose(1, 2)
        p = self.pool(c)
        return p.transpose(1, 2)

# Helper: ProbSparse Self-Attention (Informer Core Component)
class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_sparse_scores(self, queries, keys, sample_k, n_top):
        B, H, L_K, E = keys.shape
        _, _, L_Q, _ = queries.shape
        K_sample = keys[:, :, torch.randint(0, L_K, (sample_k,), device=queries.device), :]
        Q_K_sample = torch.matmul(queries, K_sample.transpose(-2, -1))
        M = Q_K_sample.max(dim=-1)[0] - torch.div(Q_K_sample.sum(dim=-1), L_K)
        M_top = M.topk(n_top, dim=-1, sorted=False)[1]
        M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, E)
        Q_reduce = torch.gather(queries, 2, M_top_expanded)
        return Q_reduce, M_top

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries_p = queries.permute(0, 2, 1, 3)
        keys_p = keys.permute(0, 2, 1, 3)
        values_p = values.permute(0, 2, 1, 3)
        U_part = min(max(1, int(self.factor * np.ceil(np.log(L_K)))), L_K)
        u = min(max(1, int(self.factor * np.ceil(np.log(L_Q)))), L_Q)
        Q_reduce, M_top = self._prob_sparse_scores(queries_p, keys_p, U_part, u)
        scale = self.scale or 1.0 / np.sqrt(D)
        scores_top = torch.matmul(Q_reduce, keys_p.transpose(-2, -1)) * scale
        attn_top = torch.softmax(scores_top, dim=-1)
        V_reduce = torch.matmul(self.dropout(attn_top), values_p)
        V_mean = values_p.mean(dim=-2, keepdim=True)  # official Informer: V.mean(dim=-2) for non-top-u queries
        context = V_mean.expand(B, H, L_Q, D).clone()
        M_top_expanded = M_top.unsqueeze(-1).expand(-1, -1, -1, D)
        context.scatter_(2, M_top_expanded, V_reduce)
        return context.permute(0, 2, 1, 3).contiguous(), None

class ProbSparseAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# Helper: Informer Architecture PyTorch Module (Official AAAI 2021)
class InformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.enc_proj = nn.Linear(num_features, d_model)
        self.pos_emb_enc = PositionalEmbedding(lookback, d_model)
        self.drop_enc = nn.Dropout(dropout_rate)

        self.num_layers = num_layers
        self.enc_attn = nn.ModuleList([
            ProbSparseAttentionLayer(ProbAttention(factor=5, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm1_enc = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_layers)
        ])
        self.norm2_enc = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.distill = nn.ModuleList([DistillLayer(d_model) for _ in range(num_layers - 1)])
        self.drop = nn.Dropout(dropout_rate)

        # Generative Decoder Components (Zero Placeholder Padding)
        dec_seq_len = lookback // 4 + horizon
        self.pos_emb_dec = PositionalEmbedding(dec_seq_len, d_model)
        self.dec_attn = ProbSparseAttentionLayer(ProbAttention(factor=5, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.norm1_dec = nn.LayerNorm(d_model)
        self.cross_attn = ProbSparseAttentionLayer(ProbAttention(factor=5, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.norm2_dec = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model * horizon, horizon)

    def forward(self, x):
        # x: [batch, lookback, num_features]
        batch_size = x.size(0)
        enc = self.drop_enc(self.pos_emb_enc(self.enc_proj(x)))

        for i in range(self.num_layers):
            attn_out, _ = self.enc_attn[i](enc, enc, enc)
            enc = self.norm1_enc[i](enc + self.drop(attn_out))
            ffn_out = self.ffn_enc[i](enc)
            enc = self.norm2_enc[i](enc + self.drop(ffn_out))
            if i < self.num_layers - 1:
                enc = self.distill[i](enc)

        # Generative-Style Decoder Construction: Start token + Zero placeholder
        start_token = enc[:, -self.lookback//4:, :]
        zero_placeholder = torch.zeros(batch_size, self.horizon, enc.size(-1), device=x.device)
        dec_in = torch.cat([start_token, zero_placeholder], dim=1)
        dec = self.pos_emb_dec(dec_in)

        dec_attn_out, _ = self.dec_attn(dec, dec, dec)
        dec = self.norm1_dec(dec + self.drop(dec_attn_out))
        cross_attn_out, _ = self.cross_attn(queries=dec, keys=enc, values=enc)
        dec = self.norm2_dec(dec + self.drop(cross_attn_out))

        dec_target = dec[:, -self.horizon:, :]
        out = self.out_head(dec_target.reshape(batch_size, -1))
        return out


import time
# Config Parameters
LOOKBACK = 96      # 48 hours history (96 * 30 min)
HORIZON = 48       # 24 hours forecast (48 * 30 min)
BATCH_SIZE = 256
SEEDS = [164, 256, 355, 1234, 2026]
output_md_filename = "02_tfm_ifm_pytorch.md"
steps_to_eval = [0, 5, 11, 47]
step_labels = {0: 'Step 0 (30 min)', 5: 'Step 5 (3 hr)', 11: 'Step 11 (6 hr)', 47: 'Step 47 (24 hr)'}

print('Pre-building sequence tensors...')
X_train_t, y_train_t, _, _ = create_windowed_tensors(X_train_scaled, y_train_scaled, LOOKBACK, HORIZON)
X_val_t, y_val_t, _, _     = create_windowed_tensors(X_val_scaled, y_val_scaled, LOOKBACK, HORIZON)
X_test_t, y_test_t, X_test_seq, y_test_seq = create_windowed_tensors(X_test_scaled, y_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

print(f"Starting Automated {len(SEEDS)}-Seed Loop for {output_md_filename} in PyTorch...")

for seed_idx, SEED in enumerate(SEEDS, 1):
    print(f"\n=========================================================================")
    print(f"RUNNING SEED {SEED} ({seed_idx}/{len(SEEDS)})")
    print(f"=========================================================================")

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # Pre-built Dataset DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    # Build Model
    model = InformerModel(lookback=LOOKBACK, num_features=X_train_scaled.shape[1], horizon=HORIZON).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    # Training Loop with Early Stopping & Single Outer tqdm Progress Bar (%)
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
            time.sleep(0.005)  # Rest GPU per batch to keep temperature cool
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                out = model(batch_X)
                loss = criterion(out, batch_y)
                time.sleep(0.002)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # Dynamic tqdm bar update
        epoch_pbar.set_postfix({'train_loss': f"{train_loss:.5f}", 'val_loss': f"{val_loss:.5f}"} )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            best_model_weights = {k: v.cpu().clone() for k, v in target_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_pbar.write(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break

    if best_model_weights is not None:
        target_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        target_model.load_state_dict(best_model_weights)

    # Inference on Test Set
    model.eval()
    y_pred_list = []
    with torch.inference_mode():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            out = model(batch_X)
            y_pred_list.append(out.cpu().numpy())
    
    y_pred_scaled = np.vstack(y_pred_list)

    # Inverse transform predictions and actual values back to kW scale
    y_test_seq_unscaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    y_pred_unscaled     = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    # Extract step vectors
    actual_by_step = {step: y_test_seq_unscaled[:, step] for step in steps_to_eval}
    predictions_by_step = {step: y_pred_unscaled[:, step] for step in steps_to_eval}

    # Build Markdown evaluation output text
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

    output_lines.append("================================================================Match=================\n")
    full_output_text = "\n".join(output_lines)
    print(full_output_text)

    # Append to markdown file
    with open(output_md_filename, "a", encoding="utf-8") as f:
        f.write(full_output_text + "\n")
        
    print(f"Successfully saved SEED {SEED} metrics to {output_md_filename}")
    gc.collect()

print(f"\nFinished running all {len(SEEDS)} SEEDs in PyTorch!")

