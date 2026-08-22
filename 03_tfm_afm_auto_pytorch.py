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
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # VRAM Limit 50%
else:
    print(f"CPU Multithreading Optimized with {num_cpus} threads")

# Auto-load MSVC environment (PATH, INCLUDE, LIB) for torch.compile()
vcvars_path = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if os.path.exists(vcvars_path):
    msvc_env = subprocess.check_output(f'cmd.exe /c ""{vcvars_path}" && set"', text=True)
    for line in msvc_env.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v


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

# Helper: Series Decomposition Layer (Autoformer Core Component)
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x_tr = x.transpose(1, 2)
        trend = self.avg_pool(x_tr).transpose(1, 2)
        if trend.size(1) > x.size(1):
            trend = trend[:, :x.size(1), :]
        elif trend.size(1) < x.size(1):
            trend = F.pad(trend, (0, 0, 0, x.size(1) - trend.size(1)))
        seasonal = x - trend
        return seasonal, trend

# Helper: Auto-Correlation Mechanism (Autoformer Core Component)
class AutoCorrelation(nn.Module):
    def __init__(self, factor=1, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg(self, values, corr):
        batch, head, length, channel = values.shape
        top_k = max(1, int(self.factor * np.log(length)))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, index = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        
        tmp_values = values.repeat(1, 1, 2, 1)
        delays_agg = torch.zeros_like(values)
        time_seq = torch.arange(length, device=values.device).unsqueeze(0)
        for i in range(top_k):
            pattern = tmp_corr[:, i].view(batch, 1, 1, 1)
            offsets = (index[:, i].unsqueeze(1) + time_seq).unsqueeze(1).unsqueeze(-1).expand(batch, head, length, channel)
            sliced_values = torch.gather(tmp_values, 2, offsets)
            delays_agg = delays_agg + pattern * sliced_values
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        orig_L = L
        if L > S:
            zeros = torch.zeros(B, L - S, H, D, device=queries.device)
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        elif L < S:
            zeros = torch.zeros(B, S - L, H, E, device=queries.device)
            queries = torch.cat([queries, zeros], dim=1)

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        values_perm = values.permute(0, 2, 1, 3)
        corr_perm = corr.permute(0, 1, 3, 2)
        out = self.time_delay_agg(values_perm, corr_perm)
        out = out.permute(0, 2, 1, 3).contiguous()
        if orig_L < S:
            out = out[:, :orig_L, :, :]
        return out, None

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_correlation = correlation
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
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# Helper: Autoformer Architecture PyTorch Module
class AutoformerModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.proj = nn.Linear(num_features, d_model)
        self.decomp_init = SeriesDecomp(kernel_size=25)

        self.num_layers = num_layers
        self.enc_attn = nn.ModuleList([
            AutoCorrelationLayer(AutoCorrelation(factor=1, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.decomp1_enc = nn.ModuleList([SeriesDecomp(kernel_size=25) for _ in range(num_layers)])
        self.ffn_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_layers)
        ])
        self.decomp2_enc = nn.ModuleList([SeriesDecomp(kernel_size=25) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout_rate)

        # Decoder Auto-Correlation & decomp
        self.cross_attn = AutoCorrelationLayer(AutoCorrelation(factor=1, attention_dropout=dropout_rate), d_model=d_model, n_heads=num_heads)
        self.decomp_dec = SeriesDecomp(kernel_size=25)
        self.out_head = nn.Linear(d_model * horizon, horizon)

    def forward(self, x):
        # x: [batch, lookback, num_features]
        batch_size = x.size(0)
        x_proj = self.proj(x)
        seasonal_enc, trend_enc = self.decomp_init(x_proj)

        for i in range(self.num_layers):
            attn_out, _ = self.enc_attn[i](seasonal_enc, seasonal_enc, seasonal_enc)
            seasonal_enc, _ = self.decomp1_enc[i](seasonal_enc + self.drop(attn_out))
            ffn_out = self.ffn_enc[i](seasonal_enc)
            seasonal_enc, _ = self.decomp2_enc[i](seasonal_enc + self.drop(ffn_out))

        trend_part = trend_enc[:, -1, :].unsqueeze(1).repeat(1, self.horizon, 1)
        seasonal_part = seasonal_enc[:, -1, :].unsqueeze(1).repeat(1, self.horizon, 1)

        cross_attn_out, _ = self.cross_attn(queries=seasonal_part, keys=seasonal_enc, values=seasonal_enc)
        seasonal_dec = seasonal_part + self.drop(cross_attn_out)
        seasonal_dec, trend_extra = self.decomp_dec(seasonal_dec)
        trend_part = trend_part + trend_extra

        combined = seasonal_dec + trend_part
        out = self.out_head(combined.reshape(batch_size, -1))
        return out


import time
# Config Parameters
LOOKBACK = 96      # 48 hours history (96 * 30 min)
HORIZON = 48       # 24 hours forecast (48 * 30 min)
BATCH_SIZE = 256
SEEDS = [164, 256, 355, 1234, 2026]
output_md_filename = "03_tfm_afm_pytorch.md"
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Build Model
    model = AutoformerModel(lookback=LOOKBACK, num_features=X_train_scaled.shape[1], horizon=HORIZON).to(device)

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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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
            batch_X = batch_X.to(device)
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

