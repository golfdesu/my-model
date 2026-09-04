import json
import sys
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
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
data_path = '../data_cleaned/acn_caltech_ready2.csv'

df = pd.read_csv(data_path)
df['connectionTime'] = pd.to_datetime(df['connectionTime'])
df = df.set_index('connectionTime')
df = df.sort_index()  # safety: enforce chronological order before time-based split

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

# Known-future (calendar) inputs for the TFT decoder - deterministic functions of the
# timestamp, so their values at forecast timesteps are known at inference (TFT paper, IJF 2021)
FUTURE_KNOWN_COLS = [c for c in ['weekend', 'holiday', 'is_business_hour',
                                 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
                                 'Month_sin', 'Month_cos'] if c in df.columns]
X_fk = df[FUTURE_KNOWN_COLS].astype('float32')

X_fk_train = X_fk[:train_len]
X_fk_val   = X_fk[train_len : train_len + val_len]
X_fk_test  = X_fk[train_len + val_len :]

scaler_fk = MinMaxScaler()
X_fk_train_scaled = scaler_fk.fit_transform(X_fk_train)
X_fk_val_scaled   = scaler_fk.transform(X_fk_val)
X_fk_test_scaled  = scaler_fk.transform(X_fk_test)
print(f"Known-future decoder inputs enabled: {len(FUTURE_KNOWN_COLS)} calendar features")

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

# Helper 1b: Future-known windows for the TFT decoder ([N, horizon, F_known])
# aligned exactly with y_seq (same loop bounds as create_windowed_tensors)
def create_future_known_tensors(Xfk_data, lookback, horizon):
    fk_seq = []
    for i in range(len(Xfk_data) - lookback - horizon + 1):
        fk_seq.append(Xfk_data[i + lookback : i + lookback + horizon])
    return torch.tensor(np.array(fk_seq, dtype=np.float32))

# Helper: Positional Embedding Layer in PyTorch
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, d_model)
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        return x + self.pos_emb(positions)

# Helper: PyTorch Gaussian Noise Layer
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

        bias = float(np.mean(predicted - actual))
    negative_pct = float(np.mean(predicted < 0) * 100)

    return dict(mae=mae, rmse=rmse, r2=r2, wape=wape, mape=mape, mae_peak=mae_peak, wape_peak=wape_peak, bias=bias, negative_pct=negative_pct)

# Helper: Gated Residual Network (GRN - TFT Core Component)
class GatedResidualNetwork(nn.Module):
    def __init__(self, in_features, d_model, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(in_features, d_model)
        self.dense2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(in_features, d_model)
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self.res_proj = nn.Linear(in_features, d_model) if in_features != d_model else nn.Identity()

    def forward(self, x):
        a = self.drop(self.dense2(F.elu(self.dense1(x))))
        g = torch.sigmoid(self.gate(x))
        return self.norm(self.res_proj(x) + a * g)

# Helper: Fully Vectorized Variable Selection Network (VSN - TFT Official IJF 2021)
class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features=27, d_model=64, dropout_rate=0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Feature Selection Weights GRN ([B, T, F] -> weights: [B, T, F, 1])
        self.weight_grn = GatedResidualNetwork(num_features, num_features, dropout_rate)

        # Vectorized Feature-Specific GRN Weights (Parallel processing in 4D Tensor)
        self.dense1_w = nn.Parameter(torch.empty(num_features, d_model))
        self.dense1_b = nn.Parameter(torch.empty(num_features, d_model))

        self.dense2_w = nn.Parameter(torch.empty(num_features, d_model, d_model))
        self.dense2_b = nn.Parameter(torch.empty(num_features, d_model))

        self.gate_w = nn.Parameter(torch.empty(num_features, d_model))
        self.gate_b = nn.Parameter(torch.empty(num_features, d_model))

        self.res_w = nn.Parameter(torch.empty(num_features, d_model))
        self.res_b = nn.Parameter(torch.empty(num_features, d_model))

        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.dense1_w, self.gate_w, self.res_w]:
            nn.init.xavier_uniform_(w.unsqueeze(1))
        nn.init.xavier_uniform_(self.dense2_w)
        for b in [self.dense1_b, self.dense2_b, self.gate_b, self.res_b]:
            nn.init.zeros_(b)

    def forward(self, inputs):
        # inputs: [B, T, F]
        weights = torch.softmax(self.weight_grn(inputs), dim=-1).unsqueeze(-1)
        x_unflat = inputs.unsqueeze(-1) # [B, T, F, 1]

        # 1 -> d_model transformations using broadcasting
        w1 = self.dense1_w.view(1, 1, self.num_features, self.d_model)
        b1 = self.dense1_b.view(1, 1, self.num_features, self.d_model)
        d1 = F.elu(x_unflat * w1 + b1) # [B, T, F, d_model]

        # d_model -> d_model transformation
        b2 = self.dense2_b.view(1, 1, self.num_features, self.d_model)
        d2_linear = torch.einsum('btfi,fio->btfo', d1, self.dense2_w)
        d2 = self.drop(d2_linear + b2)

        wg = self.gate_w.view(1, 1, self.num_features, self.d_model)
        bg = self.gate_b.view(1, 1, self.num_features, self.d_model)
        g = torch.sigmoid(x_unflat * wg + bg)

        wr = self.res_w.view(1, 1, self.num_features, self.d_model)
        br = self.res_b.view(1, 1, self.num_features, self.d_model)
        res = x_unflat * wr + br

        processed = self.norm(res + d2 * g) # [B, T, F, d_model]
        return torch.sum(processed * weights, dim=2) # [B, T, d_model]

# Helper: Vectorized Pinball (Quantile) Loss Function for TFT (P10, P50, P90)
def pinball_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    # y_pred: [batch, horizon, len(quantiles)]
    # y_true: [batch, horizon]
    error = y_true.unsqueeze(-1) - y_pred
    q = torch.tensor(quantiles, device=y_pred.device).view(1, 1, -1)
    return torch.mean(torch.max((q - 1) * error, q * error))

# Helper: Interpretable Multi-Head Attention (TFT paper, IJF 2021 Sec 3.2)
# - A single shared value matrix W_V is used by all heads
# - Attention outputs are averaged over heads before the final projection W_O
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_k)      # shared V across heads -> [.., d_attn]
        self.out_proj = nn.Linear(self.d_k, d_model)    # W_O: d_attn -> d_model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, attn_mask=None):
        B, L_q, _ = query.shape
        L_k = key.size(1)
        qh = self.q_proj(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,Lq,d_k]
        kh = self.k_proj(key).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)    # [B,H,Lk,d_k]
        v_shared = self.v_proj(value)                                                    # [B,Lk,d_attn]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / (self.d_k ** 0.5)              # [B,H,Lq,Lk]
        if attn_mask is not None:
            scores = scores + attn_mask.to(scores.dtype)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(self.dropout(attn), v_shared.unsqueeze(1))               # [B,H,Lq,d_attn]
        attn_avg = head_out.mean(dim=1)                                                  # average over heads
        return self.out_proj(attn_avg), None

# Helper: TFT Architecture PyTorch Module (Official IJF 2021 Seq2Seq TFT with Quantiles)
class TFTModel(nn.Module):
    def __init__(self, lookback, num_features, horizon, num_future_known=9, d_model=64, num_heads=4, num_layers=1, dropout_rate=0.1, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_features = num_features
        self.quantiles = quantiles

        # 1. Variable Selection Networks for Encoder & Decoder
        self.vsn_enc = VariableSelectionNetwork(num_features, d_model, dropout_rate)
        self.vsn_dec = VariableSelectionNetwork(num_future_known, d_model, dropout_rate)

        # 2. Locality Processing: Seq2Seq LSTM (Encoder & Decoder LSTM)
        self.lstm_enc = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.lstm_dec = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)

        # 3. Interpretable Temporal Multi-Head Attention over Full Sequence
        self.mha = InterpretableMultiHeadAttention(d_model, num_heads, dropout_rate)
        self.drop_attn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.LayerNorm(d_model)

        # Register Persistent Buffer for Causal Mask
        total_len = lookback + horizon
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((total_len, total_len), float('-inf')), diagonal=1),
            persistent=False
        )

        # 4. Post-Attention Gated Residual Network & Quantile Output Projection Layer
        self.grn_post = GatedResidualNetwork(d_model, d_model, dropout_rate)
        self.out_head = nn.Linear(d_model, len(quantiles))

    def forward(self, x, x_future):
        # x: [batch, lookback, num_features] past-observed inputs
        # x_future: [batch, horizon, num_future_known] known-future calendar inputs

        # 1. Variable Selection
        vsn_enc_out = self.vsn_enc(x)                  # [batch, lookback, d_model]
        vsn_dec_out = self.vsn_dec(x_future)           # [batch, horizon, d_model]

        # 2. Seq2Seq LSTM Processing
        enc_out, (h_n, c_n) = self.lstm_enc(vsn_enc_out)
        dec_out, _ = self.lstm_dec(vsn_dec_out, (h_n, c_n))

        # Concatenate Encoder & Decoder sequences -> [batch, lookback + horizon, d_model]
        full_seq = torch.cat([enc_out, dec_out], dim=1)

        # 3. Causal Interpretable Self-Attention over Full Sequence using Buffer Mask
        attn_out, _ = self.mha(full_seq, full_seq, full_seq, attn_mask=self.causal_mask)
        norm_seq = self.norm_attn(full_seq + self.drop_attn(attn_out))

        # 4. Post-Attention GRN on Decoder Horizon Portion
        dec_norm = norm_seq[:, -self.horizon:, :]      # [batch, horizon, d_model]
        grn_out = self.grn_post(dec_norm)             # [batch, horizon, d_model]

        # 5. Quantile Output Projection per Horizon Step -> [batch, horizon, 3] (P10, P50, P90)
        out = self.out_head(grn_out)
        return out

import time
# Config Parameters
LOOKBACK = 96      # 48 hours history (96 * 30 min)
HORIZON = 48       # 24 hours forecast (48 * 30 min)
BATCH_SIZE = 256
SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]
output_json_filename = "04_tfm_tft_pytorch_results.json"
results_data = {
    "model_name": "04_tfm_tft_pytorch",
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

FK_train_t = create_future_known_tensors(X_fk_train_scaled, LOOKBACK, HORIZON)
FK_val_t   = create_future_known_tensors(X_fk_val_scaled, LOOKBACK, HORIZON)
FK_test_t  = create_future_known_tensors(X_fk_test_scaled, LOOKBACK, HORIZON)

train_dataset = TensorDataset(X_train_t, FK_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, FK_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, FK_test_t, y_test_t)

print(f"Starting Automated {len(SEEDS)}-Seed Loop for 04_tfm_tft_pytorch in PyTorch...")

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

    # Pre-built Dataset DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(device.type == 'cuda'))

    # Build Model
    QUANTILES = [0.1, 0.5, 0.9]
    P50_IDX = 1  # Index of quantile 0.5 (median) for baseline comparison
    model = TFTModel(lookback=LOOKBACK, num_features=X_train_scaled.shape[1], horizon=HORIZON,
                     num_future_known=len(FUTURE_KNOWN_COLS), quantiles=QUANTILES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    # Training Loop with Early Stopping & Single Outer tqdm Progress Bar (%)
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history   = []
    best_epoch         = 1
    patience_counter = 0
    best_model_weights = None

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Seed {SEED} Training", leave=True)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for batch_X, batch_fk, batch_y in train_loader:
            batch_X, batch_fk, batch_y = batch_X.to(device, non_blocking=True), batch_fk.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_X, batch_fk)
            loss = pinball_loss(out, batch_y, QUANTILES)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch_X, batch_fk, batch_y in val_loader:
                batch_X, batch_fk, batch_y = batch_X.to(device, non_blocking=True), batch_fk.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                out = model(batch_X, batch_fk)
                loss = pinball_loss(out, batch_y, QUANTILES)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        # Dynamic tqdm bar update
        epoch_pbar.set_postfix({'train_loss': f"{train_loss:.5f}", 'val_loss': f"{val_loss:.5f}"} )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
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
        for batch_X, batch_fk, _ in test_loader:
            batch_X, batch_fk = batch_X.to(device), batch_fk.to(device)
            out = model(batch_X, batch_fk)
            y_pred_list.append(out.cpu().numpy())
    
    y_pred_all = np.vstack(y_pred_list)          # [N, horizon, 3] (P10, P50, P90)
    y_pred_scaled = y_pred_all[:, :, P50_IDX]    # [N, horizon] (P50 median for evaluation)

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

    # 48-step full horizon evaluation (24-hour error degradation)
    mae_48 = [float(mean_absolute_error(y_test_seq_unscaled[:, s], y_pred_unscaled[:, s])) for s in range(HORIZON)]
    rmse_48 = [float(np.sqrt(mean_squared_error(y_test_seq_unscaled[:, s], y_pred_unscaled[:, s]))) for s in range(HORIZON)]

    all_predictions[f"seed_{SEED}"] = y_pred_unscaled.astype(np.float32)

    if best_val_loss < best_overall_val_loss and best_model_weights is not None:
        best_overall_val_loss = best_val_loss
        best_seed_id = SEED
        torch.save(best_model_weights, f"04_tfm_tft_pytorch_best.pt")
        results_data["best_seed"] = int(SEED)
        print(f"  [Checkpoint] New overall best model saved from SEED {SEED} (Val Loss: {best_val_loss:.6f}) -> 04_tfm_tft_pytorch_best.pt")

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
np.savez_compressed(f"04_tfm_tft_pytorch_predictions.npz", **all_predictions)
print(f"Successfully saved all seed predictions to 04_tfm_tft_pytorch_predictions.npz")

print(f"\n======================================================================")
print(f"FINAL SUMMARY ACROSS {len(SEEDS)} SEEDS — 04_tfm_tft_pytorch")
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
    "batch_size": BATCH_SIZE if 'BATCH_SIZE' in globals() or 'BATCH_SIZE' in locals() else None,
    "seeds": SEEDS if 'SEEDS' in globals() or 'SEEDS' in locals() else None,
    "total_parameters": results_data.get("total_parameters", None)
}
results_data["summary"] = summary_dict
with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
print(f"Successfully saved final results to {output_json_filename}")
print(f"\nFinished running all {len(SEEDS)} SEEDs in PyTorch!")

