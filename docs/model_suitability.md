# Model Suitability Analysis — ACN Caltech EV Charging Dataset

> Analysis date: 2026-08-19  
> Dataset: `acn_caltech_ready2.csv`  
> Task: **Multi-step ahead load forecasting** (LOOKBACK = 96 steps / 48 h, HORIZON = 48 steps / 24 h)

---

## 1. Dataset Characteristics

| Property | Value |
|---|---|
| **Total rows** | 32,435 timesteps |
| **Sampling interval** | 30 minutes |
| **Date range** | 2018-05-10 → 2020-03-16 (~22 months) |
| **Input features** | 30 (weather, lag, rolling stats, calendar encodings) |
| **Target** | `kWhDelivered` (EV station aggregate load, kWh) |
| **Time-series gaps** | **0** — perfectly continuous |
| **Zero values** | **7,083 / 32,435 = 21.8%** (no-EV periods) |
| **Target distribution** | Highly right-skewed — P10=0.0, P50=7.0, P90=43.0, Max=124.8 kWh |
| **Daily seasonality (ACF Lag 48)** | **0.714** — strong 24-hour cycle |
| **Non-stationarity** | **Pronounced** — chunk mean drifts from ~23 kWh (early 2018) down to ~9 kWh (early 2020) |

### Key ACF Values

| Lag (steps) | Wall-clock | ACF |
|---|---|---|
| 1 | 30 min | 0.944 |
| 2 | 1 h | 0.861 |
| 4 | 2 h | 0.664 |
| 24 | 12 h | −0.070 |
| **48** | **24 h** | **0.714** ← dominant seasonal period |
| 96 | 48 h | 0.589 |
| 144 | 72 h | 0.579 |

### Non-stationarity (Chunk-wise Statistics)

| Chunk | Period | Mean (kWh) | Std (kWh) |
|---|---|---|---|
| 1 | Early | 22.93 | 18.15 |
| 2 | | 27.66 | 21.55 |
| 3 | | 24.43 | 24.30 |
| 4 | | 12.80 | 15.90 |
| 5 | | 13.06 | 15.56 |
| 6 | | 11.27 | 14.96 |
| 7 | | 8.98 | 12.50 |
| 8 | | 10.80 | 15.07 |
| 9 | | 7.98 | 13.17 |
| 10 | Late | 8.94 | 14.30 |

The mean load decreases by ~61% across the observation window, indicating a strong downward trend (likely related to fleet composition changes or seasonality across years).

---

## 2. Model Suitability Summary

| # | Model | Input Type | Suitability | Notes |
|---|---|---|---|---|
| 01 | **Vanilla Transformer** | Multivariate (sequence) | ✅✅ Strong | See §3.5 |
| 02 | **Informer** | Multivariate (sequence) | ✅✅ Strong | See §3.6 |
| 03 | **Autoformer** | Multivariate (sequence) | ✅✅ Strong | See §3.7 |
| 04 | **TFT** | Multivariate (sequence) | ✅✅ Strong | See §3.8 |
| 05 | **PatchTST** | Multivariate (sequence) | ✅✅ Strong | See §3.9 |
| 06 | **Decoder-Only Transformer** | Multivariate (sequence) | ✅ Good | See §3.10 |
| 07 | **Encoder-Decoder Transformer** | Multivariate (sequence) | ✅ Good | See §3.10 |
| 08 | **LSTM** | Multivariate (sequence) | ✅ Good | See §3.4 |
| 09 | **DLinear** | Univariate | ⚠️ Limited | See §3.2 |
| 10 | **XGBoost** | Multivariate (tabular) | ✅ Good | See §3.3 |
| 11 | **LightGBM** | Multivariate (tabular) | ✅ Good | See §3.3 |
| 12 | **SARIMA** | Univariate | ❌ Unsuitable | See §3.1 |
| 13 | **iTransformer** | Multivariate (sequence) | ✅✅ Strong | See §3.11 |
| 14 | **TimesNet** | Multivariate (sequence) | ✅✅ Strong | See §3.12 |
| 15 | **NLinear** | Univariate | ⚠️ Limited | See §3.13 |

---

## 3. Detailed Analysis Per Model

### 3.1 SARIMA ❌ Unsuitable

**Reasons why SARIMA is a poor fit for this dataset:**

1. **Strong non-stationarity** — The conditional mean drifts from ~23 kWh to ~9 kWh over 22 months. Even with seasonal differencing `(d=1, D=1)`, SARIMA assumes the differenced series is stationary. The drift in variance (Std: 18 → 12) additionally violates this assumption.

2. **Univariate only** — SARIMA cannot incorporate the 30 engineered features (weather, `kWhLag_*`, calendar encodings, rolling statistics). These covariates contain highly predictive signal that is entirely discarded.

3. **Sparse zero values (21.8%)** — SARIMA's likelihood estimation assumes a Gaussian error structure. Extended zero-load periods (782 runs, average 9 steps / ~4.5 h, max 106 steps / 53 h) systematically violate this, biasing parameter estimates.

4. **Walk-forward inference is prohibitively slow** — Each full SARIMA refit on a growing window takes 20–90+ seconds. Evaluating even a small subset of the test set requires hours of compute time, making it impractical for iterative experimentation.

5. **HPO is of limited value** — The ARIMA order search space `(p, d, q) × (P, D, Q)` is small and its gains are modest relative to the fundamental structural mismatches above.

> **Role in this project:** Use SARIMA as a **statistical lower-bound benchmark only**. Its results are expected to be the weakest of all models, establishing the floor above which all ML/DL models should improve.

---

### 3.2 DLinear ⚠️ Limited

**Reasons for limited suitability:**

1. **Univariate by design** — The original DLinear paper (Zeng et al., AAAI 2023) uses only the target variable's own history. All 30 covariates (weather, calendar, engineered lags) are discarded, leaving substantial predictive information unused.

2. **Non-stationarity weakens linear projections** — The shared linear weights `Linear_trend` and `Linear_seasonal` are optimized globally across training. With a 61% drift in mean load, weights fit to early (high-load) periods generalize poorly to late (low-load) periods.

3. **No mechanism to handle sparsity** — The model cannot distinguish between "zero because no EV is connected" and "zero because of a genuine load dip," which are semantically different.

**Strengths that partially offset these issues:**
- Extremely fast training and inference
- Nearly zero hyperparameters → negligible tuning cost
- Serves as an interpretability reference point

> **Role in this project:** Use DLinear as a **model-complexity benchmark** — it answers the question *"does attention/recurrence actually help over a simple linear decomposition?"* Report results with a clear note that the model uses univariate input only, which is an intentional design choice matching the original paper.

---

### 3.3 XGBoost & LightGBM ✅ Good

**Why tree ensembles are well-suited:**

1. **Native multivariate support** — All 30 features are consumed directly. Engineered lag features (`kWhLag_48`, `kWhLag_96`, `kWhMean_48`, `kWhMean_672`) are exactly the type of features tree-based models exploit best.

2. **Robustness to non-stationarity** — Per-split decision boundaries adapt locally; there is no global parameter that drifts with the changing mean.

3. **Handles skewed, sparse targets gracefully** — Tree splits based on `reg:squarederror` / `mae` objective are not distorted by zero-inflation in the way Gaussian likelihood methods are.

4. **Early stopping on validation MAE** prevents overfitting without manual epoch tuning.

**Key difference between the two:**
- **LightGBM** grows leaf-wise (faster, better on large feature sets, preferred for this dataset size)
- **XGBoost** grows level-wise (slightly more regularized, comparable accuracy at higher compute cost)

> **Role in this project:** Primary **ML baseline** representing the best achievable performance without sequential modelling. Acts as the bar that deep sequence models must clear to justify their added complexity.

---

### 3.4 LSTM ✅ Good

**Suitability arguments:**
- Sequential hidden state naturally captures the strong short-lag autocorrelation (ACF Lag 1 = 0.944, Lag 2 = 0.861)
- `nn.LSTM` with 2–3 layers and dropout handles the 30-dimensional feature input well
- Dual aggregation head (last-step + global mean pooling) is well-matched to the HORIZON=48 direct prediction setup

**Limitations relative to attention-based models:**
- Fixed hidden state bottleneck may compress the 96-step lookback less efficiently than multi-head attention
- No built-in decomposition mechanism to address the non-stationarity

> **Role in this project:** **Deep learning baseline** — isolates the contribution of *attention* by holding everything else constant vs. the Vanilla Transformer (same head structure, same feature input, only the sequence encoder differs).

---

### 3.5 Vanilla Transformer (Encoder-Only) ✅✅ Strong

**Why it fits well:**
- Multi-head self-attention captures both short-range dependencies (high ACF at Lag 1–4) and the dominant 24-hour periodic pattern simultaneously
- LOOKBACK=96 steps = exactly 2 × seasonal period (s=48), giving the encoder full visibility of two complete daily cycles
- `d_model`, `num_heads`, `d_ff`, `num_layers` hyperparameters are efficiently explored via Optuna

**Minor concern:**
- Vanilla encoder without decomposition may struggle with the pronounced non-stationarity; RevIN or instance normalization could help if validation loss plateaus

> **Role:** Primary deep learning benchmark for attention-based forecasting.

---

### 3.6 Informer ✅✅ Strong

**Specific advantages for this dataset:**
- ProbSparse self-attention reduces memory and compute for LOOKBACK=96 with minimal accuracy loss
- Distillation layers progressively compress the encoder representation — appropriate given that ACF decays significantly by Lag 24
- Encoder–decoder structure is natural for 48-step multi-step horizon forecasting

**Potential concern:**
- Distillation halves sequence length at each layer; with `num_layers=2`, the encoded sequence is compressed to 96/4=24 steps, which may discard some useful sub-daily patterns

---

### 3.7 Autoformer ✅✅ Strong

**Specific advantages for this dataset:**
- **Series decomposition** (moving-average trend/seasonal split) directly addresses the non-stationarity identified in the data: trend component captures the multi-month mean drift, seasonal component isolates the daily 48-step cycle
- Auto-Correlation mechanism computes period-based attention naturally aligned with the daily seasonality (s=48)

> **Role:** Best architecturally motivated model for this dataset's non-stationarity + strong daily seasonality combination. Expected to be among the top performers.

---

### 3.8 TFT (Temporal Fusion Transformer) ✅✅ Strong

**Specific advantages:**
- Variable Selection Network (VSN) learns to downweight irrelevant features from the 30-dimensional input, mitigating noise from weakly correlated weather variables
- Gated Residual Networks (GRN) allow nonlinear transformations per feature type before temporal modelling
- Quantile output (P10, P50, P90) provides calibrated uncertainty estimates — directly useful for EV charging station capacity planning

**Implementation note:**
> The current implementation (`04_tfm_tft_auto_pytorch.py`) uses `dec_placeholder = x[:, -1:, :]` for the decoder input, rather than injecting known-future covariates (e.g., calendar features: `Hour_sin`, `DayOfWeek_sin`). This means the decoder does not fully exploit TFT's design intent. Future improvement: pass time-varying known-future features (hour, day-of-week, holiday) into the decoder VSN to unlock the full TFT advantage.

---

### 3.9 PatchTST ✅✅ Strong

**Specific advantages:**
- **Patch-based tokenization** (patch_len=16, stride=8) converts the 96-step lookback into ~11 non-overlapping local segments, reducing sequence length fed to the Transformer encoder
- **Channel-independent** (CI) encoding processes each of the 30 input features independently before combining — reduces parameter count while maintaining representation quality
- **RevIN** instance normalization directly counteracts non-stationarity at inference time by normalizing per sample

> **Role:** Expected to perform strongly given its explicit handling of both non-stationarity (RevIN) and computational efficiency (patch tokenization) — two of the main challenges identified in this dataset.

---

### 3.10 Decoder-Only & Encoder-Decoder Transformer ✅ Good

**Decoder-Only:**
- Causal (masked) self-attention ensures no future information leakage
- Appropriate for autoregressive-style reasoning, though the implementation uses a direct 48-step head (non-autoregressive), limiting the advantage of the causal mask
- Slightly simpler than full Encoder-Decoder; may be competitive if the encoder–decoder cross-attention in the full model does not converge well

**Encoder-Decoder:**
- Standard Transformer architecture (Vaswani et al., 2017)
- Cross-attention between encoded context and decoder query is theoretically ideal for multi-step sequence-to-sequence forecasting
- More parameters than encoder-only variants → requires more data or stronger regularization to avoid overfitting on ~19,000 training samples

---

### 3.11 iTransformer (ICLR 2024 Spotlight) ✅✅ Strong

**Why it fits well:**
- **Inverted Dimension Processing:** Treats each feature/variate as an individual token and applies self-attention across variates rather than time steps.
- **Capitalizes on 30 Multivariate Features:** Captures rich inter-series dependencies between weather, calendar, and lag features that standard time-attention models struggle to extract.
- **Robust Temporal Embedding:** Maps each feature's entire 96-step lookback sequence via a linear projection, capturing full temporal trajectory before variate interaction.

> **Role:** Represents the cutting-edge SOTA in multivariate attention forecasting. Expected to be a top performer alongside PatchTST and Autoformer.

---

### 3.12 TimesNet (ICLR 2023) ✅✅ Strong

**Why it fits well:**
- **1D-to-2D Periodic Transformation:** Uses FFT to detect dominant periods (such as daily s=48) and reshapes 1D sequences into 2D variation spaces.
- **2D Inception Convolutions:** Captures intra-period (hourly changes within a day) and inter-period (day-to-day patterns) variations simultaneously.
- **Fills the CNN Paradigm:** Represents the convolutional deep learning paradigm in the benchmark comparison.

> **Role:** Best convolutional / 2D-variation architecture for data with strong multi-periodic seasonality.

---

### 3.13 NLinear (AAAI 2023) ⚠️ Limited (Benchmark Role)

**Why it fits:**
- **Normalization Against Drift:** Subtracts the last observed timestep value $X[-1]$ before linear projection and adds it back after: $\hat{Y} = \text{Linear}(X - X[-1]) + X[-1]$.
- **Directly Addresses Non-Stationarity:** Solves DLinear's main weakness on non-stationary series where mean load drifts across time chunks.
- **Univariate Baseline:** Tests the limits of pure normalized linear mapping without exogenous features.

> **Role:** Direct comparison with DLinear to isolate the benefit of simple instance normalization under distribution drift.

---

## 4. Recommended Usage in Research

### Expected Performance Ranking (estimated)
```
Best  ← iTransformer / Autoformer / PatchTST / TFT / TimesNet
       Vanilla Transformer / Informer / Encoder-Decoder
       LSTM / Decoder-Only / XGBoost / LightGBM
       NLinear / DLinear
Worst ← SARIMA
```

> Note: Actual ranking depends on HPO results. Tree models (XGBoost, LightGBM) can outperform Transformers on tabular-style features when training data is limited.

### Reporting Guidelines

| Model | Reporting Note |
|---|---|
| **SARIMA** | State clearly: univariate, statistical lower-bound benchmark |
| **DLinear / NLinear** | State clearly: univariate, no covariates used (by design per original paper); compare NLinear vs DLinear for normalization impact |
| **XGBoost / LightGBM** | Report as ML baseline (multivariate tabular, Direct Multi-Output) |
| **LSTM** | Report as recurrent DL baseline (ablation: attention vs. recurrence) |
| **TimesNet** | Report as temporal convolutional / 2D-variation baseline (ablation: convolution vs. attention) |
| **iTransformer / PatchTST / TFT / Autoformer** | Report as modern SOTA Transformer family benchmarks |
| **All Models** | Report HPO-tuned hyperparameters per model from `best_params.json` files |

---

## 5. Conclusion

The dataset is **well-suited for deep learning sequence models** that can leverage its 30 engineered features, strong daily seasonality, and multivariate covariate structure. The architecturally strongest fits are:

- **iTransformer** — inverts attention to variate dimension, fully leveraging the 30 multivariate features
- **Autoformer** — explicit decomposition matches the dataset's non-stationarity + 48-step daily cycle
- **PatchTST** — patch tokenization + RevIN directly addresses computational efficiency and non-stationarity
- **TimesNet** — 2D Inception conv across FFT-extracted daily periods (s=48)

