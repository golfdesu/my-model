# Model Architecture Pitfalls & Bug Catalog

## 1. State Space Models (Mamba / S-Mamba / PowerMamba / TimeMachine)

### Pitfall 1.1: The 3-Layer Overparameterization Trap
- **Empirical Proof**: In S-Mamba HPO on Erawan H100 (26 trials), `num_layers = 3` was sampled 14 times.
  - Every single `num_layers = 3` trial took **16 to 28.5 minutes** (compared to 1.5–5 min for layer 1).
  - Every single `num_layers = 3` trial had **worse validation loss** (0.00338–0.00395) than Trial 3 with `num_layers = 2` (0.003263).
- **Literature Proof**: S-Mamba paper (Wang et al., 2024, arXiv:2403.11144, Section 4) explicitly uses **only 1 bidirectional Mamba layer** (`e_layers=1` or `2`).
- **Rule**: In Optuna search space for Mamba models, **strictly restrict `num_layers` to `[1, 2]`**. Never allow 3 layers for sequence length $L = 96$.

### Pitfall 1.2: Pure PyTorch Recurrent Scan vs Custom CUDA
- **The Issue**: Tri Dao's original Mamba paper relies on a custom C++/CUDA kernel keeping state $h$ inside on-chip SRAM. In portable Pure PyTorch, it falls back to a Python `for t in range(seq_len):` loop.
- **Optimization**:
  1. Pre-vectorize $Bx = (\Delta \odot B) \odot x$ outside the loop in a single GPU operation (cuts 96 kernel dispatches to 1).
  2. Pre-allocate buffer `y = torch.empty(...)` instead of `y_list.append()` + `torch.stack()`.

---

## 2. Linear Models (DLinear / NLinear)
- **Speed**: Fastest architectures in the benchmark (1–2 seconds per trial).
- **Paper Rule**: Fit only 1 linear layer without non-linear activations for trend/seasonal or last-value normalization.

---

## 3. Tree-based Models (XGBoost & LightGBM)
- **Multi-output Setup**: Train 48 independent models (one per horizon step) per seed. Total = 240 models for 5 seeds.
- **LightGBM Bagging**: Must set `subsample_freq=1` when `subsample < 1.0`, otherwise bagging is silently ignored by LightGBM.
- **Histogram Bins**: `max_bin=128` halves training time with negligible accuracy change.

---

## 4. PyTorch Namespace Shadowing (`F` vs `torch.nn.functional`)
- **Pitfall**: In models with multidimensional tensors, code unpacking `B, L, F = x.shape` shadows the top-level import `import torch.nn.functional as F`.
- **Symptom**: Calling `F.softmax(...)`, `F.relu(...)`, or `F.interpolate(...)` later in the method raises `AttributeError: 'int' object has no attribute 'softmax'`.
- **Rule**: Never use `F` as a dimension variable name. Always use `D`, `C`, `K`, or `num_feats`.

---

## 5. Erawan HPC Triton JIT Traps (`torch.bmm` outer products)
- **Pitfall**: Using `torch.bmm` where one tensor has a unit dimension (e.g., `(B, 1, K)` and `(B, K, D)`) triggers PyTorch 2.1+'s Triton JIT compiler to generate specialized outer-product kernels.
- **Symptom**: Crashes with `CalledProcessError: gcc ... Python.h: No such file or directory` on Rocky Linux HPC nodes without root packages.
- **Rule**: Replace outer-product `bmm` with vectorized elementwise broadcast multiply and reduction:
  ```python
  (tensor_a * tensor_b.unsqueeze(1)).sum(dim=-1)
  ```

---

## 6. Architecture Class Name Parity & Aliases
- **Pitfall**: Defining `class ModelName(nn.Module):` but referencing `model = ModelNameModel(...)` in benchmark `run_seed` causes `NameError: name 'ModelNameModel' is not defined`.
- **Rule**: Always verify instantiation matches class name exactly, and provide explicit aliases (e.g., `NHiTSModel = NHiTS`) when backward compatibility is helpful.

---

## 7. Architecture Incompatibility on Intermittent EV Charging Load

Based on the 32-model 10-seed production benchmark on Caltech ACN load ($L=96, H=48$):

### Pitfall 7.1: Channel Independence (CI) & RevIN on Exogenous-Driven Intermittent Load
- **Models**: `06_tfm_ptst` (PatchTST).
- **Pitfall**:
  1. *Channel Independence*: Isolating the target series prevents cross-attention with calendar (`Hour_sin/cos`, `DayOfWeek`) and weather (`temp`, `rhum`) features. EV load cannot be forecasted reliably without human behavioral context.
  2. *RevIN Distortion*: Normalizing by lookback mean and std breaks when lookback mean is near zero (night) but future mean is peak (daytime). Lookback vs future mean differs by up to 22.58 kW, trapping denormalized predictions near zero ($R^2=0.5531$, Peak MAE=18.74 kW).
- **Rule**: Never rely on pure channel-independent architectures without cross-variate attention for human-driven EV charging load.

### Pitfall 7.2: Linear Decomposition & Last-Value Normalization on Pulse Spikes
- **Models**: `11_dlinear`, `12_nlinear`, `25_tide`.
- **Pitfall**:
  1. *DLinear Moving Average*: Kernel smoothing strips sharp burst charging spikes, and 1-layer linear mapping fails to capture non-linear charging curves ($R^2=0.4325$, Horizon Deg +171.7%).
  2. *NLinear Last-Value Offset*: Subtracting and adding $X_{-1}$ forces day forecasts to start from night zero-baseline, or night forecasts to start from daytime peak, causing catastrophic peak error (Peak MAE=21.40 kW, worst in benchmark).
  3. *TiDE Dense MLP*: Linear decoding deteriorates by +172.7% over 48 steps without temporal attention.
- **Rule**: Avoid linear models that rely on stationarity or moving average smoothing on stochastic pulse loads.

### Pitfall 7.3: Phase Correlation & Basis Expansion Ringing on Discontinuous Pulses
- **Models**: `05_tfm_afm` (Autoformer), `26_nbeats` (N-BEATS), `08_tfm_timesnet` (TimesNet).
- **Pitfall**:
  1. *Auto-Correlation (FFT)*: Assumes repetitive identical waveform phases; smooths stochastic charging pulses into sinusoidal waves, yielding 22.89% negative predictions.
  2. *Fourier & Polynomial Basis*: Fourier harmonic expansion on steep discontinuous step loads triggers Gibbs phenomenon (ringing artifacts and large over/undershoots, $R^2=0.5478$).
  3. *2D Periodicity Folding*: 2D Inception convolutions on folded 1D series introduce spatial smearing artifacts across days.
- **Rule**: Prefer point-wise and cross-variate attention architectures over frequency-domain phase matching on non-smooth load data.

### Pitfall 7.4: Overparameterization & Multi-Stream Overhead on Short Lookback
- **Models**: `31_scinet` (18.37M params, Rank 24), `15_timemachine` (428s/seed, Rank 18).
- **Pitfall**: Excessive parameter scaling (18.37M params) or quadruple Mamba state-space scanning yields no accuracy benefit over compact architectures (e.g. Model 00: 180k params, 38s/seed, Rank 13).
- **Rule**: Cap parameter budget and avoid redundant multi-directional scans for sequence lengths $L \le 96$.\n