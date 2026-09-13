# EV Charging Load Forecasting Benchmark Suite

A high-performance multi-seed benchmarking and publication engine for Electric Vehicle (EV) aggregate charging load forecasting, evaluating **32 deep learning, state-space, tree-based, and statistical architectures** across deterministic seeds.

Part of the **3-Pillar Scientific Research Ecosystem**:
1. **Obsidian Thesis Vault**: Theoretical foundations, mathematical proofs, literature digests, and BibTeX citations.
2. **Hyperparameter Optimization (`../hyperparameter_tuning`)**: Optuna TPE search with the 1-SE parsimony rule (*Hastie et al., 2009*).
3. **Multi-Seed Benchmark Suite (This Repository)**: 10-seed robustness evaluation, hardware profiling, and automated publication tables/plots.

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
# Linux / HPC
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Single Model Benchmark (10 Seeds)
```bash
# Run Proposed Custom Transformer (Model 00)
python 00_tfm_custom_pytorch.py

# Or any canonical baseline (e.g., PatchTST, iTransformer, S-Mamba)
python 06_tfm_ptst_pytorch.py
```

### 3. Run on HPC Cluster (SLURM)
```bash
sbatch run_benchmark_00_custom.sbatch
```

### 4. Aggregate Results into Publication Tables & Plots
```bash
python tools/aggregate_benchmark.py
```
Generates:
- `outputs/benchmark_summary.csv` — Comprehensive metric summary table
- `docs/benchmark_summary.md` — Markdown summary table with bold/underline rankings
- `docs/benchmark_summary.tex` — Publication-ready Booktabs LaTeX table
- `plots/benchmark_mae_ranking.png` — Model ranking bar chart with error bars
- `plots/benchmark_horizon_mae.png` — Multi-step error propagation across $H = 1 \dots 48$

---

## 📊 Scientific Protocol & Invariants

All models are strictly evaluated under non-negotiable scientific ground truths to prevent data leakage:
- **Dataset**: Caltech & JPL Adaptive Charging Network (ACN) datasets.
- **Sequence Geometry**: Lookback $L = 96$ steps (48 hours), Forecast Horizon $H = 48$ steps (24 hours) at 30-minute sampling.
- **Chronological Split**: Train (first 60%), Validation (next 20%), Test (final 20%) — no random splitting or temporal leakage.
- **Preprocessing**: Feature scaling fitted strictly on the Training set.
- **10-Seed Robustness**: Evaluated on seeds `[42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]`.

---

## 🏛️ Model Catalog (32 Architectures)

| ID | Model Script | Key Mechanism / Paradigm | Provenance / Canonical Reference |
|:---|:---|:---|:---|
| **00** | `00_tfm_custom_pytorch.py` | **Proposed Custom Transformer** (EEO-Attention + Fused QKV Fast Attention) | **Thesis Contribution** |
| **01** | `01_tfm_enc_pytorch.py` | Vanilla Transformer Encoder | Vaswani et al. (NeurIPS 2017) |
| **02** | `02_tfm_dec_pytorch.py` | Vanilla Transformer Decoder (Causal Autoregressive) | Vaswani et al. (NeurIPS 2017) |
| **03** | `03_tfm_encdec_pytorch.py` | Full Seq2Seq Transformer (Cross-Attention) | Vaswani et al. (NeurIPS 2017) |
| **04** | `04_tfm_ifm_pytorch.py` | Informer (ProbSparse Self-Attention) | Zhou et al. (AAAI 2021) |
| **05** | `05_tfm_afm_pytorch.py` | Autoformer (Series Decomp + AutoCorrelation) | Wu et al. (NeurIPS 2021) |
| **06** | `06_tfm_ptst_pytorch.py` | PatchTST (Patching + Channel Independence) | Nie et al. (ICLR 2023) |
| **07** | `07_tfm_itfm_pytorch.py` | iTransformer (Inverted Tokens + Variate-Attention) | Liu et al. (ICLR 2024) |
| **08** | `08_tfm_timesnet_pytorch.py` | TimesNet (2D-FFT Multi-Period Inception) | Wu et al. (ICLR 2023) |
| **09** | `09_lstm_baseline_pytorch.py` | Stacked LSTM Recurrent Network | Hochreiter & Schmidhuber (1997) |
| **10** | `10_gru_baseline_pytorch.py` | Gated Recurrent Unit (GRU) Network | Cho et al. (EMNLP 2014) |
| **11** | `11_dlinear_baseline_pytorch.py` | DLinear (Moving Average Decomposition + Linear) | Zeng et al. (AAAI 2023) |
| **12** | `12_nlinear_baseline_pytorch.py` | NLinear (Reversible Last-Value Normalization) | Zeng et al. (AAAI 2023) |
| **13** | `13_smamba_baseline_pytorch.py` | S-Mamba (Bidirectional Selective State Space) | Wang et al. (2024); Gu & Dao (2023) |
| **14** | `14_powermamba_baseline_pytorch.py` | PowerMamba (Decomposed Dual-Path SSM) | Menati et al. (2024) |
| **15** | `15_timemachine_baseline_pytorch.py` | TimeMachine (Cross-Time/Channel Mamba) | Ahamed & Cheng (2024) |
| **16** | `16_s4d_baseline_pytorch.py` | S4D (Diagonal Structured State Space Model) | Gu et al. (ICLR 2022) |
| **17** | `17_xgboost_baseline.py` | XGBoost (Histogram Gradient Boosted Trees) | Chen & Guestrin (KDD 2016) |
| **18** | `18_lightgbm_baseline.py` | LightGBM (Gradient Boosted Trees with Bagging) | Ke et al. (NeurIPS 2017) |
| **19** | `19_sarima_baseline.py` | SARIMA (Seasonal Autoregressive Integrated MA) | Box & Jenkins (1970) |
| **20** | `20_tfm_mft_pytorch.py` | Multi-Factor Transformer (Factor Embedding) | Multi-Factor Benchmark |
| **21** | `21_cnn_lstm_tfm_pytorch.py` | CNN-LSTM-Transformer Hybrid | Hybrid Deep Learning |
| **22** | `22_tfm_fedformer_pytorch.py` | FEDformer (Frequency Enhanced Decomp Transformer) | Zhou et al. (ICML 2022) |
| **23** | `23_tcn_baseline_pytorch.py` | Temporal Convolutional Network (Dilated Causal) | Bai et al. (2018) |
| **24** | `24_nhits_baseline_pytorch.py` | N-HiTS (Neural Hierarchical Interpolation) | Challu et al. (AAAI 2023) |
| **25** | `25_tide_baseline_pytorch.py` | TiDE (Time-series Dense Encoder) | Das et al., Google (TMLR 2023) |
| **26** | `26_nbeats_baseline_pytorch.py` | N-BEATS (Neural Basis Expansion Analysis) | Oreshkin et al. (ICLR 2020) |
| **27** | `27_moderntcn_baseline_pytorch.py` | ModernTCN (Large-Kernel Depthwise Conv) | Dong et al. (ICLR 2024) |
| **28** | `28_crossformer_baseline_pytorch.py` | Crossformer (Two-Stage Cross-Time/Dim Attention) | Zhang & Yan (ICLR 2023) |
| **29** | `29_segrnn_baseline_pytorch.py` | SegRNN (Segment-wise Recurrent Neural Network) | Lin et al. (ICLR 2024) |
| **30** | `30_nstransformer_baseline_pytorch.py` | Non-stationary Transformer (Series De-stationarization) | Liu et al. (NeurIPS 2022) |
| **31** | `31_scinet_baseline_pytorch.py` | SCINet (Sample-Convolution and Interaction Network) | Liu et al. (NeurIPS 2022) |

---

## 🤖 AI Agent Guidelines

If you are an AI assistant collaborating on this codebase, refer to [**`AGENTS.md`**](AGENTS.md) for strict operational rules, hardware-specific constraints (e.g., Erawan H100 execution, Triton JIT avoidance, and BFloat16 prohibitions), and reproducibility requirements.
