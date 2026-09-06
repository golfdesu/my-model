# AGENTS.md — Multi-Seed Benchmark Suite Operational Manual & Ground Rules

Welcome to the **EV Charging Load Forecasting & Benchmark Suite** workspace (`golfdesu/my-model`).
This repository is the dedicated Multi-Seed Production Benchmark and Publication Engine of our research program, paired with the hyperparameter tuning repository and guided by the master academic thesis vault.

---

## 1. The 3-Pillar Research Ecosystem

This workspace operates as part of a tightly coupled **3-Pillar Scientific Ecosystem**:

```
+-------------------------------------------------------------------------------+
|                      PILLAR 1: OBSIDIAN THESIS VAULT                          |
|             Location: C:\Users\chaya\Documents\Obsidian\Thesis                |
|  - Literature Digest (paper_digest.md)       - BibTeX Citations (.bib)        |
|  - Research Gaps (research_gaps.md)          - EDA & Features (dataset_*.md)  |
|  - Proposed Architectures & Orthogonal Reg.  - Theoretical Foundations        |
+---------------------------------------+---------------------------------------+
                                        | (Theory, Invariants & Architecture)
                                        v
+---------------------------------------+---------------------------------------+
|                  PILLAR 2: HYPERPARAMETER OPTIMIZATION                        |
|        Location: C:\Users\chaya\Documents\Program\Practice\hyperparameter_tuning |
|                           Repo: golfdesu/HPO                                  |
|  - 32 HPO Scripts (00_hpo_*.py to 31_hpo_*.py)                                |
|  - Optuna TPE Search (50 trials, 30 epochs, patience 10)                      |
|  - Parsimony Selection Rule (Hastie et al., 2009; epsilon = 0.01)             |
|  - Master Production Configs: configs/selected_production_params.json         |
+---------------------------------------+---------------------------------------+
                                        | (Selected Parsimonious Hyperparameters)
                                        v
+---------------------------------------+---------------------------------------+
|                 PILLAR 3: MULTI-SEED BENCHMARK SUITE                          |
|               Location: C:\Users\chaya\Documents\Program\Practice\model        |
|                         Repo: golfdesu/my-model                               |
|  - 32 Benchmark Models (00_*.py to 31_*.py)                                   |
|  - 10-Seed Robustness Evaluation ([42, 123, ..., 9999])                       |
|  - 9 Evaluation Metrics + VRAM + Runtime + Multi-step Loss Tracking           |
|  - Publication Aggregator: tools/aggregate_benchmark.py                       |
|  - LaTeX, CSV, Markdown Tables & Publication Figures                          |
+---------------------------------------+---------------------------------------+
```

### 1.1 External Academic Knowledge Base (Obsidian Thesis Vault)
**Path**: `C:\Users\chaya\Documents\Obsidian\Thesis`  
Whenever agents need theoretical context, literature review details, mathematical proofs, or citation keys, consult the following canonical files:
- `paper_digest.md`: Deep literature digests for all time-series architectures, empirical claims, and benchmarks.
- `research_gaps.md` & `progress_summary_and_research_gaps.md`: Identified research gaps and open scientific questions in EV aggregate charging load forecasting.
- `proposed_architectures.md` & `transformer_research_ideas.md`: Mathematical formulation of Model 00 (Proposed Custom Transformer with Attention Orthogonal Regularization: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$).
- `dataset_extraction_report.md`: Exploratory data analysis, feature descriptions, station dynamics, and lookback/horizon characteristics on the Caltech ACN dataset.
- `thesis_references.bib`: Master BibTeX citations with standardized citation keys.

### 1.2 Cross-Workspace Pipeline & Transitions
- **Tuning (`../hyperparameter_tuning/`)**: Conducts 50-trial Optuna HPO and parsimony selection (`configs/selected_production_params.json`).
- **Benchmarking (Here)**: Implements production architectures with selected parameters, running 10-seed deterministic evaluation (`SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]`). Raw results output to `<model_id>_results.json`.
- **Aggregation & Publishing (Here)**: Run `python tools/aggregate_benchmark.py` to compile all results into LaTeX tables (`docs/benchmark_summary.tex`), Markdown tables (`docs/benchmark_summary.md`), CSV files (`outputs/benchmark_summary.csv`), and publication-quality plots in `plots/`.

---

## 2. Model Catalog (All 32 Models: 00 to 31)

| ID | Benchmark File Name (`model/`) | HPO File Name (`hyperparameter_tuning/`) | Architecture Key Mechanism | Reference Paper / Provenance |
|:---|:---|:---|:---|:---|
| **00** | `00_tfm_custom_pytorch.py` | `00_hpo_tfm_custom_pytorch.py` | **Custom Transformer** w/ Attention Orthogonal Regularization | Proposed Architecture (Thesis) |
| **01** | `01_tfm_enc_pytorch.py` | `01_hpo_tfm_pytorch.py` | **Vanilla Transformer Encoder** (MHA + Positional Encoding) | Vaswani et al. (NeurIPS 2017) |
| **02** | `02_tfm_dec_pytorch.py` | `02_hpo_dec_pytorch.py` | **Vanilla Transformer Decoder** (Causal Autoregressive Masking) | Vaswani et al. (NeurIPS 2017) |
| **03** | `03_tfm_encdec_pytorch.py` | `03_hpo_encdec_pytorch.py` | **Full Seq2Seq Transformer** (Cross-Attention Encoder-Decoder) | Vaswani et al. (NeurIPS 2017) |
| **04** | `04_tfm_ifm_pytorch.py` | `04_hpo_ifm_pytorch.py` | **Informer** (ProbSparse Self-Attention + Distillation) | Zhou et al. (AAAI 2021) |
| **05** | `05_tfm_afm_pytorch.py` | `05_hpo_afm_pytorch.py` | **Autoformer** (Series Decomposition + AutoCorrelation) | Wu et al. (NeurIPS 2021) |
| **06** | `06_tfm_ptst_pytorch.py` | `06_hpo_ptst_pytorch.py` | **PatchTST** (Patching + Channel Independence + RevIN) | Nie et al. (ICLR 2023) |
| **07** | `07_tfm_itfm_pytorch.py` | `07_hpo_itfm_pytorch.py` | **iTransformer** (Inverted Tokens + Variate-Attention) | Liu et al. (ICLR 2024) |
| **08** | `08_tfm_timesnet_pytorch.py` | `08_hpo_timesnet_pytorch.py` | **TimesNet** (2D-FFT Top-k Periods + 2D Inception Block) | Wu et al. (ICLR 2023) |
| **09** | `09_lstm_baseline_pytorch.py` | `09_hpo_lstm_pytorch.py` | **LSTM Baseline** (Multi-layer LSTM + Input Jitter) | Hochreiter & Schmidhuber (1997) |
| **10** | `10_gru_baseline_pytorch.py` | `10_hpo_gru_pytorch.py` | **GRU Baseline** (Gated Recurrent Unit + Multi-Feature Proj) | Cho et al. (EMNLP 2014) |
| **11** | `11_dlinear_baseline_pytorch.py` | `11_hpo_dlinear_pytorch.py` | **DLinear** (Moving Average Decomp + 1-Layer Linear) | Zeng et al. (AAAI 2023) |
| **12** | `12_nlinear_baseline_pytorch.py` | `12_nlinear_baseline_pytorch.py` | **NLinear** (Last-value Normalization: $\hat{Y} = W(X - X_{-1}) + X_{-1}$) | Zeng et al. (AAAI 2023) |
| **13** | `13_smamba_baseline_pytorch.py` | `13_hpo_smamba_pytorch.py` | **S-Mamba** (Bidirectional Selective State Space Model) | Wang et al. (2024); Gu & Dao (2023) |
| **14** | `14_powermamba_baseline_pytorch.py` | `14_powermamba_baseline_pytorch.py` | **PowerMamba** (Series Decomp + Dual-Path Selective SSM) | Menati et al. (2024) |
| **15** | `15_timemachine_baseline_pytorch.py` | `15_hpo_timemachine_pytorch.py` | **TimeMachine** (Quadruple Cross-Time/Channel Mamba) | Ahamed & Cheng (2024) |
| **16** | `16_s4d_baseline_pytorch.py` | `16_hpo_s4d_pytorch.py` | **S4D Baseline** (Diagonal State Space Kernel + Cauchy Conv) | Gu et al. (ICLR 2022) |
| **17** | `17_xgboost_baseline.py` | `17_hpo_xgboost.py` | **XGBoost** (Direct Multi-step Histogram GBDT) | Chen & Guestrin (KDD 2016) |
| **18** | `18_lightgbm_baseline.py` | `18_hpo_lightgbm.py` | **LightGBM** (Direct Multi-step GBDT with Subsample Bagging) | Ke et al. (NeurIPS 2017) |
| **19** | `19_sarima_baseline.py` | `19_hpo_sarima.py` | **SARIMA** (Statistical Seasonal ARIMA $(p,d,q)(P,D,Q)_{48}$) | Box & Jenkins (1970) |
| **20** | `20_tfm_mft_pytorch.py` | `20_hpo_mft_pytorch.py` | **Multi-Factor Transformer** (Factor Embedding + MHA) | Multi-Factor Benchmark |
| **21** | `21_cnn_lstm_tfm_pytorch.py` | `21_hpo_cnn_lstm_tfm_pytorch.py` | **CNN-LSTM-Transformer** (Conv1D + BiLSTM + Attention) | Hybrid DL Benchmark |
| **22** | `22_tfm_fedformer_pytorch.py` | `22_hpo_fedformer_pytorch.py` | **FEDformer** (Frequency Enhanced Decomp + Fourier Cross-Attn) | Zhou et al. (ICML 2022) |
| **23** | `23_tcn_baseline_pytorch.py` | `23_hpo_tcn_pytorch.py` | **TCN Baseline** (Dilated Causal Convolutions + ResBlocks) | Bai et al. (2018) |
| **24** | `24_nhits_baseline_pytorch.py` | `24_hpo_nhits_pytorch.py` | **N-HiTS** (Multi-rate Hierarchical Interpolation + Residuals) | Challu et al. (AAAI 2023) |
| **25** | `25_tide_baseline_pytorch.py` | `25_hpo_tide_pytorch.py` | **TiDE** (MLP ResBlock Enc-Dec + Covariates + Linear Skip) | Das et al., Google (TMLR 2023) |
| **26** | `26_nbeats_baseline_pytorch.py` | `26_hpo_nbeats_pytorch.py` | **N-BEATS** (Doubly Residual Stacks: Trend, Seasonality, Generic) | Oreshkin et al. (ICLR 2020) |
| **27** | `27_moderntcn_baseline_pytorch.py` | `27_hpo_moderntcn_pytorch.py` | **ModernTCN** (Large-Kernel Depthwise Conv + ConvFFN) | Dong et al. (ICLR 2024) |
| **28** | `28_crossformer_baseline_pytorch.py` | `28_hpo_crossformer_pytorch.py` | **Crossformer** (DSW Embedding + Two-Stage Cross-Time/Dim Attn) | Zhang & Yan (ICLR 2023) |
| **29** | `29_segrnn_baseline_pytorch.py` | `29_hpo_segrnn_pytorch.py` | **SegRNN** (Segment-wise Recurrent GRU + Direct Step Decode) | Lin et al. (ICLR 2024) |
| **30** | `30_nstransformer_baseline_pytorch.py` | `30_hpo_nstransformer_pytorch.py` | **Non-stationary Transformer** (Series Stationarization + $\tau,\Delta$ Attn) | Liu et al. (NeurIPS 2022) |
| **31** | `31_scinet_baseline_pytorch.py` | `31_hpo_scinet_pytorch.py` | **SCINet** (Recursive Downsample-Convolve-Interact SCI-Blocks Tree) | Liu et al. (NeurIPS 2022) |

---

## 3. Scientific Ground Truth & Paper Invariants (NON-NEGOTIABLE)

Every agent must strictly maintain complete compliance with these scientific invariants. Violating any of these constitutes scientific leakage and will invalidate benchmark results:

1. **Chronological Splitting Protocol**:
   - **Train**: First 60% of chronological data.
   - **Validation**: Next 20% of chronological data.
   - **Test**: Final 20% of chronological data.
   - **Rule**: NEVER shuffle sequences or apply random K-Fold cross-validation across time.
2. **Target & Preprocessing Invariants**:
   - **Target**: `kWhDelivered` (EV aggregate station load).
   - **Excluded Noise Features**: `prcp`, `tempDiff_48`, and `cldc` must remain dropped.
   - **MinMaxScaler**: Fit **ONLY on the Training split**. Transform validation/test splits using train statistics.
3. **Geometry**:
   - **Lookback ($L$)**: 96 steps (48 hours at 30-min intervals).
   - **Forecast Horizon ($H$)**: 48 steps (24 hours at 30-min intervals).
4. **Reproducibility & Multi-Seed Protocol**:
   - Multi-Seed Benchmark: `SEEDS = [42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999]`.
   - Every seed must run with full determinism (`torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all`).

---

## 4. Hardware & Execution Constraints (Erawan HPC vs Local)

The project targets both local development (Windows / CPU / CUDA) and the **Erawan HPC cluster (`compute4` node with NVIDIA H100 80GB HBM3)**:

### ⚠️ Critical HPC Traps & Constraints
- **Trap 1: NEVER use `torch.compile` on Erawan**:
  - The Rocky Linux base image on `compute4` lacks `python3-devel` headers. Calling `torch.compile` crashes with `fatal error: Python.h: No such file or directory`. Always use PyTorch CUDA eager mode.
- **Trap 2: LightGBM must run in CPU mode**:
  - Prebuilt Linux wheels from PyPI lack the CUDA Tree Learner. Enforce `device='cpu'`, `n_jobs=-1`, and `max_bin=128`.
- **Trap 3: XGBoost runs with native CUDA GPU**:
  - XGBoost on Linux has full CUDA support. Enforce `tree_method='hist'`, `'device': 'cuda'`.
- **H100 Speed Optimization Checklist**:
  ```python
  if device.type == 'cuda':
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      torch.backends.cudnn.benchmark = True
  ```
- **No Artificial Memory Caps**: Do not set `torch.cuda.set_per_process_memory_fraction(0.5)` on HPC nodes; utilize the full 80GB HBM3.

---

## 5. Automated Publication Aggregator (`tools/aggregate_benchmark.py`)

Run this tool to automatically synthesize all completed multi-seed benchmark results:

```bash
python tools/aggregate_benchmark.py
```

### Generated Artifacts:
- **`outputs/benchmark_summary.csv`**: Machine-readable metrics across all models (Mean +/- Std for 9 metrics, Params, Peak VRAM, Runtime).
- **`docs/benchmark_summary.md`**: Markdown summary table with best and runner-up bold/underline formatting.
- **`docs/benchmark_summary.tex`**: Publication-ready Booktabs LaTeX table for inclusion in thesis manuscripts.
- **`plots/benchmark_mae_ranking.png`**: Ranking bar chart of all models with error bars.
- **`plots/benchmark_horizon_mae.png`**: Multi-step horizon error propagation curves ($H=1 \dots 48$).
- **`plots/benchmark_pareto_efficiency.png`**: Multi-objective Pareto frontier (MAE vs Parameters & Runtime).

---

## 6. Institutional Knowledge & Specialized Skills

### 6.1 Institutional Memory (`.wikiskill/wiki/`)
This project implements **Google Research: WikiSkill** (*arXiv:2608.27454*) under `.wikiskill/wiki/`:
- [`INDEX.md`](.wikiskill/wiki/INDEX.md): Knowledge catalog
- [`workspace_and_thesis_linkage.md`](.wikiskill/wiki/workspace_and_thesis_linkage.md): 3-Pillar research ecosystem architecture
- [`paper_invariants.md`](.wikiskill/wiki/paper_invariants.md): Scientific ground truth & schema parity
- [`erawan_hpc_playbook.md`](.wikiskill/wiki/erawan_hpc_playbook.md): HPC & H100 execution rules & Triton JIT avoidance
- [`model_pitfalls.md`](.wikiskill/wiki/model_pitfalls.md): Architecture traps, Mamba restrictions, and variable shadowing rules
- [`optimization_guide.md`](.wikiskill/wiki/optimization_guide.md): High-throughput compute guide
- [`production_hyperparameters.md`](.wikiskill/wiki/production_hyperparameters.md): Validated HPO configurations & parsimony registry

### 6.2 Project Specialized Skills (`.agents/skills/`)
- **`code-params-auditor`**: Audits forecasting code files and validates hyperparameter JSON files (`configs/audit_state.json`).
- **`hpo-parsimony-selector`**: Audits, selects, and structures production-ready hyperparameters using the $\epsilon$-tolerance rule (`configs/selected_production_params.json`).
- **`paper-alignment-auditor`**: Audits model implementations against canonical research papers (`configs/paper_alignment_state.json`).

### 6.3 Automated Preflight & Gating Scripts (`.wikiskill/skills/`)
- **`preflight_check.py`**: Scans model scripts before launch to catch prohibited patterns.
- **`validate_gating.py`**: Enforces rollback if any proposed modification degrades performance or breaks compliance.

### 6.4 SKILL.state Mandatory Protocol for Paper & Code Audits
Whenever an agent is tasked with **auditing, checking, or re-checking models against papers or code** (e.g. "ตรวจโค้ดกับเปเปอร์", "re-check paper alignment", "audit model X"):
1. **ANTI-WASTE DIRECTIVE**: Do NOT perform ad-hoc exhaustive reading of raw `.py` files or dumping hundreds of lines of code/PDFs into context.
2. **MANDATORY SKILL STATE ACTIVATION**:
   - First, inspect `configs/paper_alignment_state.json` (and `configs/audit_state.json`) to read existing compliance status and known deviations with zero token waste.
   - Run the programmatic audit engine:
     `python ../hyperparameter_tuning/.agents/skills/paper-alignment-auditor/scripts/paper_audit_engine.py`
3. **PERSISTENT STATE RECORDING**:
   - Whenever an issue is discovered or remediated, immediately update `configs/paper_alignment_state.json` (`status`: `"ALIGNED"` | `"DEVIATED"`, with exact mechanism details).
   - This ensures institutional memory is preserved so subsequent agents know model integrity instantly without re-auditing.

---

## 7. Agent Operational Directives & Safety Rules

1. **Explicit Permission Required**: Under `RULE[user_global]`, do NOT proactively modify or refactor code files without explaining the proposed changes and receiving explicit user authorization.
2. **Preserve Comments & Docstrings**: Keep all existing docstrings, mathematical formulas, and provenance notes intact.
3. **Audit Against Wiki Before Launch**: Cross-reference against `erawan_hpc_playbook.md`, `model_pitfalls.md`, and `paper_invariants.md` before submitting SLURM jobs or executing long runs.
