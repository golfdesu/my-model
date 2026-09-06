# Comprehensive Benchmark Results (10-Seed Average)

> Dataset: `acn_caltech_ready2.csv` (Lookback $L=96$, Horizon $H=48$)
> Sorted by: `mae_mean` (Lowest is Best)

| Rank | Model Architecture | Family | MAE (kWh) ↓ | RMSE (kWh) ↓ | WAPE (%) ↓ | Peak MAE ↓ | R² ↑ | Params | Time (s) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `07_tfm_itfm_pytorch` | Transformer | **5.0937 ± 0.1477** 🏆 | **7.5109 ± 0.1545** | **59.82% ± 1.73%** | **11.1202** | **0.7055** | N/A | 60.7s |
| 2 | `18_lightgbm_baseline` | Tree-based GBDT | 5.3865 ± 0.0059 | 8.1150 ± 0.0070 | 63.26% ± 0.07% | 14.7259 | 0.6563 | 48,000 | 196.4s |
| 3 | `04_tfm_ifm_pytorch` | Transformer | 5.3972 ± 0.1241 | 8.0262 ± 0.1322 | 63.38% ± 1.46% | 13.2090 | 0.6637 | 120,592 | 208.3s |
| 4 | `17_xgboost_baseline` | Tree-based GBDT | 5.4623 ± 0.0099 | 8.1352 ± 0.0087 | 64.15% ± 0.12% | 14.7549 | 0.6546 | 48,000 | 49.5s |
| 5 | `01_tfm_enc_pytorch` | Transformer | 5.5415 ± 0.0866 | 7.9135 ± 0.1106 | 65.08% ± 1.02% | 12.8813 | 0.6731 | 180,336 | 20.0s |
| 6 | `00_tfm_custom_pytorch` | Custom Proposed | 5.5811 ± 0.1324 | 7.8832 ± 0.1215 | 65.54% ± 1.55% | 12.5785 | 0.6756 | 180,336 | 38.2s |
| 7 | `10_gru_baseline_pytorch` | Recurrent (RNN) | 5.5895 ± 0.1014 | 7.9918 ± 0.1039 | 65.64% ± 1.19% | 13.2555 | 0.6666 | 33,264 | 22.5s |
| 8 | `09_lstm_baseline_pytorch` | Recurrent (RNN) | 5.6338 ± 0.0815 | 8.0907 ± 0.1488 | 66.16% ± 0.96% | 13.1166 | 0.6583 | 37,488 | 19.4s |
| 9 | `15_timemachine_baseline_pytorch` | State Space Model | 5.6652 ± 0.0910 | 8.1436 ± 0.0790 | 66.53% ± 1.07% | 14.6931 | 0.6539 | 38,448 | 428.0s |
| 10 | `02_tfm_dec_pytorch` | Transformer | 5.7191 ± 0.0914 | 8.0587 ± 0.0850 | 67.16% ± 1.07% | 12.9092 | 0.6610 | 192,624 | 43.2s |
| 11 | `16_s4d_baseline_pytorch` | State Space Model | 5.7690 ± 0.1409 | 8.3191 ± 0.1640 | 67.75% ± 1.65% | 13.4052 | 0.6387 | 27,984 | 19.4s |
| 12 | `14_powermamba_baseline_pytorch` | State Space Model | 5.7845 ± 0.1200 | 8.2071 ± 0.1632 | 67.93% ± 1.41% | 14.2565 | 0.6483 | 45,360 | 107.4s |
| 13 | `13_smamba_baseline_pytorch` | State Space Model | 5.8187 ± 0.1023 | 8.2148 ± 0.0850 | 68.33% ± 1.20% | 15.0795 | 0.6478 | 362,480 | 1625.3s |
| 14 | `05_tfm_afm_pytorch` | Transformer | 5.9289 ± 0.1330 | 8.6172 ± 0.1860 | 69.63% ± 1.56% | 14.2734 | 0.6123 | 926,978 | 48.1s |
| 15 | `03_tfm_encdec_pytorch` | Transformer | 5.9328 ± 0.1112 | 8.2210 ± 0.1065 | 69.67% ± 1.31% | 13.1369 | 0.6472 | 1,282,800 | 75.8s |
| 16 | `08_tfm_timesnet_pytorch` | CNN / 2D Temporal | 6.0995 ± 0.1824 | 8.8674 ± 0.2681 | 71.63% ± 2.14% | 16.1503 | 0.5893 | N/A | 127.9s |
| 17 | `12_nlinear_baseline_pytorch` | Linear & Decomp | 6.2466 ± 0.0036 | 10.4387 ± 0.0011 | 73.36% ± 0.04% | 21.3976 | 0.4313 | 4,656 | 21.8s |
| 18 | `11_dlinear_baseline_pytorch` | Linear & Decomp | 6.9832 ± 0.0540 | 10.4371 ± 0.0227 | 82.01% ± 0.63% | 19.6748 | 0.4315 | 9,312 | 5.1s |
| 19 | `06_tfm_ptst_pytorch` | Transformer | 7.2949 ± 0.1759 | 10.7911 ± 0.2136 | 85.67% ± 2.07% | 14.3890 | 0.3921 | 542,056 | 75.9s |
| 20 | `19_sarima_baseline` | Statistical Baseline | nan ± nan | nan ± nan | nan% ± nan% | N/A | N/A | 4 | N/A |