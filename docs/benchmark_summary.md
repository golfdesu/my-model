# Comprehensive Benchmark Results (10-Seed Average)

> Dataset: `acn_caltech_ready2.csv` (Lookback $L=96$, Horizon $H=48$)
> Sorted by: `mae_mean` (Lowest is Best)

| Rank | Model Architecture | Family | MAE (kWh) ↓ | RMSE (kWh) ↓ | WAPE (%) ↓ | Peak MAE ↓ | R² ↑ | Params | Time (s) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `28_crossformer_baseline_pytorch` | Transformer (Crossformer) | **5.0004 ± 0.1176** 🏆 | **7.4610 ± 0.1038** | **58.72% ± 1.38%** | 12.9167 | **0.7094** | 1,645,360 | 36.5s |
| 2 | `03_tfm_encdec_pytorch` | Transformer | 5.0528 ± 0.1323 | 7.8750 ± 0.1611 | 59.34% ± 1.55% | 13.3395 | 0.6762 | 178,497 | 130.0s |
| 3 | `07_tfm_itfm_pytorch` | Transformer | 5.0937 ± 0.1477 | 7.5109 ± 0.1545 | 59.82% ± 1.73% | **11.1202** | 0.7055 | N/A | 60.7s |
| 4 | `23_tcn_baseline_pytorch` | CNN / 2D Temporal | 5.1188 ± 0.1077 | 7.4927 ± 0.1225 | 60.11% ± 1.27% | 13.3974 | 0.7069 | 161,392 | 20.3s |
| 5 | `27_moderntcn_baseline_pytorch` | CNN (ModernTCN) | 5.1498 ± 0.0340 | 8.0072 ± 0.0735 | 60.48% ± 0.40% | 15.7185 | 0.6654 | 767,112 | 37.2s |
| 6 | `24_nhits_baseline_pytorch` | Basis Expansion | 5.2916 ± 0.0827 | 7.7747 ± 0.1053 | 62.14% ± 0.97% | 12.9767 | 0.6845 | 2,308,530 | 27.5s |
| 7 | `18_lightgbm_baseline` | Tree-based GBDT | 5.3865 ± 0.0059 | 8.1150 ± 0.0070 | 63.26% ± 0.07% | 14.7259 | 0.6563 | 48,000 | 196.4s |
| 8 | `04_tfm_ifm_pytorch` | Transformer | 5.3972 ± 0.1241 | 8.0262 ± 0.1322 | 63.38% ± 1.46% | 13.2090 | 0.6637 | 120,592 | 208.3s |
| 9 | `17_xgboost_baseline` | Tree-based GBDT | 5.4623 ± 0.0099 | 8.1352 ± 0.0087 | 64.15% ± 0.12% | 14.7549 | 0.6546 | 48,000 | 49.5s |
| 10 | `10_gru_baseline_pytorch` | Recurrent (RNN) | 5.4808 ± 0.1177 | 7.8744 ± 0.1160 | 64.37% ± 1.38% | 13.3875 | 0.6763 | 54,640 | 23.5s |
| 11 | `13_smamba_baseline_pytorch` | State Space Model | 5.4959 ± 0.2483 | 8.7047 ± 0.4660 | 64.54% ± 2.92% | 17.2869 | 0.6034 | 333,288 | 358.2s |
| 12 | `22_tfm_fedformer_pytorch` | Transformer | 5.5627 ± 0.0700 | 7.9165 ± 0.0809 | 65.33% ± 0.82% | 12.6828 | 0.6729 | 54,530 | 30.5s |
| 13 | `00_tfm_custom_pytorch` | Custom Proposed | 5.5811 ± 0.1324 | 7.8832 ± 0.1215 | 65.54% ± 1.55% | 12.5785 | 0.6756 | 180,336 | 38.2s |
| 14 | `02_tfm_dec_pytorch` | Transformer | 5.5937 ± 0.1361 | 8.0554 ± 0.1082 | 65.69% ± 1.60% | 12.9435 | 0.6613 | 654,960 | 53.9s |
| 15 | `01_tfm_enc_pytorch` | Transformer | 5.6313 ± 0.0893 | 8.0440 ± 0.1092 | 66.13% ± 1.05% | 14.0934 | 0.6622 | 29,136 | 25.1s |
| 16 | `20_tfm_mft_pytorch` | Transformer (MFT) | 5.6435 ± 0.2722 | 8.2366 ± 0.3430 | 66.28% ± 3.20% | 13.0577 | 0.6453 | 231,024 | 32.4s |
| 17 | `09_lstm_baseline_pytorch` | Recurrent (RNN) | 5.6596 ± 0.0804 | 8.1586 ± 0.1706 | 66.47% ± 0.94% | 13.7514 | 0.6525 | 62,960 | 12.9s |
| 18 | `15_timemachine_baseline_pytorch` | State Space Model | 5.6652 ± 0.0910 | 8.1436 ± 0.0790 | 66.53% ± 1.07% | 14.6931 | 0.6539 | 38,448 | 428.0s |
| 19 | `21_cnn_lstm_tfm_pytorch` | Hybrid Architecture | 5.7343 ± 0.1865 | 8.4143 ± 0.2966 | 67.34% ± 2.19% | 13.8747 | 0.6301 | 685,104 | 17.3s |
| 20 | `16_s4d_baseline_pytorch` | State Space Model | 5.7690 ± 0.1409 | 8.3191 ± 0.1640 | 67.75% ± 1.65% | 13.4052 | 0.6387 | 27,984 | 19.4s |
| 21 | `14_powermamba_baseline_pytorch` | State Space Model | 5.7845 ± 0.1200 | 8.2071 ± 0.1632 | 67.93% ± 1.41% | 14.2565 | 0.6483 | 45,360 | 107.4s |
| 22 | `30_nstransformer_baseline_pytorch` | Transformer (NS-Tfm) | 5.8956 ± 0.1437 | 8.6152 ± 0.1732 | 69.24% ± 1.69% | 15.2207 | 0.6125 | 113,873 | 24.2s |
| 23 | `05_tfm_afm_pytorch` | Transformer | 5.9289 ± 0.1330 | 8.6172 ± 0.1860 | 69.63% ± 1.56% | 14.2734 | 0.6123 | 926,978 | 48.1s |
| 24 | `31_scinet_baseline_pytorch` | Convolutional (SCINet) | 6.0479 ± 0.3763 | 8.5736 ± 0.4783 | 71.03% ± 4.42% | 14.5122 | 0.6152 | 18,376,753 | 44.1s |
| 25 | `26_nbeats_baseline_pytorch` | Basis Expansion | 6.0587 ± 0.0586 | 9.3085 ± 0.0304 | 71.15% ± 0.69% | 17.5791 | 0.5478 | 1,036,864 | 26.6s |
| 26 | `29_segrnn_baseline_pytorch` | Recurrent (SegRNN) | 6.0661 ± 0.1210 | 9.4696 ± 0.0945 | 71.24% ± 1.42% | 16.9112 | 0.5320 | 800,012 | 38.8s |
| 27 | `08_tfm_timesnet_pytorch` | CNN / 2D Temporal | 6.0995 ± 0.1824 | 8.8674 ± 0.2681 | 71.63% ± 2.14% | 16.1503 | 0.5893 | N/A | 127.9s |
| 28 | `12_nlinear_baseline_pytorch` | Linear & Decomp | 6.2466 ± 0.0036 | 10.4387 ± 0.0011 | 73.36% ± 0.04% | 21.3976 | 0.4313 | 4,656 | 30.3s |
| 29 | `11_dlinear_baseline_pytorch` | Linear & Decomp | 6.9656 ± 0.1071 | 10.4281 ± 0.0434 | 81.80% ± 1.26% | 19.6542 | 0.4325 | 9,312 | 24.6s |
| 30 | `25_tide_baseline_pytorch` | Linear / Dense MLP | 6.9962 ± 0.0631 | 10.4550 ± 0.0326 | 82.16% ± 0.74% | 19.4354 | 0.4295 | 1,833,396 | 16.3s |
| 31 | `06_tfm_ptst_pytorch` | Transformer | 7.2949 ± 0.1759 | 10.7911 ± 0.2136 | 85.67% ± 2.07% | 14.3890 | 0.3921 | 542,056 | 75.9s |
| 32 | `19_sarima_baseline` | Statistical Baseline | 10.1119 ± 0.0000 | 13.8129 ± 0.0000 | 108.27% ± 0.00% | 20.3742 | 0.1407 | 4 | 11593.7s |