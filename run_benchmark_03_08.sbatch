#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: Benchmark Models 03, 05, 06, 07, 08
# (Model 04 TFT is excluded as it is a probabilistic model)
# Cluster: Erawan HPC (NSTDA / CMU)
# Partition: gpu-h100 (NVIDIA H100 80GB HBM3)
# ==============================================================================

#SBATCH --job-name=bench_03_08
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

# Create logs directory if it does not exist
mkdir -p logs

# 1. Environment Activation
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
else
    echo "[Warning] .venv not found in current or parent directory. Using default python."
fi

# 2. Hardware and Environment Inspection
echo "=============================================================================="
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Node Allocated   : ${SLURMD_NODENAME}"
echo "Start Time       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python Executable: $(which python)"
echo "PyTorch Version  : $(python -c 'import torch; print(torch.__version__)')"
echo "GPU Model        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=============================================================================="

# 3. Sequential Execution (Guarantees 100% Dedicated GPU per Model)

echo -e "\n>>> [1/5] Starting 03_tfm_afm_pytorch.py (Autoformer)..."
START_03=$(date +%s)
python -u 03_tfm_afm_pytorch.py
END_03=$(date +%s)
DIFF_03=$((END_03 - START_03))
echo ">>> [1/5] Completed 03_tfm_afm_pytorch.py (Elapsed: ${DIFF_03}s)"

echo -e "\n>>> [2/5] Starting 05_tfm_ptft_pytorch.py (PatchTST)..."
START_05=$(date +%s)
python -u 05_tfm_ptft_pytorch.py
END_05=$(date +%s)
DIFF_05=$((END_05 - START_05))
echo ">>> [2/5] Completed 05_tfm_ptft_pytorch.py (Elapsed: ${DIFF_05}s)"

echo -e "\n>>> [3/5] Starting 06_tfm_dec_pytorch.py (Vanilla Decoder)..."
START_06=$(date +%s)
python -u 06_tfm_dec_pytorch.py
END_06=$(date +%s)
DIFF_06=$((END_06 - START_06))
echo ">>> [3/5] Completed 06_tfm_dec_pytorch.py (Elapsed: ${DIFF_06}s)"

echo -e "\n>>> [4/5] Starting 07_tfm_encdec_pytorch.py (Encoder-Decoder)..."
START_07=$(date +%s)
python -u 07_tfm_encdec_pytorch.py
END_07=$(date +%s)
DIFF_07=$((END_07 - START_07))
echo ">>> [4/5] Completed 07_tfm_encdec_pytorch.py (Elapsed: ${DIFF_07}s)"

echo -e "\n>>> [5/5] Starting 08_lstm_baseline_pytorch.py (LSTM Baseline)..."
START_08=$(date +%s)
python -u 08_lstm_baseline_pytorch.py
END_08=$(date +%s)
DIFF_08=$((END_08 - START_08))
echo ">>> [5/5] Completed 08_lstm_baseline_pytorch.py (Elapsed: ${DIFF_08}s)"

echo -e "\n=============================================================================="
echo "BENCHMARK SUMMARY (Total Execution Times):"
echo "  03 Autoformer      : ${DIFF_03} seconds"
echo "  05 PatchTST        : ${DIFF_05} seconds"
echo "  06 Vanilla Decoder : ${DIFF_06} seconds"
echo "  07 Encoder-Decoder : ${DIFF_07} seconds"
echo "  08 LSTM Baseline   : ${DIFF_08} seconds"
echo "End Time             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
echo "[INFO] All 5 models finished successfully. Releasing GPU node now."

exit 0
