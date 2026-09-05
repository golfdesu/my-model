#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: Benchmark FP32 vs BF16 for 01 & 02
# Cluster: Erawan HPC (NSTDA / CMU)
# Partition: gpu-h100 (NVIDIA H100 80GB HBM3)
# ==============================================================================

#SBATCH --job-name=bench_01_02
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

echo -e "\n>>> [1/3] Starting 01_tfm_enc_pytorch.py (Vanilla Transformer Encoder)..."
START_01=$(date +%s)
python -u 01_tfm_enc_pytorch.py
END_01=$(date +%s)
DIFF_01=$((END_01 - START_01))
echo ">>> [1/3] Completed 01_tfm_enc_pytorch.py (Elapsed: ${DIFF_01}s)"

echo -e "\n>>> [2/3] Starting 02_tfm_dec_pytorch.py (Vanilla Transformer Decoder)..."
START_02=$(date +%s)
python -u 02_tfm_dec_pytorch.py
END_02=$(date +%s)
DIFF_02=$((END_02 - START_02))
echo ">>> [2/3] Completed 02_tfm_dec_pytorch.py (Elapsed: ${DIFF_02}s)"

echo -e "\n>>> [3/3] Starting 03_tfm_encdec_pytorch.py (Vanilla Transformer Enc-Dec)..."
START_03=$(date +%s)
python -u 03_tfm_encdec_pytorch.py
END_03=$(date +%s)
DIFF_03=$((END_03 - START_03))
echo ">>> [3/3] Completed 03_tfm_encdec_pytorch.py (Elapsed: ${DIFF_03}s)"

echo -e "\n=============================================================================="
echo "BENCHMARK SUMMARY (Total Execution Times):"
echo "  01 Transformer Encoder : ${DIFF_01} seconds"
echo "  02 Transformer Decoder : ${DIFF_02} seconds"
echo "  03 Transformer Enc-Dec : ${DIFF_03} seconds"
echo "End Time              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
echo "[INFO] All 3 Vanilla Transformer models finished. Releasing GPU node now."

# Automatically terminates job and releases GPU back to Slurm queue
exit 0
