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

echo -e "\n>>> [1/4] Starting 01_tfm_tfm_pytorch.py (FP32/TF32 Baseline)..."
START_01_FP32=$(date +%s)
python -u 01_tfm_tfm_pytorch.py
END_01_FP32=$(date +%s)
DIFF_01_FP32=$((END_01_FP32 - START_01_FP32))
echo ">>> [1/4] Completed 01_tfm_tfm_pytorch.py (Elapsed: ${DIFF_01_FP32}s)"

echo -e "\n>>> [2/4] Starting 01_tfm_tfm_pytorch_bf16.py (BF16 Mixed Precision)..."
START_01_BF16=$(date +%s)
python -u 01_tfm_tfm_pytorch_bf16.py
END_01_BF16=$(date +%s)
DIFF_01_BF16=$((END_01_BF16 - START_01_BF16))
echo ">>> [2/4] Completed 01_tfm_tfm_pytorch_bf16.py (Elapsed: ${DIFF_01_BF16}s)"

echo -e "\n>>> [3/4] Starting 02_tfm_ifm_pytorch.py (FP32/TF32 Baseline)..."
START_02_FP32=$(date +%s)
python -u 02_tfm_ifm_pytorch.py
END_02_FP32=$(date +%s)
DIFF_02_FP32=$((END_02_FP32 - START_02_FP32))
echo ">>> [3/4] Completed 02_tfm_ifm_pytorch.py (Elapsed: ${DIFF_02_FP32}s)"

echo -e "\n>>> [4/4] Starting 02_tfm_ifm_pytorch_bf16.py (BF16 Mixed Precision)..."
START_02_BF16=$(date +%s)
python -u 02_tfm_ifm_pytorch_bf16.py
END_02_BF16=$(date +%s)
DIFF_02_BF16=$((END_02_BF16 - START_02_BF16))
echo ">>> [4/4] Completed 02_tfm_ifm_pytorch_bf16.py (Elapsed: ${DIFF_02_BF16}s)"

echo -e "\n=============================================================================="
echo "BENCHMARK SUMMARY (Total Execution Times):"
echo "  01 Transformer (FP32) : ${DIFF_01_FP32} seconds"
echo "  01 Transformer (BF16) : ${DIFF_01_BF16} seconds"
echo "  02 Informer    (FP32) : ${DIFF_02_FP32} seconds"
echo "  02 Informer    (BF16) : ${DIFF_02_BF16} seconds"
echo "End Time              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
echo "[INFO] All 4 models finished. Releasing GPU node now."

# Automatically terminates job and releases GPU back to Slurm queue
exit 0
