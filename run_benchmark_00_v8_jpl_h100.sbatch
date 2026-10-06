#!/bin/bash
# ==============================================================================
# SLURM Job Script: Benchmark Model 00 v8 on NASA JPL V3
# Architecture   : Model 00 v8 (Adaptive Gated Residual Readout + Composite Loss)
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3) on 'gpu-h100'
# Dataset Target : NASA JPL V3 (data_cleaned/acn_jpl_ready_v3.csv)
# Execution Mode : 10-Seed Multi-Seed Benchmark ([42, 123, 456, ..., 9999])
# Output Target  : Root directory (.)
# ==============================================================================
#SBATCH --job-name=bench_00_v8_jpl_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

mkdir -p logs

export DATASET_TARGET="acn_jpl_ready_v3"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Model        : Model 00 Custom Transformer v8"
echo "Dataset Target      : NASA JPL V3 (${DATASET_TARGET})"
echo "Output Directory    : $(pwd) (Root)"
echo "Unified Log File    : logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "=============================================================================="

# 1. Environment & CUDA Setup
if command -v module &> /dev/null; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../environment/.venv/bin/activate" ]; then
    source ../environment/.venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -d "$HOME/project/my-model/.venv" ]; then
    source "$HOME/project/my-model/.venv/bin/activate"
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

echo "Python Executable   : $(which python)"

# 2. Strict Hardware Assertion: NVIDIA H100 Verification
echo "------------------------------------------------------------------------------"
echo "Checking GPU Hardware..."
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
echo "Detected Primary GPU: ${GPU_NAME}"

if [[ "${GPU_NAME}" != *"H100"* ]]; then
    echo "❌ FATAL ERROR: Detected GPU '${GPU_NAME}' is NOT an NVIDIA H100!"
    echo "The job configuration strictly enforces execution on an H100 node (compute4)."
    echo "Aborting job to maintain benchmark fidelity."
    exit 1
fi
echo "✅ Confirmed: Running strictly on NVIDIA H100 80GB HBM3."
echo "------------------------------------------------------------------------------"

# 3. High-Performance Hopper & H100 Settings
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS=16

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'Device 0: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

TOTAL_START=$(date +%s)

# 4. Execute Model 00 v8
echo ""
echo "=============================================================================="
echo ">>> [STARTED] Model 00 v8 Benchmark on NASA JPL V3 at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

python -u 00_tfm_custom_pytorch.py
EXIT_CODE=$?

TOTAL_END=$(date +%s)
DIFF_T=$((TOTAL_END - TOTAL_START))
ELAPSED_MIN=$((DIFF_T / 60))

echo ""
echo "=============================================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> [FINISHED] Model 00 v8 COMPLETED in ${ELAPSED_MIN}m (${DIFF_T}s)"
    echo ">>> Output artifacts generated directly in root (.):"
    ls -lh 00_tfm_custom_pytorch* 2>/dev/null || true
else
    echo ">>> [FAILED] Model 00 v8 FAILED with exit code ${EXIT_CODE} after ${ELAPSED_MIN}m (${DIFF_T}s)"
fi
echo "=============================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
