#!/bin/bash
# ==============================================================================
# SLURM Job Script: Model 00 (Custom Transformer w/ Attention Orthogonal Reg)
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3) on 'gpu-h100'
# Execution Mode : 10-Seed Multi-Seed Benchmark ([42, 123, 456, ..., 9999])
# ==============================================================================
#SBATCH --job-name=bench_00_custom_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

mkdir -p logs

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Model        : Model 00 Custom Transformer (10-Seed Robustness Evaluation)"
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

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'Device 0: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

# 4. Run Model 00 10-Seed Benchmark
START_TIME=$(date +%s)
echo "=============================================================================="
echo ">>> [STARTED] Model 00 Custom Transformer at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

python -u 00_tfm_custom_pytorch.py
EXIT_CODE=$?

END_TIME=$(date +%s)
DIFF_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((DIFF_TIME / 60))

echo "=============================================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> [SUCCESS] Model 00 Custom Transformer COMPLETED in ${ELAPSED_MIN}m (${DIFF_TIME}s)"
    echo "Outputs saved to: 00_tfm_custom_pytorch_results.json, 00_tfm_custom_pytorch_best.pt, 00_tfm_custom_pytorch_predictions.npz"
else
    echo ">>> [FAILED]  Model 00 Custom Transformer FAILED with exit code ${EXIT_CODE} after ${ELAPSED_MIN}m (${DIFF_TIME}s)"
fi
echo "=============================================================================="
