#!/bin/bash
# ==============================================================================
# SLURM Job Script: 2x2 Factorial Ablation Study on ACN JPL V3
# Investigates: EEO vs Orthogonal Regularization for Model 00 (Trial 19 Params)
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3) on 'gpu-h100'
# Execution Mode : 10-Seed Multi-Seed Evaluation for V2, V3, and V4 (V1 pre-computed)
# Output Target  : Root directory (.)
# ==============================================================================
#SBATCH --job-name=ablation_00_jpl_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00

mkdir -p logs

export DATASET_TARGET="acn_jpl_ready_v3"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Task         : Model 00 2x2 Factorial Ablation Study (V2, V3, V4)"
echo "Dataset Target      : ACN JPL V3 (${DATASET_TARGET})"
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

# 4. Execute Ablation Engine for V2, V3, V4 on ACN JPL V3 (saving directly to root .)
echo ""
echo "=============================================================================="
echo ">>> [STARTED] Running JPL Ablation Engine (V2, V3, V4 across 10 Seeds each)..."
echo "=============================================================================="

python -u 00_ablation_eeo_ortho_pytorch.py --dataset jpl --variants v2_no_eeo v3_no_ortho v4_backbone --output-dir .

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_DIFF / 60))

echo ""
echo "=============================================================================="
echo ">>> JPL ABLATION STUDY COMPLETED IN ${TOTAL_MIN}m (${TOTAL_DIFF}s)"
echo "Finished at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
