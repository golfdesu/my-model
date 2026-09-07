#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: 10-Seed Benchmark for Model 06 (PatchTST)
# Target Architecture: NVIDIA H100 80GB HBM3 (Erawan HPC - Node compute4 strictly)
# Model Evaluated: 06_tfm_ptst_pytorch.py (PatchTST: Patching + Channel Independence + RevIN)
# Evaluation Protocol: 10 Seeds ([42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999])
# Estimated Total Runtime: ~1.0 - 1.5 hours on H100
# ==============================================================================
#SBATCH --job-name=bench_06_ptst
#SBATCH --partition=gpu-h100              # Strictly H100 partition only
#SBATCH --nodelist=compute4               # Pin specifically to Erawan compute4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                      # 1x NVIDIA H100 80GB GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs outputs

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "=============================================================================="

# 1. Environment & Module Setup
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
elif [ -d "$HOME/project/HPO/.venv" ]; then
    source "$HOME/project/HPO/.venv/bin/activate"
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

# Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

# 2. Execution of Model 06
echo ""
echo "=============================================================================="
echo ">>> STARTING: 06 PatchTST (06_tfm_ptst_pytorch.py) on NVIDIA H100"
echo ">>> Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

START_T=$(date +%s)
python -u 06_tfm_ptst_pytorch.py
EXIT_CODE=$?
END_T=$(date +%s)
DIFF_T=$((END_T - START_T))

echo ""
echo "=============================================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Status: COMPLETED SUCCESSFULLY in ${DIFF_T}s ($((DIFF_T / 60)) min)"
else
    echo "Status: FAILED with exit code ${EXIT_CODE} after ${DIFF_T}s"
fi
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

exit ${EXIT_CODE}
