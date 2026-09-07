#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: 10-Seed Benchmark for Models 04, 05, 06
# Target Architecture: NVIDIA H100 80GB HBM3 (Erawan HPC - Node compute4 strictly)
# Models Evaluated:
#   - 04_tfm_ifm_pytorch.py  : Informer (ProbSparse Attention + Distillation)
#   - 05_tfm_afm_pytorch.py  : Autoformer (AutoCorrelation + Decomposition)
#   - 06_tfm_ptst_pytorch.py : PatchTST (Patching + Channel Independence)
# Evaluation Protocol: 10 Seeds ([42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999])
# Estimated Total Runtime: ~1.5 - 2.5 hours on H100
# ==============================================================================
#SBATCH --job-name=bench_04_05_06
#SBATCH --partition=gpu-h100              # Strictly H100 partition only
#SBATCH --nodelist=compute4               # Pin specifically to Erawan compute4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                      # 1x NVIDIA H100 80GB GPU
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
else
    echo "Warning: No virtualenv found, using default python: $(which python)"
fi

# Diagnostics
echo "Python Executable   : $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
nvidia-smi

# 2. Sequential Execution Function
run_benchmark() {
    local step_num=$1
    local script_name=$2
    local model_name=$3

    echo ""
    echo "=============================================================================="
    echo ">>> [${step_num}] STARTING: ${model_name} (${script_name})"
    echo ">>> Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    local start_t=$(date +%s)
    python -u "${script_name}"
    local exit_code=$?
    local end_t=$(date +%s)
    local diff_t=$((end_t - start_t))

    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [${step_num}] [SUCCESS]: ${model_name} completed in ${diff_t}s ($((diff_t / 60)) min)"
    else
        echo ">>> [${step_num}] [ERROR]: ${model_name} FAILED with exit code ${exit_code} after ${diff_t}s"
    fi
    return ${exit_code}
}

TOTAL_START=$(date +%s)
FAILED_MODELS=()

MODELS=(
    "04_tfm_ifm_pytorch.py:04 Informer"
    "05_tfm_afm_pytorch.py:05 Autoformer"
    "06_tfm_ptst_pytorch.py:06 PatchTST"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT_STEP=1

for entry in "${MODELS[@]}"; do
    IFS=":" read -r script_name model_name <<< "${entry}"
    run_benchmark "${CURRENT_STEP}/${TOTAL_MODELS}" "${script_name}" "${model_name}"
    if [ $? -ne 0 ]; then
        FAILED_MODELS+=("${model_name}")
    fi
    CURRENT_STEP=$((CURRENT_STEP + 1))
done

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))

echo ""
echo "=============================================================================="
echo "BENCHMARK SUITE COMPLETE"
echo "Total Elapsed Time: ${TOTAL_DIFF}s ($((TOTAL_DIFF / 60)) min)"
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo "Status: ALL 3 MODELS COMPLETED SUCCESSFULLY"
else
    echo "Status: COMPLETED WITH FAILURES"
    echo "Failed Models: ${FAILED_MODELS[*]}"
fi
echo "=============================================================================="
