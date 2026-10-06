#!/bin/bash
# ==============================================================================
# SLURM Job Script: Benchmark Model 00 (Proposed) Trial 15 on Caltech V3
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3) on 'gpu-h100'
# Dataset Target : ACN Caltech V3 (data_cleaned/acn_caltech_ready_v3.csv)
# Execution Mode : 10-Seed Multi-Seed Benchmark ([42, 123, 456, ..., 9999])
# Parameters     : Trial 15 Parsimonious Robust Parameters (Dropout 0.20, LR 0.00043)
# Output Target  : outputs/acn_caltech_v3/00_tfm_custom_pytorch/ & Root
# ==============================================================================
#SBATCH --job-name=bench_00_trial15_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

mkdir -p logs outputs/acn_caltech_v3/00_tfm_custom_pytorch

export DATASET_TARGET="acn_caltech_ready_v3"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Model        : Model 00 Custom Transformer (Trial 15 Parsimonious)"
echo "Dataset Target      : ACN Caltech V3 (${DATASET_TARGET})"
echo "Output Directory    : outputs/acn_caltech_v3/00_tfm_custom_pytorch/"
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

# 4. Helper Function to Run and Archive Model 00
run_benchmark() {
    local script_name=$1
    local tag=$2
    local model_name=$3
    local out_prefix=$4
    local start_t=$(date +%s)

    echo ""
    echo "=============================================================================="
    echo ">>> [STARTED]  [${tag}] ${model_name} (${script_name}) at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="

    python -u "${script_name}"
    local exit_code=$?
    local end_t=$(date +%s)
    local diff_t=$((end_t - start_t))
    local elapsed_min=$((diff_t / 60))

    echo ""
    echo "=============================================================================="
    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [FINISHED] [${tag}] ${model_name} COMPLETED in ${elapsed_min}m (${diff_t}s)"
        mkdir -p "outputs/acn_caltech_v3/${out_prefix}"
        cp -v "${out_prefix}_results.json" "outputs/acn_caltech_v3/${out_prefix}/" 2>/dev/null || true
        cp -v "${out_prefix}_best.pt" "outputs/acn_caltech_v3/${out_prefix}/" 2>/dev/null || true
        cp -v "${out_prefix}_predictions.npz" "outputs/acn_caltech_v3/${out_prefix}/" 2>/dev/null || true
        echo ">>> [ARCHIVED] Stored in outputs/acn_caltech_v3/${out_prefix}/ and root"
    else
        echo ">>> [FAILED]   [${tag}] ${model_name} FAILED with exit code ${exit_code} after ${elapsed_min}m (${diff_t}s)"
    fi
    echo "=============================================================================="
    echo ""
}

# 5. Execute Model 00 (Proposed) Trial 15
run_benchmark "00_tfm_custom_pytorch.py" "M00_T15" "Proposed Custom Transformer (Trial 15)" "00_tfm_custom_pytorch"

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DIFF / 3600))
TOTAL_MIN=$(((TOTAL_DIFF % 3600) / 60))

echo "=============================================================================="
echo ">>> BENCHMARK COMPLETED FOR MODEL 00 (TRIAL 15) ON CALTECH V3 (NVIDIA H100)"
echo "Total Elapsed Time  : ${TOTAL_HOURS}h ${TOTAL_MIN}m ($TOTAL_DIFF seconds)"
echo "Finished at         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
