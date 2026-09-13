#!/bin/bash
# ==============================================================================
# SLURM Job Script: Model 00 v4 — Two Dataset Benchmarks (No Weather)
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3), partition gpu-h100
# Execution: Sequential 10-seed benchmarks; stop on the first failure
# ============================================================================
#SBATCH --job-name=bench_00_v4_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:30:00

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

START_TIME=$(date +%s)
CURRENT_SCRIPT="initialization"

on_error() {
    local exit_code=$?
    local elapsed=$(( $(date +%s) - START_TIME ))
    echo "=============================================================================="
    echo ">>> [FAILED] ${CURRENT_SCRIPT} failed with exit code ${exit_code} after ${elapsed}s"
    echo "=============================================================================="
    exit "${exit_code}"
}
trap on_error ERR

run_benchmark() {
    local script="$1"
    local log_name="$2"
    CURRENT_SCRIPT="${script}"

    echo "=============================================================================="
    echo ">>> [STARTED] ${script} at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "    Dataset benchmark ${log_name}"
    echo "=============================================================================="

    python -u "${script}" 2>&1 | tee "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${log_name}.out"

    echo "=============================================================================="
    echo ">>> [SUCCESS] ${script} completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="
}

echo "=============================================================================="
echo "SLURM Job ID        : ${SLURM_JOB_ID:-local}"
echo "Allocated Node      : ${SLURMD_NODENAME:-unknown}"
echo "Partition           : ${SLURM_JOB_PARTITION:-unknown}"
echo "Job Name            : ${SLURM_JOB_NAME:-bench_00_v4_all}"
echo "Working Directory   : $(pwd)"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Target Model        : Model 00 v4 — two sequential benchmarks (no weather)"
echo "=============================================================================="

# Activate the project environment when available.
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "../environment/.venv/bin/activate" ]]; then
    source ../environment/.venv/bin/activate
elif [[ -f "../.venv/bin/activate" ]]; then
    source ../.venv/bin/activate
elif [[ -d "${HOME}/project/my-model/.venv" ]]; then
    source "${HOME}/project/my-model/.venv/bin/activate"
elif [[ -f "${HOME}/.venv/bin/activate" ]]; then
    source "${HOME}/.venv/bin/activate"
else
    echo "Warning: no virtual environment found; using $(command -v python)"
fi

# H100-specific environment and TF32 settings.
if command -v module >/dev/null 2>&1; then
    module purge
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL ERROR: nvidia-smi is unavailable."
    exit 1
fi

nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
echo "Detected Primary GPU: ${GPU_NAME}"
if [[ "${GPU_NAME}" != *"H100"* ]]; then
    echo "FATAL ERROR: detected GPU '${GPU_NAME}' is not an NVIDIA H100."
    exit 1
fi

echo "Confirmed: running on NVIDIA H100."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device 0: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

run_benchmark "00_v4_acn_caltech_no_weather.py" "acn_caltech_no_weather"
run_benchmark "00_v4_acn_jpn_no_weather.py" "acn_jpn_no_weather"

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
echo "=============================================================================="
echo ">>> [SUCCESS] Both Model 00 v4 no-weather benchmarks completed in ${TOTAL_TIME}s"
echo "    Results are archived by each Python script in its dataset-specific output directory."
echo "=============================================================================="
