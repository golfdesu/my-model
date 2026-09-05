#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: Benchmark Remaining 7 Models
# Models: 13, 14, 15, 18, 19, 20, 12 (Excluding 04 TFT)
# Cluster: Erawan HPC (NSTDA / CMU)
# Partition: gpu-h100 (NVIDIA H100 80GB HBM3)
# ==============================================================================

#SBATCH --job-name=bench_remaining
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

mkdir -p logs

# 1. Environment Activation
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
else
    echo "[Warning] .venv not found. Using default python."
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

# Helper function: runs model and records elapsed wall-clock time
run_model() {
    local step_num=$1
    local script_name=$2
    local model_name=$3

    echo -e "\n------------------------------------------------------------------------------"
    echo ">>> [${step_num}] STARTING: ${model_name} (${script_name})..."
    local start_t=$(date +%s)
    python -u "${script_name}"
    local exit_code=$?
    local end_t=$(date +%s)
    local diff_t=$((end_t - start_t))
    if [ ${exit_code} -eq 0 ]; then
        echo ">>> [${step_num}] COMPLETED: ${model_name} in ${diff_t} seconds."
    else
        echo ">>> [${step_num}] ERROR: ${model_name} exited with code ${exit_code} after ${diff_t}s."
    fi
}

# 3. Sequential Pipeline Execution

# --- Phase 1: High-Throughput Deep Learning & Linear Baselines ---
run_model "1/7" "13_tfm_itfm_pytorch.py"           "iTransformer"
run_model "2/7" "14_tfm_timesnet_pytorch.py"       "TimesNet"
run_model "3/7" "15_nlinear_baseline_pytorch.py"   "NLinear"

# --- Phase 2: State Space Models (SSM) ---
run_model "4/7" "18_powermamba_baseline_pytorch.py" "PowerMamba"
run_model "5/7" "19_timemachine_baseline_pytorch.py" "TimeMachine"
run_model "6/7" "20_s4d_baseline_pytorch.py"       "S4D Baseline"

# --- Phase 3: Statistical Baseline (CPU) ---
run_model "7/7" "12_sarima_baseline.py"            "SARIMA Baseline"

echo -e "\n=============================================================================="
echo "ALL REMAINING BENCHMARKS COMPLETED!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
exit 0
