#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: Benchmark Models 13, 14, 15
# (13 iTransformer, 14 TimesNet, 15 NLinear)
# Cluster: Erawan HPC (NSTDA / CMU)
# Partition: gpu-h100 (NVIDIA H100 80GB HBM3)
# ==============================================================================

#SBATCH --job-name=bench_13_15
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

# 3. Execution Pipeline
run_model "1/3" "13_tfm_itfm_pytorch.py"         "iTransformer"
run_model "2/3" "14_tfm_timesnet_pytorch.py"     "TimesNet"
run_model "3/3" "15_nlinear_baseline_pytorch.py" "NLinear"

echo -e "\n=============================================================================="
echo "BENCHMARK COMPLETED FOR MODELS 13, 14, 15!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
exit 0
