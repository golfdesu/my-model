#!/bin/bash
# ==============================================================================
# Slurm Master Batch Script: Benchmark Models 03 to 20 (Excl. 04 TFT)
# Reordered: Fast GPU Deep Learning models first, CPU/Statistical (10, 11, 12) last.
# Cluster: Erawan HPC (NSTDA / CMU)
# Partition: gpu-h100 (NVIDIA H100 80GB HBM3)
# ==============================================================================

#SBATCH --job-name=bench_all
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

# 3. Execution Pipeline

# --- Phase 1: High-Throughput GPU Deep Learning & Linear Models ---
run_model "01/17" "03_tfm_afm_pytorch.py"            "Autoformer"
run_model "02/17" "05_tfm_ptft_pytorch.py"           "PatchTST"
run_model "03/17" "06_tfm_dec_pytorch.py"            "Vanilla Decoder"
run_model "04/17" "07_tfm_encdec_pytorch.py"         "Encoder-Decoder"
run_model "05/17" "08_lstm_baseline_pytorch.py"      "LSTM Baseline"
run_model "06/17" "09_dlinear_baseline_pytorch.py"   "DLinear"
run_model "07/17" "13_tfm_itfm_pytorch.py"           "iTransformer"
run_model "08/17" "14_tfm_timesnet_pytorch.py"       "TimesNet"
run_model "09/17" "15_nlinear_baseline_pytorch.py"   "NLinear"
run_model "10/17" "16_gru_baseline_pytorch.py"       "GRU Baseline"
run_model "11/17" "17_smamba_baseline_pytorch.py"     "S-Mamba"
run_model "12/17" "18_powermamba_baseline_pytorch.py" "PowerMamba"
run_model "13/17" "19_timemachine_baseline_pytorch.py" "TimeMachine"
run_model "14/17" "20_s4d_baseline_pytorch.py"       "S4D Baseline"

# --- Phase 2: Tree-Based & Statistical Models (Ran at the end) ---
run_model "15/17" "10_xgboost_baseline.py"           "XGBoost Baseline"
run_model "16/17" "11_lightgbm_baseline.py"          "LightGBM Baseline"
run_model "17/17" "12_sarima_baseline.py"            "SARIMA Baseline"

echo -e "\n=============================================================================="
echo "ALL 17 BENCHMARKS COMPLETED!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
exit 0
