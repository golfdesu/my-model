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

# --- Phase 1: Custom & Vanilla Transformers ---
run_model "01/20" "00_tfm_custom_pytorch.py"         "Custom Transformer"
run_model "02/20" "01_tfm_enc_pytorch.py"            "Transformer Encoder"
run_model "03/20" "02_tfm_dec_pytorch.py"            "Transformer Decoder"
run_model "04/20" "03_tfm_encdec_pytorch.py"         "Transformer Enc-Dec"

# --- Phase 2: Advanced Transformers ---
run_model "05/20" "04_tfm_ifm_pytorch.py"            "Informer"
run_model "06/20" "05_tfm_afm_pytorch.py"            "Autoformer"
run_model "07/20" "06_tfm_ptst_pytorch.py"           "PatchTST"
run_model "08/20" "07_tfm_itfm_pytorch.py"           "iTransformer"
run_model "09/20" "08_tfm_timesnet_pytorch.py"       "TimesNet"

# --- Phase 3: Recurrent & Linear Models ---
run_model "10/20" "09_lstm_baseline_pytorch.py"      "LSTM Baseline"
run_model "11/20" "10_gru_baseline_pytorch.py"       "GRU Baseline"
run_model "12/20" "11_dlinear_baseline_pytorch.py"   "DLinear"
run_model "13/20" "12_nlinear_baseline_pytorch.py"   "NLinear"

# --- Phase 4: State Space Models (SSM) ---
run_model "14/20" "13_smamba_baseline_pytorch.py"     "S-Mamba"
run_model "15/20" "14_powermamba_baseline_pytorch.py" "PowerMamba"
run_model "16/20" "15_timemachine_baseline_pytorch.py" "TimeMachine"
run_model "17/20" "16_s4d_baseline_pytorch.py"       "S4D Baseline"

# --- Phase 5: Tree-Based & Statistical Models ---
run_model "18/20" "17_xgboost_baseline.py"           "XGBoost Baseline"
run_model "19/20" "18_lightgbm_baseline.py"          "LightGBM Baseline"
run_model "20/20" "19_sarima_baseline.py"            "SARIMA Baseline"

echo -e "\n=============================================================================="
echo "ALL 20 BENCHMARKS COMPLETED!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
exit 0
