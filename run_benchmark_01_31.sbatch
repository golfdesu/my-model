#!/bin/bash
# ==============================================================================
# SLURM Job Script: Master Multi-Seed Benchmark (Models 01 to 31) on NVIDIA H100
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3)
# Execution Mode : Pure Sequential (One-by-one) | Walltime: 24:00:00
# Multi-Seed     : 10 Deterministic Seeds ([42, 123, 456, 789, ..., 9999])
# ==============================================================================
#SBATCH --job-name=bench_seq_01_31_h100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodelist=compute4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

mkdir -p logs
PROGRESS_LOG="logs/benchmark_progress_summary.txt"
COMPLETED_LOG="logs/benchmark_completed.txt"
FAILED_LOG="logs/benchmark_failed.txt"

rm -f "${PROGRESS_LOG}" "${COMPLETED_LOG}" "${FAILED_LOG}"

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Execution Mode      : Pure Sequential (One-by-one)"
echo "Target Walltime     : 24:00:00"
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
elif [ -f "$HOME/project/my-model/.venv/bin/activate" ]; then
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

# 4. Sequential Execution Function
run_model() {
    local script_name=$1
    local tag=$2
    local model_name=$3
    local start_t=$(date +%s)

    echo ""
    echo "=============================================================================="
    echo ">>> [STARTED] [${tag}] ${model_name} (${script_name}) at $(date '+%Y-%m-%d %H:%M:%S')"
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
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] [${tag}] ${model_name} (${elapsed_min}m)" >> "${PROGRESS_LOG}"
        echo "${tag}: ${model_name} (${script_name}) - SUCCESS" >> "${COMPLETED_LOG}"
    else
        echo ">>> [FAILED]   [${tag}] ${model_name} FAILED with exit code ${exit_code} after ${elapsed_min}m (${diff_t}s)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAILED ] [${tag}] ${model_name} (Exit ${exit_code}, ${elapsed_min}m)" >> "${PROGRESS_LOG}"
        echo "${tag}: ${model_name} (${script_name}) - FAILED (Exit ${exit_code})" >> "${FAILED_LOG}"
    fi
    echo "=============================================================================="
    echo ""
}

TOTAL_START=$(date +%s)

# 5. Catalog of Models 01 to 31
MODELS=(
    "01_tfm_enc_pytorch.py:M01:01 Vanilla Transformer Encoder"
    "02_tfm_dec_pytorch.py:M02:02 Vanilla Transformer Decoder"
    "03_tfm_encdec_pytorch.py:M03:03 Seq2Seq Transformer"
    "04_tfm_ifm_pytorch.py:M04:04 Informer"
    "05_tfm_afm_pytorch.py:M05:05 Autoformer"
    "06_tfm_ptst_pytorch.py:M06:06 PatchTST"
    "07_tfm_itfm_pytorch.py:M07:07 iTransformer"
    "08_tfm_timesnet_pytorch.py:M08:08 TimesNet"
    "09_lstm_baseline_pytorch.py:M09:09 LSTM Baseline"
    "10_gru_baseline_pytorch.py:M10:10 GRU Baseline"
    "11_dlinear_baseline_pytorch.py:M11:11 DLinear"
    "12_nlinear_baseline_pytorch.py:M12:12 NLinear"
    "13_smamba_baseline_pytorch.py:M13:13 S-Mamba"
    "14_powermamba_baseline_pytorch.py:M14:14 PowerMamba"
    "15_timemachine_baseline_pytorch.py:M15:15 TimeMachine"
    "16_s4d_baseline_pytorch.py:M16:16 S4D Baseline"
    "17_xgboost_baseline.py:M17:17 XGBoost Direct"
    "18_lightgbm_baseline.py:M18:18 LightGBM Direct"
    "19_sarima_baseline.py:M19:19 SARIMA Baseline"
    "20_tfm_mft_pytorch.py:M20:20 Multi-Factor Transformer"
    "21_cnn_lstm_tfm_pytorch.py:M21:21 CNN-LSTM-Transformer"
    "22_tfm_fedformer_pytorch.py:M22:22 FEDformer"
    "23_tcn_baseline_pytorch.py:M23:23 TCN Baseline"
    "24_nhits_baseline_pytorch.py:M24:24 N-HiTS Baseline"
    "25_tide_baseline_pytorch.py:M25:25 TiDE Baseline"
    "26_nbeats_baseline_pytorch.py:M26:26 N-BEATS Baseline"
    "27_moderntcn_baseline_pytorch.py:M27:27 ModernTCN"
    "28_crossformer_baseline_pytorch.py:M28:28 Crossformer"
    "29_segrnn_baseline_pytorch.py:M29:29 SegRNN"
    "30_nstransformer_baseline_pytorch.py:M30:30 Non-stationary Transformer"
    "31_scinet_baseline_pytorch.py:M31:31 SCINet"
)

TOTAL_MODELS=${#MODELS[@]}
CURRENT_IDX=1

echo "=============================================================================="
echo "Starting Sequential Benchmark of ${TOTAL_MODELS} Models on NVIDIA H100..."
echo "=============================================================================="

# 6. Sequential Execution Loop
for entry in "${MODELS[@]}"; do
    IFS=":" read -r script_name model_tag model_name <<< "${entry}"
    
    echo ""
    echo "##############################################################################"
    echo ">>> PROGRESS: Running Model [${CURRENT_IDX}/${TOTAL_MODELS}] — [${model_tag}] ${model_name}"
    echo "##############################################################################"
    
    if [ ! -f "${script_name}" ]; then
        echo "❌ Script file '${script_name}' not found! Skipping..."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NOT FOUND] [${model_tag}] ${script_name}" >> "${FAILED_LOG}"
    else
        run_model "${script_name}" "${model_tag}" "${model_name}"
    fi
    
    CURRENT_IDX=$((CURRENT_IDX + 1))
done

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_DIFF / 60))
TOTAL_HR=$(echo "scale=2; ${TOTAL_DIFF} / 3600" | bc 2>/dev/null || echo "$((TOTAL_DIFF / 3600))")

echo ""
echo "=============================================================================="
echo "🎉 ALL 31 MODELS SEQUENTIAL RUN COMPLETE"
echo "Total Elapsed Time: ${TOTAL_DIFF}s (~${TOTAL_MIN} min / ${TOTAL_HR} hours)"
echo "------------------------------------------------------------------------------"
echo "Completed Models Summary:"
if [ -f "${COMPLETED_LOG}" ]; then
    cat "${COMPLETED_LOG}"
fi
echo "------------------------------------------------------------------------------"
if [ -f "${FAILED_LOG}" ] && [ -s "${FAILED_LOG}" ]; then
    echo "⚠️ Status: COMPLETED WITH FAILURES"
    echo "Failed Models:"
    cat "${FAILED_LOG}"
else
    echo "✅ Status: ALL 31 BASELINES COMPLETED SUCCESSFULLY!"
fi
echo "=============================================================================="

# 7. Automated Publication Aggregation
echo ""
echo "=============================================================================="
echo "Running Automated Benchmark Aggregator..."
echo "=============================================================================="
if [ -f "tools/aggregate_benchmark.py" ]; then
    python tools/aggregate_benchmark.py
    echo "✅ Aggregation complete. Summary tables and figures generated in docs/ and outputs/."
else
    echo "⚠️ Warning: tools/aggregate_benchmark.py not found. Skipping automatic aggregation."
fi
echo "=============================================================================="
echo "SLURM Job Finished at $(date '+%Y-%m-%d %H:%M:%S')"
