#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script: Multi-Seed Robustness Benchmark for 20 Models
# Target Architecture: NVIDIA H100 80GB HBM3 (Erawan HPC - Node compute4 strictly)
# Models Evaluated: 01, 02, 03, 09, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
# Evaluation Protocol: 10 Seeds ([42, 123, 456, 789, 1024, 2024, 2025, 2026, 3407, 9999])
# Estimated Total Runtime: ~3.0 - 5.0 hours total on H100
# ==============================================================================
#SBATCH --job-name=bench_20_models
#SBATCH --partition=gpu-h100              # Strictly H100 partition only
#SBATCH --nodelist=compute4               # Pin specifically to Erawan compute4 (H100 node)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                      # 1x NVIDIA H100 80GB GPU
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

# Print Diagnostics
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
    "01_tfm_enc_pytorch.py:01 Vanilla Transformer Encoder"
    "02_tfm_dec_pytorch.py:02 Vanilla Transformer Decoder"
    "03_tfm_encdec_pytorch.py:03 Seq2Seq Transformer"
    "09_lstm_baseline_pytorch.py:09 LSTM Baseline"
    "10_gru_baseline_pytorch.py:10 GRU Baseline"
    "11_dlinear_baseline_pytorch.py:11 DLinear"
    "12_nlinear_baseline_pytorch.py:12 NLinear"
    "13_smamba_baseline_pytorch.py:13 S-Mamba Canonical"
    "20_tfm_mft_pytorch.py:20 Multi-Factor Transformer"
    "21_cnn_lstm_tfm_pytorch.py:21 CNN-LSTM-Transformer"
    "22_tfm_fedformer_pytorch.py:22 FEDformer"
    "23_tcn_baseline_pytorch.py:23 TCN Baseline"
    "24_nhits_baseline_pytorch.py:24 N-HiTS Baseline"
    "25_tide_baseline_pytorch.py:25 TiDE Baseline"
    "26_nbeats_baseline_pytorch.py:26 N-BEATS Baseline"
    "27_moderntcn_baseline_pytorch.py:27 ModernTCN Baseline"
    "28_crossformer_baseline_pytorch.py:28 Crossformer Baseline"
    "29_segrnn_baseline_pytorch.py:29 SegRNN Baseline"
    "30_nstransformer_baseline_pytorch.py:30 Non-stationary Transformer"
    "31_scinet_baseline_pytorch.py:31 SCINet Baseline"
)

TOTAL_COUNT=${#MODELS[@]}
CURRENT_IDX=1

for entry in "${MODELS[@]}"; do
    IFS=":" read -r script_name model_name <<< "$entry"
    run_benchmark "${CURRENT_IDX}/${TOTAL_COUNT}" "$script_name" "$model_name"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        FAILED_MODELS+=("$script_name (Code: $EXIT_CODE)")
    fi
    CURRENT_IDX=$((CURRENT_IDX + 1))
done

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))

echo ""
echo "=============================================================================="
echo "ALL 20 BENCHMARKS FINISHED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total Elapsed Walltime: ${TOTAL_DIFF} seconds ($((TOTAL_DIFF / 60)) minutes)"
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo "Overall Status        : [SUCCESS] All 20 models executed perfectly!"
else
    echo "Overall Status        : [WARNING] Some models failed: ${FAILED_MODELS[*]}"
fi
echo "=============================================================================="
exit 0
