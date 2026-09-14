#!/bin/bash
# ==============================================================================
# SLURM Job Script: Model 00 Inverted Custom Transformer Remaining Benchmarks
# Target Hardware: compute4 node (NVIDIA H100 80GB HBM3) on 'gpu-h100'
# Execution Mode : 10-Seed Multi-Seed Benchmark ([42, 123, 456, ..., 9999])
#
# Workload Order:
#   1. Model 00 on Caltech without Weather (00_tfm_custom_caltech_no_weather.py)
#   2. Model 00 on JPN without Weather     (00_tfm_custom_jpn_no_weather.py)
#   3. Model 00 on JPN with Weather        (00_tfm_custom_jpn.py)
# ==============================================================================
#SBATCH --job-name=bench_00_remaining
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

mkdir -p logs

echo "=============================================================================="
echo "SLURM Job ID        : $SLURM_JOB_ID"
echo "Allocated Node      : $SLURMD_NODENAME"
echo "Partition           : $SLURM_JOB_PARTITION"
echo "Job Name            : $SLURM_JOB_NAME"
echo "Start Time          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Working Directory   : $(pwd)"
echo "Target Workloads    : Model 00 Remaining 3 Benchmarks (Caltech No-Weather, JPN No-Weather, JPN With-Weather)"
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

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'Device 0: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

TOTAL_START=$(date +%s)

# ==============================================================================
# Workload 1: Model 00 on Caltech without Weather
# ==============================================================================
echo ""
echo "=============================================================================="
echo ">>> [1/3 STARTED] Model 00 Caltech No-Weather at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
T1_START=$(date +%s)
python -u 00_tfm_custom_caltech_no_weather.py
T1_EXIT=$?
T1_END=$(date +%s)
T1_DIFF=$((T1_END - T1_START))

if [ ${T1_EXIT} -eq 0 ]; then
    echo ">>> [1/3 SUCCESS] Model 00 Caltech No-Weather COMPLETED in $((T1_DIFF / 60))m (${T1_DIFF}s)"
    mkdir -p outputs/acn_caltech_no_weather/00_tfm_custom_pytorch
    cp 00_tfm_custom_caltech_no_weather_results.json outputs/acn_caltech_no_weather/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json 2>/dev/null || true
    cp 00_tfm_custom_caltech_no_weather_best.pt outputs/acn_caltech_no_weather/00_tfm_custom_pytorch/00_tfm_custom_pytorch_best.pt 2>/dev/null || true
    cp 00_tfm_custom_caltech_no_weather_predictions.npz outputs/acn_caltech_no_weather/00_tfm_custom_pytorch/00_tfm_custom_pytorch_predictions.npz 2>/dev/null || true
else
    echo ">>> [1/3 FAILED] Model 00 Caltech No-Weather FAILED with exit code ${T1_EXIT}"
fi

# ==============================================================================
# Workload 2: Model 00 on JPN without Weather
# ==============================================================================
echo ""
echo "=============================================================================="
echo ">>> [2/3 STARTED] Model 00 JPN No-Weather at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
T2_START=$(date +%s)
python -u 00_tfm_custom_jpn_no_weather.py
T2_EXIT=$?
T2_END=$(date +%s)
T2_DIFF=$((T2_END - T2_START))

if [ ${T2_EXIT} -eq 0 ]; then
    echo ">>> [2/3 SUCCESS] Model 00 JPN No-Weather COMPLETED in $((T2_DIFF / 60))m (${T2_DIFF}s)"
    mkdir -p outputs/acn_jpn_no_weather/00_tfm_custom_pytorch
    cp 00_tfm_custom_jpn_no_weather_results.json outputs/acn_jpn_no_weather/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json 2>/dev/null || true
    cp 00_tfm_custom_jpn_no_weather_best.pt outputs/acn_jpn_no_weather/00_tfm_custom_pytorch/00_tfm_custom_pytorch_best.pt 2>/dev/null || true
    cp 00_tfm_custom_jpn_no_weather_predictions.npz outputs/acn_jpn_no_weather/00_tfm_custom_pytorch/00_tfm_custom_pytorch_predictions.npz 2>/dev/null || true
else
    echo ">>> [2/3 FAILED] Model 00 JPN No-Weather FAILED with exit code ${T2_EXIT}"
fi

# ==============================================================================
# Workload 3: Model 00 on JPN with Weather
# ==============================================================================
echo ""
echo "=============================================================================="
echo ">>> [3/3 STARTED] Model 00 JPN with Weather at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
T3_START=$(date +%s)
python -u 00_tfm_custom_jpn.py
T3_EXIT=$?
T3_END=$(date +%s)
T3_DIFF=$((T3_END - T3_START))

if [ ${T3_EXIT} -eq 0 ]; then
    echo ">>> [3/3 SUCCESS] Model 00 JPN with Weather COMPLETED in $((T3_DIFF / 60))m (${T3_DIFF}s)"
    mkdir -p outputs/acn_jpn/00_tfm_custom_pytorch
    cp 00_tfm_custom_jpn_results.json outputs/acn_jpn/00_tfm_custom_pytorch/00_tfm_custom_pytorch_results.json 2>/dev/null || true
    cp 00_tfm_custom_jpn_best.pt outputs/acn_jpn/00_tfm_custom_pytorch/00_tfm_custom_pytorch_best.pt 2>/dev/null || true
    cp 00_tfm_custom_jpn_predictions.npz outputs/acn_jpn/00_tfm_custom_pytorch/00_tfm_custom_pytorch_predictions.npz 2>/dev/null || true
else
    echo ">>> [3/3 FAILED] Model 00 JPN with Weather FAILED with exit code ${T3_EXIT}"
fi

TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
echo ""
echo "=============================================================================="
echo "ALL REMAINING BENCHMARKS FINISHED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total Wall Time: $((TOTAL_DIFF / 60))m (${TOTAL_DIFF}s)"
echo "Workload 1 (Caltech No-Weather) Exit: ${T1_EXIT}"
echo "Workload 2 (JPN No-Weather)     Exit: ${T2_EXIT}"
echo "Workload 3 (JPN With-Weather)   Exit: ${T3_EXIT}"
echo "=============================================================================="
