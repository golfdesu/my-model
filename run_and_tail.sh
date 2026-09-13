#!/bin/bash
set -euo pipefail

# 1. Submit batch job and extract Job ID
SUBMIT_OUTPUT=$(sbatch run_benchmark_00_v4_all_h100.sbatch)
echo "${SUBMIT_OUTPUT}"

JOB_ID=$(echo "${SUBMIT_OUTPUT}" | awk '{print $NF}')
LOG_FILE="logs/bench_00_v4_all_${JOB_ID}.out"

echo "=============================================================================="
echo ">>> Job ID: ${JOB_ID}"
echo ">>> Waiting for ${LOG_FILE} to start streaming..."
echo "=============================================================================="

# 2. Wait until SLURM creates the log file
while [[ ! -f "${LOG_FILE}" ]]; do
    sleep 1
done

# 3. Stream log live
echo ">>> Streaming live logs (Press Ctrl+C to detach without canceling the job):"
echo "------------------------------------------------------------------------------"
tail -n +1 -f "${LOG_FILE}"
