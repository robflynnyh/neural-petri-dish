#!/usr/bin/env bash
# Template for Neural Petri Dish long-running experiments submitted on Stanage with Slurm.
# Copy this file to an issue-specific path, fill in the variables and command,
# and keep the EXIT trap intact so Linear wakes Symphony when the job ends.
# Submit from Stanage with: sbatch <issue-specific-wrapper>.sh
#
# For sweeps or GPU arrays, prefer a separate lightweight finalizer job submitted
# with --dependency=afterany:<array_job_id>. That finalizer can inspect all array
# logs/results and call scripts/callbacks/linear_experiment_callback.py once.

#SBATCH --job-name=npd-example
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -uo pipefail

LINEAR_ENV_FILE="${LINEAR_ENV_FILE:-${HOME}/.config/neural-petri-dish/linear.env}"
if [ -f "${LINEAR_ENV_FILE}" ]; then
  set -a
  . "${LINEAR_ENV_FILE}"
  set +a
fi

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-neural-petri-dish/ROB-000}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-000}"
RESULTS_PATH="${RESULTS_PATH:-test_cases/artifacts/example}"
LOG_PATH="${LOG_PATH:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/npd-example-${SLURM_JOB_ID:-manual}.log}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm:${SLURM_JOB_ID:-unknown}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch ${SLURM_SUBMIT_SCRIPT:-scripts/templates/slurm_experiment_wrapper.template.sh}}"

cd "${REPO_DIR}"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ -z "${LINEAR_API_KEY:-}" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  python3 scripts/callbacks/linear_experiment_callback.py \
    --issue "${LINEAR_ISSUE}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${RESULTS_PATH}" \
    --runner-label "${RUNNER_LABEL}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULTS_PATH}"

if command -v module >/dev/null 2>&1; then
  module load Anaconda3/2022.10
fi
if [ -n "${CONDA_ENV:-}" ]; then
  # shellcheck disable=SC1090
  source activate "${CONDA_ENV}"
fi

# Replace this block with the real experiment command. Keep output tee'd or
# redirected to LOG_PATH so the callback can include useful failure evidence.
PYTHONPATH=. python3 scripts/benchmark_tensor_normal_rounds.py --device cpu --rounds 1 2>&1 | tee -a "${LOG_PATH}"
