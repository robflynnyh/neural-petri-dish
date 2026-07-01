#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${REPO_DIR}/test_cases/artifacts/gpu_mutation"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"

mkdir -p "${ARTIFACT_DIR}"
cd "${REPO_DIR}"

exec "${WITH_GPU}" any --idle-seconds "${GPU_IDLE_SECONDS:-5}" -- \
  python scripts/gpu_mutation_benchmark.py \
    --device cuda \
    --mode "${MUTATION_MODE:-shared_rank1_factored}" \
    --population "${POPULATION:-200000}" \
    --steps "${STEPS:-200}" \
    --output-json "${ARTIFACT_DIR}/${MUTATION_MODE:-shared_rank1_factored}_latest.json" \
    --output-csv "${ARTIFACT_DIR}/runs.csv" \
    "$@"
