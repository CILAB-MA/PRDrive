#!/usr/bin/env bash
# Run the current puffer_drive setup once per seed, sequentially on one GPU.
#
# Usage:
#   ./scripts/train_seeds.sh 0 sp_10k
#   ./scripts/train_seeds.sh 1 sp_10k
#
# Args:
#   $1  GPU id
#   $2  experiment name; checkpoints go to /data/prdrive/experiments/<name>/
#
# Seeds are the three values below. --train.seed resets the vecenv,
# --vec.seed is the env/map seed passed into Drive.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -x "${ROOT}/.venv/bin/puffer" ]]; then
  export PATH="${ROOT}/.venv/bin:${PATH}"
elif ! command -v puffer >/dev/null 2>&1; then
  echo "puffer not found; activate .venv or install the package" >&2
  exit 1
fi

GPU_ID="${1:-${GPU_ID:-0}}"
EXP_NAME="${2:-${EXP_NAME:-}}"
if [[ -z "${EXP_NAME}" ]]; then
  echo "usage: $0 GPU_ID EXP_NAME" >&2
  exit 1
fi
DATA_DIR="/data/prdrive/experiments/${EXP_NAME}"
SEEDS=(11 3 42)
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000000}"
NUM_MAPS="${NUM_MAPS:-10000}"

export WANDB_ENTITY="${WANDB_ENTITY:-cilab-ma}"
WANDB_PROJECT="${WANDB_PROJECT:-scp_drive}"
WANDB_GROUP="${WANDB_GROUP:-pre}"
TAG_PREFIX="${TAG_PREFIX:-sp}"

echo "========== puffer_drive seeds =========="
mkdir -p "${DATA_DIR}"

echo "  GPU=${GPU_ID}"
echo "  exp=${EXP_NAME}"
echo "  data_dir=${DATA_DIR}"
echo "  seeds=${SEEDS[*]}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
echo "  num_maps=${NUM_MAPS}"
echo "  wandb entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${WANDB_GROUP}"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo ">>> seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" puffer train puffer_drive \
    --train.seed "${SEED}" \
    --vec.seed "${SEED}" \
    --train.total-timesteps "${TOTAL_TIMESTEPS}" \
    --train.data-dir "${DATA_DIR}" \
    --env.num-maps "${NUM_MAPS}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --tag "${TAG_PREFIX}-${SEED}" \
    --eval.render-human-replay-eval False \
    --eval.render-self-play-eval False 
done

echo ""
echo "Finished seeds: ${SEEDS[*]}"
