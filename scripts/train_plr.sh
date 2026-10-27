#!/usr/bin/env bash
# Train puffer_drive with Prioritized Level Replay.
#
# Overrides map_sampling_strategy to prioritized. Map scores default to MaxMC
# (MAP_SCORE=max_mc; also return, pvl, learnability). The queue length comes
# from controllable_counts.npy next to the training bins.
#
# Usage:
#   ./scripts/train_plr.sh 0 maxmc_10k
#   MAP_SCORE=learnability ./scripts/train_plr.sh 1 learn_10k
#   MAP_SCORE=pvl ./scripts/train_plr.sh 0 pvl_10k
#
# wandb group follows MAP_SCORE: max_mc → maxmc, learnability → learn.
#
# Args:
#   $1  GPU id
#   $2  experiment name; checkpoints go to /data/prdrive/experiments/<name>/
#
# Seeds are the three values below. --train.seed resets the vecenv,
# --vec.seed is the env/map seed passed into Drive (and LevelSampler).
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
MAP_DIR="${MAP_DIR:-/data/prdrive/binaries/training_80000}"
EVAL_MAP_DIR="${EVAL_MAP_DIR:-/data/prdrive/binaries/validation}"
COUNTS="${MAP_DIR}/controllable_counts.npy"
EVAL_COUNTS="${EVAL_MAP_DIR}/controllable_counts.npy"
SEEDS=(11 3 42)
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000000}"
NUM_MAPS="${NUM_MAPS:-10000}"
MAP_SCORE="${MAP_SCORE:-max_mc}"
case "${MAP_SCORE}" in
  max_mc) SCORE_GROUP="maxmc" ;;
  learnability) SCORE_GROUP="learn" ;;
  *) SCORE_GROUP="${MAP_SCORE}" ;;
esac

if [[ ! -f "${COUNTS}" ]]; then
  echo "missing ${COUNTS}" >&2
  exit 1
fi

export WANDB_ENTITY="${WANDB_ENTITY:-cilab-ma}"
WANDB_PROJECT="${WANDB_PROJECT:-scp_drive}"
WANDB_GROUP="${WANDB_GROUP:-${SCORE_GROUP}}"
TAG_PREFIX="${TAG_PREFIX:-${SCORE_GROUP}}"

EVAL_ARGS=()
if [[ ! -f "${EVAL_COUNTS}" ]]; then
  echo "no ${EVAL_COUNTS}; turning eval off for this run" >&2
  EVAL_ARGS+=(--eval.human-replay-eval False --eval.self-play-eval False)
fi

echo "========== puffer_drive PLR =========="
mkdir -p "${DATA_DIR}"

echo "  GPU=${GPU_ID}"
echo "  exp=${EXP_NAME}"
echo "  data_dir=${DATA_DIR}"
echo "  map_dir=${MAP_DIR}"
echo "  strategy=prioritized score=${MAP_SCORE}"
echo "  seeds=${SEEDS[*]}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
echo "  num_maps=${NUM_MAPS}"
echo "  wandb entity=${WANDB_ENTITY} project=${WANDB_PROJECT} group=${WANDB_GROUP}"
echo "======================================"

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo ">>> seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" puffer train puffer_drive \
    --env.map-sampling-strategy prioritized \
    --env.map-score "${MAP_SCORE}" \
    --env.map-dir "${MAP_DIR}" \
    --env.num-maps "${NUM_MAPS}" \
    --train.seed "${SEED}" \
    --vec.seed "${SEED}" \
    --train.total-timesteps "${TOTAL_TIMESTEPS}" \
    --train.data-dir "${DATA_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --tag "${TAG_PREFIX}" \
    --eval.render-human-replay-eval False \
    --eval.render-self-play-eval False \
    "${EVAL_ARGS[@]}"
done

echo ""
echo "Finished seeds: ${SEEDS[*]}"
