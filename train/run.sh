#!/usr/bin/env bash
# Run the advice-detector trainer in the DGX Spark training container.
#
# Reuses the image built by the gemma4-groundedness-judge repo (gj-train:latest):
# the code is bind-mounted, so nothing needs rebuilding for this project.
#
# Usage:
#   bash train/run.sh python train/train_advice.py --config configs/train_advice.yaml
#   bash train/run.sh python train/train_advice.py --config configs/train_advice.yaml --max-steps 20   # smoke
#   bash train/run.sh                                                                   # interactive shell
#
# Environment:
#   ADVICE_TRAIN_IMAGE  image tag (default gj-train:latest)
#   JUDGE_REPO          where to find the Dockerfile if the image is missing
#   CACHE_DIR           HF cache (defaults to the judge repo's, so Gemma weights are reused)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO/.env" ]] && { set -a; source "$REPO/.env"; set +a; }

IMAGE="${ADVICE_TRAIN_IMAGE:-gj-train:latest}"
JUDGE_REPO="${JUDGE_REPO:-$HOME/repos/gemma4-groundedness-judge}"
CACHE_DIR="${CACHE_DIR:-$JUDGE_REPO/cache}"
[[ -d "$CACHE_DIR" ]] || CACHE_DIR="$REPO/cache"
mkdir -p "$CACHE_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ -d "$JUDGE_REPO/train" ]]; then
    echo ">> $IMAGE missing — building from $JUDGE_REPO/train (several minutes)..."
    docker build -t "$IMAGE" "$JUDGE_REPO/train"
  else
    echo "!! $IMAGE not found and no Dockerfile at $JUDGE_REPO/train." >&2
    echo "   Set JUDGE_REPO, or build the image from that repo first." >&2
    exit 1
  fi
fi

TTY_FLAGS=(); [ -t 0 ] && [ -t 1 ] && TTY_FLAGS=(-it)

exec docker run --rm "${TTY_FLAGS[@]}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/cache \
  -e HF_HOME=/workspace/cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e BNB_CUDA_VERSION=130 \
  -e WANDB_DISABLED="${WANDB_DISABLED:-true}" \
  -v "$REPO:/workspace/demo" \
  -v "$CACHE_DIR:/workspace/cache" \
  -w /workspace/demo \
  "$IMAGE" \
  "${@:-bash}"
