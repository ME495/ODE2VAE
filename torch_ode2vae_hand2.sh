#!/usr/bin/env sh

set -eu

DATASET_ROOT="/media/disk_16T_2026/dataset/GigaHands"
SPLIT_DIR="runs/hand_motion_splits"
LOGDIR="runs/ode2vae_hand_bnn_elbo17"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
METHOD="rk4"
DYNAMICS_DAMPING="0.02"
BETA_KL="1e-5"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python torch_ode2vae_hand.py \
    "${DATASET_ROOT}" \
    --train-text-file "${SPLIT_DIR}/train.jsonl" \
    --val-text-file "${SPLIT_DIR}/val.jsonl" \
    --history-len 10 \
    --horizon 10 \
    --warmup-horizon 3 \
    --warmup-epochs 300 \
    --curriculum-epochs 2000 \
    --epochs 20000 \
    --time-stride-aug-max 1 \
    --fps 30 \
    --method "${METHOD}" \
    --num-workers 10 \
    --val-stride 10 \
    --val-every-steps 200 \
    --batch-size 512 \
    --q 64 \
    --hidden-dim 256 \
    --dynamics-damping "${DYNAMICS_DAMPING}" \
    --lr 5e-4 \
    --lr-decay-factor 0.5 \
    --lr-decay-patience 200 \
    --min-lr 1e-5 \
    --future-discount 1.0 \
    --lambda-init-state 1.0 \
    --lambda-init-vel 1.0 \
    --lambda-delta-p 3.0 \
    --lambda-root-rot 1.0 \
    --lambda-nu 2.0 \
    --lambda-omega 2.0 \
    --lambda-pose 1.0 \
    --lambda-joint 1.0 \
    --lambda-vert 0.0 \
    --lambda-boundary-vel 20.0 \
    --lambda-speed-mag 10.0 \
    --lambda-vel 0.75 \
    --beta-kl "${BETA_KL}" \
    --logdir "${LOGDIR}" \
    --best-of-k 4 \
    --best-of-k-aux-weight 0.1 \
    --lambda-dpose 0.05 \
    --lambda-dpose-cons 0.1
