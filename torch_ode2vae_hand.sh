#!/usr/bin/env sh

set -eu

DATASET_ROOT="/media/disk_16T_2026/dataset/GigaHands"
SPLIT_DIR="runs/hand_motion_splits"
LOGDIR="runs/ode2vae_hand_bnn_elbo"
GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"
METHOD="rk4"
DYNAMICS_DAMPING="0.0"
BETA_KL="1e-4"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python torch_ode2vae_hand.py \
    "${DATASET_ROOT}" \
    --train-text-file "${SPLIT_DIR}/train.jsonl" \
    --val-text-file "${SPLIT_DIR}/val.jsonl" \
    --history-len 10 \
    --horizon 10 \
    --warmup-horizon 3 \
    --warmup-epochs 20 \
    --curriculum-epochs 80 \
    --epochs 5000 \
    --time-stride-aug-max 2 \
    --fps 30 \
    --method "${METHOD}" \
    --random-mask \
    --random-mask-prob 0.15 \
    --num-workers 10 \
    --val-stride 10 \
    --val-every-steps 50 \
    --batch-size 512 \
    --q 64 \
    --hidden-dim 256 \
    --dynamics-damping "${DYNAMICS_DAMPING}" \
    --lr 5e-4 \
    --lr-decay-factor 0.5 \
    --lr-decay-patience 50 \
    --min-lr 1e-6 \
    --future-discount 1.0 \
    --lambda-init-state 1.0 \
    --lambda-init-vel 0.5 \
    --lambda-delta-p 2.0 \
    --lambda-root-rot 1.0 \
    --lambda-nu 1.0 \
    --lambda-omega 1.0 \
    --lambda-pose 1.0 \
    --lambda-joint 1.0 \
    --lambda-vert 0.0 \
    --lambda-vel 0.25 \
    --beta-kl "${BETA_KL}" \
    --logdir "${LOGDIR}" \
    --resume-checkpoint runs/ode2vae_hand_bnn_elbo/checkpoints/step_0012000.pt
