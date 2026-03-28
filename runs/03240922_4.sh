#!/bin/bash
set +e

export TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=500 USE_SIMPLE_TIMESTEP=1 TIME_EMBED_DIM=64 WARMUP_STEPS=20

# Sweep AR_TRAIN_FRACTION 0.6 -> 0.4
for AR in 0.6 0.5 0.4; do
  echo "=== AR_TRAIN_FRACTION=$AR WALLCLOCK=3000 ==="
  MAX_WALLCLOCK_SECONDS=3000 NUM_LAYERS=7 AR_TRAIN_FRACTION=$AR uv run torchrun --standalone --nproc_per_node=2 train_dgpt.py || true
done

# Sweep AR_TRAIN_FRACTION 1.0 -> 0.6
for AR in 1 0.8 0.6; do
  echo "=== AR_TRAIN_FRACTION=$AR WALLCLOCK=3000 ==="
  MAX_WALLCLOCK_SECONDS=12000 NUM_LAYERS=9 MODEL_DIM=384 AR_TRAIN_FRACTION=$AR uv run torchrun --standalone --nproc_per_node=2 train_dgpt.py || true
done