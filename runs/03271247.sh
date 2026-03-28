#!/bin/bash
set +e

MODEL_BASENAME=models/03271247 MAX_WALLCLOCK_SECONDS=12000 WARMDOWN_ITERS=4000 NUM_LAYERS=20 MODEL_DIM=512 uv run torchrun --standalone --nproc_per_node=2 train_gpt.py
