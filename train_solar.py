"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

GRAD_ACC_DIV = 1
if torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3090":
    torch._inductor.config.max_fusion_size = 2
    GRAD_ACC_DIV = 2


# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Model output basename (without extension)
    model_basename = os.environ.get("MODEL_BASENAME", "final_model")

    # Weights & Biases.
    wandb_project = os.environ.get("WANDB_PROJECT", "minimind")
    wandb_enabled = bool(int(os.environ.get("WANDB_ENABLED", "1")))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    checkpoint_every_val = bool(int(os.environ.get("CHECKPOINT_EVERY_VAL", "0")))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # -----------------------------
    # SOLAR DUS HYPERPARAMETERS
    # -----------------------------
    solar_enabled = bool(int(os.environ.get("SOLAR_ENABLED", "1")))
    solar_wallclock_seconds = float(os.environ.get("SOLAR_WALLCLOCK_SECONDS", "1000"))
    solar_layer_pattern = os.environ.get("SOLAR_LAYER_PATTERN", "")
    solar_warmup = int(os.environ.get("SOLAR_WARMUP", "50"))
    solar_warmdown = int(os.environ.get("SOLAR_WARMDOWN", "500"))
    solar_lr_mul = float(os.environ.get("SOLAR_LR_MUL", "1.0"))
    solar_resume_path = os.environ.get("SOLAR_RESUME_PATH", "")
    solar_dup_type = os.environ.get("SOLAR_DUP_TYPE", "dec")


# -----------------------------
# SOLAR DUS HELPER FUNCTIONS
# -----------------------------


def parse_layer_pattern(
    pattern: str, n_layers: int, dup_type: str = "full"
) -> list[int] | None:
    """
    Parse comma-separated 1-indexed layer pattern into 0-indexed list.

    Pattern is 1-indexed within each half (encoder/decoder).
    For each half, indices that are too large are skipped.

    dup_type controls how the pattern is applied:
    - "full": pattern applies to both encoder and decoder
    - "dec": pattern applies to decoder only, encoder unchanged
    - "tie": pattern applies to decoder only, encoder unchanged
    - "deconly": encoder completely removed, no skip connections, decoder-only

    Examples with n_layers=7 (3 enc + 4 dec), pattern="1,2,3,3,4":
      → full: [0,0,1,2, 3,4,5,5,6], skip map: 0→7,1→6,2→5,3→4
      → tie: [0,1,2, 3,4,5,5,6], skip map: 0→6,0→5,1→4,2→3
      → dec: [0,1,2, 3,4,5,5,6], skip map: 0→5,1→4,2→3

    Examples with n_layers=8 (4 enc + 4 dec), pattern="1,2,3,3,4":
      → full: [0,1,1,2,3, 4,5,6,6,7], skip map: 0→9,1→8,2→7,3→6,4→5
      → tie: [0,1,2,3 4,5,6,6,7], skip map: 0→8,1→7,1→6,2→5,3→4
      → dec: [0,1,2,3 4,5,6,6,7], skip map: 0→8,1→6,2→5,3→4
    """
    if dup_type not in ("full", "dec", "tie", "deconly"):
        raise ValueError(f"dup_type must be 'full', 'dec', 'tie', or 'deconly', got '{dup_type}'")
    indices = [int(i.strip()) for i in pattern.split(",")]
    if any(i < 1 or i > np.ceil(n_layers / 2) for i in indices):
        raise ValueError("Number from pattern not in range")
    if not indices:
        return None

    zero_indexed = [i - 1 for i in indices]  # 1-indexed to 0-indexed
    dec_indices = [i + n_layers // 2 for i in zero_indexed]
    if dup_type == "full":
        stop_enc_idx = n_layers // 2
        enc_indices = [stop_enc_idx - i - 1 for i in zero_indexed if i < stop_enc_idx][
            ::-1
        ]
    elif dup_type == "deconly":
        enc_indices = []
    else:
        enc_indices = [i for i in range(n_layers // 2) if i in zero_indexed]

    return enc_indices + dec_indices


def extract_optimizer_states_for_dus(base_model: GPT, optimizers: list) -> dict:
    param_to_key: dict[int, tuple] = {}
    for layer_idx, block in enumerate(base_model.blocks):
        for name, p in block.named_parameters():
            param_to_key[id(p)] = ("block", layer_idx, name)
    param_to_key[id(base_model.tok_emb.weight)] = ("tok_emb",)
    if base_model.lm_head is not None:
        param_to_key[id(base_model.lm_head.weight)] = ("lm_head",)
    if base_model.num_skip_weights > 0:
        param_to_key[id(base_model.skip_weights)] = ("skip_weights",)
    states: dict[tuple, dict] = {}
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in param_to_key and p in opt.state:
                    key = param_to_key[id(p)]
                    states[key] = {
                        k: v.detach().cpu().clone() if isinstance(v, Tensor) else v
                        for k, v in opt.state[p].items()
                    }
    return states


def _get_skip_weights_src_indices(
    layer_pattern: list[int],
    num_encoder_layers: int,
    num_skip_weights: int,
    dup_type: str,
) -> list[int] | None:
    if dup_type == "full":
        dus_enc_count = sum(x < num_encoder_layers for x in layer_pattern)
        return [x for x in layer_pattern[:dus_enc_count] if x is not None]
    if dup_type == "deconly":
        return []
    dec_to_enc_idx: list[int | None] = []
    decoder_pattern_relative = [
        x - num_encoder_layers for x in layer_pattern[num_skip_weights:]
    ]
    for ir in decoder_pattern_relative:
        enc_idx = num_encoder_layers - ir - 1
        if ir == num_skip_weights:
            dec_to_enc_idx.append(None)
        elif enc_idx not in dec_to_enc_idx:
            dec_to_enc_idx.append(enc_idx)
        elif dup_type == "dec":
            dec_to_enc_idx.append(None)
        else:
            dec_to_enc_idx.append(enc_idx)
    return [num_encoder_layers - 1 - x for x in dec_to_enc_idx if x is not None]


def inject_optimizer_states_for_dus(
    new_optimizers: list,
    base_model: GPT,
    layer_pattern: list[int],
    saved_states: dict,
    device: torch.device,
    dup_type: str = "full",
) -> None:
    """
    The idea is simple, if we duplicate layer/weight x then it should also inherit/clone the states of x.
    """
    param_to_key: dict[int, tuple] = {}
    for pos in range(len(layer_pattern)):
        for name, p in base_model.blocks[pos].named_parameters():
            param_to_key[id(p)] = ("block", layer_pattern[pos], name)
    param_to_key[id(base_model.tok_emb.weight)] = ("tok_emb",)
    if base_model.lm_head is not None:
        param_to_key[id(base_model.lm_head.weight)] = ("lm_head",)
    if base_model.num_skip_weights > 0:
        param_to_key[id(base_model.skip_weights)] = ("skip_weights",)

    skip_src_indices = _get_skip_weights_src_indices(
        layer_pattern,
        base_model.num_encoder_layers,
        base_model.num_skip_weights,
        dup_type,
    )

    for opt in new_optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in param_to_key:
                    key = param_to_key[id(p)]
                    if key in saved_states:
                        saved_state = saved_states[key]
                        if (
                            key == ("skip_weights",)
                            and skip_src_indices is not None
                            and saved_state.get("exp_avg") is not None
                        ):
                            exp_avg = saved_state["exp_avg"].to(
                                device=device, dtype=p.dtype
                            )
                            exp_avg_sq = saved_state["exp_avg_sq"].to(
                                device=device, dtype=p.dtype
                            )
                            if exp_avg.shape != p.shape:
                                new_exp_avg = exp_avg[skip_src_indices]
                                new_exp_avg_sq = exp_avg_sq[skip_src_indices]
                                opt.state[p] = {
                                    k: v
                                    for k, v in saved_state.items()
                                    if not isinstance(v, Tensor)
                                }
                                opt.state[p]["exp_avg"] = new_exp_avg
                                opt.state[p]["exp_avg_sq"] = new_exp_avg_sq
                                if "step" in saved_state:
                                    opt.state[p]["step"] = saved_state[
                                        "step"
                                    ].to(device=device)
                            else:
                                opt.state[p] = {
                                    k: v.to(device=device, dtype=p.dtype)
                                    if isinstance(v, Tensor)
                                    else v
                                    for k, v in saved_state.items()
                                }
                        else:
                            opt.state[p] = {
                                k: v.to(device=device, dtype=p.dtype)
                                if isinstance(v, Tensor)
                                else v
                                for k, v in saved_state.items()
                            }


def load_solar_checkpoint(path: str, device: torch.device) -> dict | None:
    """Load SOLAR checkpoint with model state, optimizer states, and dataloader state."""
    try:
        checkpoint = torch.load(path, map_location=device)
        return {
            "model_state": checkpoint.get("model", checkpoint),
            "optimizer_states": checkpoint.get("optimizer_states", []),
            "dataloader_state": checkpoint.get("dataloader_state", None),
            "hyperparameters": checkpoint.get("hyperparameters", {}),
            "optimizer_states_by_layer": checkpoint.get(
                "optimizer_states_by_layer", None
            ),
        }
    except Exception as e:
        print(f"Warning: Failed to load SOLAR checkpoint from {path}: {e}")
        return None


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = (
            self.c_q(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.c_k(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# SOLAR DUS MODEL WRAPPER
# -----------------------------


class GPTWithSOLARDUS(torch.nn.Module):
    """
    GPT wrapper that applies SOLAR Depth Up-Scaling (DUS) layer pattern.

    Supports three modes:
    1. Symmetric ("full"): encoder pattern is mirrored to decoder
    2. Decoder-only ("dec"): encoder unchanged, only decoder is duplicated
    3. Decoder-only ("tie"): encoder unchanged, only decoder is duplicated, encoder layers are tied meaning single encoder layer can skip to multiple decoders
    4. Decoder-only ("deconly"): encoder completely removed, no skip connections, decoder-only
    """

    def __init__(
        self, base_model: GPT, layer_pattern: list[int], dup_type: str = "full"
    ):
        super().__init__()
        self.base_model = base_model
        self.layer_pattern = layer_pattern
        self.dup_type = dup_type
        self.num_total_layers = len(layer_pattern)
        self.num_og_layers = (
            self.base_model.num_encoder_layers + self.base_model.num_decoder_layers
        )
        self.num_encoder_layers = (
            sum(x < self.base_model.num_encoder_layers for x in self.layer_pattern)
            if self.dup_type == "full"
            else 0
            if self.dup_type == "deconly"
            else self.base_model.num_encoder_layers
        )
        self.dec_to_enc_idx = self._build_enc_mapping()
        if self.dup_type == "full":
            self.base_model.skip_weights = nn.Parameter(
                self.base_model.skip_weights[
                    torch.tensor(
                        [
                            x
                            for x in self.layer_pattern[: self.num_encoder_layers]
                            if x is not None
                        ]
                    )
                ]
            )
        elif self.dup_type == "deconly":
            del self.base_model.skip_weights
            self.base_model.skip_weights = torch.empty(0, self.base_model.tok_emb.weight.shape[1], dtype=torch.float32)
            self.base_model.num_skip_weights = 0
        else:
            self.base_model.skip_weights = nn.Parameter(
                self.base_model.skip_weights[
                    torch.tensor([self.num_encoder_layers - 1 - x for x in self.dec_to_enc_idx if x is not None])
                ]
            )
        self.base_model.blocks = nn.ModuleList(
            [copy.deepcopy(self.base_model.blocks[i]) for i in self.layer_pattern]
        )

    def _build_enc_mapping(self) -> list[int]:
        """
        Build mapping from decoder position to enc index.
        None indicates that given decoder layer has NO skip to any encoder layer.
        """
        if self.dup_type == "full":
            num_dec_layers = len(self.layer_pattern) - self.num_encoder_layers
            mapping = num_dec_layers * [-1]
            cenc = self.num_encoder_layers
            for i in range(num_dec_layers):
                if (
                    self.num_og_layers % 2 == 1
                    and self.layer_pattern[self.num_encoder_layers + i]
                    == self.num_og_layers - 1
                ):
                    mapping[i] = None
                else:
                    cenc -= 1
                    mapping[i] = cenc
            return mapping

        if self.dup_type == "deconly":
            return [None] * len(self.layer_pattern)

        decoder_pattern_relative = [
            x - self.num_encoder_layers
            for x in self.layer_pattern[self.base_model.num_skip_weights :]
        ]
        mapping = []
        for ir in decoder_pattern_relative:
            enc_idx = self.num_encoder_layers - ir - 1
            if ir == self.base_model.num_skip_weights:
                mapping.append(None)
            elif enc_idx not in mapping:
                mapping.append(enc_idx)
            elif self.dup_type == "dec":
                mapping.append(None)
            else:
                mapping.append(enc_idx)
        return mapping

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.base_model.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.base_model.blocks[i](x, x0)
            skips.append(x)
        nnones = 0
        for i in range(self.num_total_layers - self.num_encoder_layers):
            if self.dec_to_enc_idx[i] is not None:
                x = (
                    x
                    + self.base_model.skip_weights[i - nnones].to(dtype=x.dtype)[
                        None, None, :
                    ]
                    * skips[self.dec_to_enc_idx[i]]
                )
            else:
                nnones += 1
            x = self.base_model.blocks[self.num_encoder_layers + i](x, x0)

        x = self.base_model.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.base_model.tie_embeddings:
            logits_proj = F.linear(x, self.base_model.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.base_model.lm_head(x)
        logits = self.base_model.logit_softcap * torch.tanh(
            logits_proj / self.base_model.logit_softcap
        )
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def apply_dus_model(
    base_model: GPT,
    pattern: list[int],
    device: torch.device,
    dup_type: str = "full",
) -> nn.Module:
    """
    Create wrapped model with SOLAR DUS layer pattern applied.

    Args:
        base_model: Original trained model
        pattern: 0-indexed list of layer indices to use
        device: Target device
        dup_type: Duplication type - "full", "dec", "tie", or "deconly"

    Returns:
        Wrapped GPT model (GPTWithSOLARDUS wrapper)
    """
    wrapped_model = GPTWithSOLARDUS(base_model, pattern, dup_type).to(device)
    return wrapped_model


def create_solar_optimizers(
    base_model: GPT,
    args: Hyperparameters,
    lr_mul: float = 1.0,
) -> list:
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.num_skip_weights > 0 and base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    token_lr *= lr_mul

    optimizers = []

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers.append(optimizer_tok)

    if not args.tie_embeddings and base_model.lm_head is not None:
        head_lr = args.head_lr * lr_mul
        optimizer_head = torch.optim.Adam(
            [
                {
                    "params": [base_model.lm_head.weight],
                    "lr": head_lr,
                    "base_lr": head_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_head)

    matrix_lr = args.matrix_lr * lr_mul
    optimizer_muon = Muon(
        matrix_params,
        lr=matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = matrix_lr
    optimizers.append(optimizer_muon)

    scalar_lr = args.scalar_lr * lr_mul
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers.append(optimizer_scalar)

    return optimizers


def solar_stage2_transition(
    base_model: GPT,
    pattern: list[int],
    dup_type: str,
    args: Hyperparameters,
    device: torch.device,
    train_loader,
    step: int,
    training_time_ms: float,
    saved_opt_states: dict | None = None,
) -> tuple:
    """
    Handle transition from stage 1 to stage 2 of SOLAR DUS.

    Returns:
        (wrapped_model, new_optimizers, new_train_loader, new_step, pattern)
    """
    # Save stage 1 model (raw state dict, compatible with train_gpt.py)
    stage1_model_path = f"{args.model_basename}_stage1.pt"
    torch.save(base_model.state_dict(), stage1_model_path)
    print(f"[SOLAR] Saved stage 1 model to {stage1_model_path}")

    # Save stage 1 checkpoint (with dataloader state for resume)
    solar_checkpoint = {
        "model": base_model.state_dict(),
        "dataloader_state": {
            "file_idx": train_loader.stream.file_idx,
            "pos": train_loader.stream.pos,
            "step": step,
        },
        "hyperparameters": {
            "num_layers": args.num_layers,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "mlp_mult": args.mlp_mult,
        },
        "optimizer_states_by_layer": saved_opt_states,
    }
    checkpoint_path = f"{args.model_basename}_solar_stage1.pt"
    torch.save(solar_checkpoint, checkpoint_path)
    print(
        f"[SOLAR] Saved stage 1 checkpoint (with dataloader state) to {checkpoint_path}"
    )

    # Apply DUS wrapper
    print(f"[SOLAR] Applying DUS with pattern length {len(pattern)}")
    new_model = apply_dus_model(base_model, pattern, device, dup_type=dup_type)

    # Compile the wrapped model
    compiled_model = torch.compile(new_model, dynamic=False, fullgraph=True)

    # Create stage 2 optimizers (optimizes the base_model's parameters directly)
    new_optimizers = create_solar_optimizers(base_model, args, lr_mul=args.solar_lr_mul)

    if saved_opt_states is not None:
        inject_optimizer_states_for_dus(
            new_optimizers,
            base_model,
            pattern,
            saved_opt_states,
            device,
            dup_type=dup_type,
        )
        print(
            f"[SOLAR] Injected optimizer states for {len(saved_opt_states)} param groups"
        )

    # Reset train loader (start from beginning for stage 2)
    new_train_loader = DistributedTokenLoader(
        args.train_files, train_loader.rank, train_loader.world_size, device
    )

    return new_model, new_optimizers, new_train_loader, 0, pattern


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    grad_accum_steps = 8 // world_size // GRAD_ACC_DIV
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    # -----------------------------
    # SOLAR RESUME CHECK
    # -----------------------------
    solar_stage2 = False
    solar_pattern = None
    if args.solar_enabled and args.solar_resume_path:
        ckpt = load_solar_checkpoint(args.solar_resume_path, device)
        if ckpt is not None:
            solar_stage2 = True
            # We'll handle this after model creation
            if args.solar_layer_pattern:
                solar_pattern = parse_layer_pattern(
                    args.solar_layer_pattern,
                    ckpt["hyperparameters"].get("num_layers", args.num_layers),
                    args.solar_dup_type,
                )
            else:
                raise ValueError("SOLAR_LAYER_PATTERN must be provided for stage 2")
            print(f"[SOLAR] Resuming from checkpoint for stage 2 training")
            print(f"[SOLAR] Layer pattern: {len(solar_pattern)} layers")
            print(f"[SOLAR] Stage 2 time limit: {args.solar_wallclock_seconds}s")

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    if master_process and args.wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=args.run_id,
            config=vars(args),
            save_code=False,
        )

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only setup for SentencePiece .model file: {args.tokenizer_path}"
        )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    if solar_stage2:
        # SOLAR Stage 2: Load checkpoint and apply DUS wrapper immediately
        ckpt = load_solar_checkpoint(args.solar_resume_path, "cpu")

        # Create base model from checkpoint config
        ckpt_hyp = ckpt["hyperparameters"]
        base_model = GPT(
            vocab_size=ckpt_hyp.get("vocab_size", args.vocab_size),
            num_layers=ckpt_hyp.get("num_layers", args.num_layers),
            model_dim=ckpt_hyp.get("model_dim", args.model_dim),
            num_heads=ckpt_hyp.get("num_heads", args.num_heads),
            num_kv_heads=ckpt_hyp.get("num_kv_heads", args.num_kv_heads),
            mlp_mult=ckpt_hyp.get("mlp_mult", args.mlp_mult),
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        base_model.load_state_dict(ckpt["model_state"])

        # Apply DUS wrapper (returns wrapped model, not a new GPT)
        wrapped_model = (
            apply_dus_model(base_model, solar_pattern, device, args.solar_dup_type)
            .to(device)
            .bfloat16()
        )
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(base_model)
        compiled_model = torch.compile(wrapped_model, dynamic=False, fullgraph=True)
        model: nn.Module = (
            DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
            if distributed
            else compiled_model
        )

        # Create stage 2 optimizers (optimizes base_model parameters directly)
        optimizers = create_solar_optimizers(base_model, args, lr_mul=args.solar_lr_mul)
        saved_opt_states = ckpt.get("optimizer_states_by_layer")
        if saved_opt_states:
            inject_optimizer_states_for_dus(
                optimizers,
                base_model,
                solar_pattern,
                saved_opt_states,
                device,
                dup_type=args.solar_dup_type,
            )
            print(
                f"[SOLAR] Restored optimizer states for {len(saved_opt_states)} param groups"
            )
    else:
        # Normal training or SOLAR stage 1
        base_model = (
            GPT(
                vocab_size=args.vocab_size,
                num_layers=args.num_layers,
                model_dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                tie_embeddings=args.tie_embeddings,
                tied_embed_init_std=args.tied_embed_init_std,
                logit_softcap=args.logit_softcap,
                rope_base=args.rope_base,
                qk_gain_init=args.qk_gain_init,
            )
            .to(device)
            .bfloat16()
        )
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(base_model)
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
        model: nn.Module = (
            DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
            if distributed
            else compiled_model
        )

        optimizers = create_solar_optimizers(base_model, args)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}"
    )
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float, stage2: bool = False) -> float:
        # Stage 2 uses different warmup/warmdown
        if stage2:
            if step < args.solar_warmup:
                return (step + 1) / args.solar_warmup
            if max_wallclock_ms is not None:
                remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
                warmdown_ms = args.solar_warmdown * (elapsed_ms / max(step, 1))
                if remaining_ms <= warmdown_ms:
                    return max(remaining_ms / max(warmdown_ms, 1e-9), 0.0)
            return 1.0

        # Stage 1 / normal training
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    solar_transition_done = False
    # Initialize stage2_wallclock_ms for resume mode
    stage2_wallclock_ms = (
        1000.0 * args.solar_wallclock_seconds if solar_stage2 else None
    )

    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )

        should_validate = last_step or (
            args.val_loss_every > 0 and step % args.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if master_process and args.wandb_enabled:
                wandb.log(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "val_bpb": val_bpb,
                        "train_time_ms": training_time_ms,
                    },
                    step=step,
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if master_process and args.checkpoint_every_val and not last_step:
                ckpt_path = f"{args.model_basename}_step{step}.pt"
                save_dict = {
                    "model": base_model.state_dict(),
                    "dataloader_state": {
                        "file_idx": train_loader.stream.file_idx,
                        "pos": train_loader.stream.pos,
                        "step": step,
                    },
                    "hyperparameters": {
                        "vocab_size": args.vocab_size,
                        "num_layers": args.num_layers,
                        "model_dim": args.model_dim,
                        "num_heads": args.num_heads,
                        "num_kv_heads": args.num_kv_heads,
                        "mlp_mult": args.mlp_mult,
                    },
                    "optimizer_states_by_layer": extract_optimizer_states_for_dus(
                        base_model, optimizers
                    ),
                }
                torch.save(save_dict, ckpt_path)
                log0(f"saved checkpoint: {ckpt_path}")

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                stage_str = "[SOLAR Stage 2] " if solar_stage2 else ""
                log0(
                    f"{stage_str}stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # SOLAR Stage Transition Check
        if args.solar_enabled and not solar_stage2 and not solar_transition_done:
            solar_wallclock_ms = 1000.0 * args.solar_wallclock_seconds
            max_wallclock_ms = 1000.0 * args.max_wallclock_seconds
            # Transition when we have SOLAR_WALLCLOCK_SECONDS remaining
            if elapsed_ms >= max_wallclock_ms - solar_wallclock_ms:
                log0(
                    f"[SOLAR] Stage 1 complete at {elapsed_ms:.0f}ms, transitioning to stage 2"
                )

                # Run validation before transitioning to show stage 1 metrics
                torch.cuda.synchronize()
                training_time_ms += 1000.0 * (time.perf_counter() - t0)
                val_loss, val_bpb = eval_val(
                    args,
                    model,
                    rank,
                    world_size,
                    device,
                    grad_accum_steps,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                log0(
                    f"[SOLAR] Stage 1 final: step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{training_time_ms:.0f}ms"
                )
                if master_process and args.wandb_enabled:
                    wandb.log(
                        {
                            "step": step,
                            "val_loss": val_loss,
                            "val_bpb": val_bpb,
                            "train_time_ms": training_time_ms,
                        },
                        step=step,
                    )
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Parse layer pattern
                solar_pattern = parse_layer_pattern(
                    args.solar_layer_pattern, args.num_layers, args.solar_dup_type
                )
                if solar_pattern is None:
                    raise ValueError("SOLAR_LAYER_PATTERN must be provided for stage 2")

                # Clean up stage 1 objects to free memory before transition
                log0("[SOLAR] Cleaning up stage 1 model and optimizers...")
                log0(
                    f"[SOLAR] CUDA memory before cleanup: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
                )
                saved_opt_states = extract_optimizer_states_for_dus(
                    base_model, optimizers
                )
                if distributed and isinstance(model, DDP):
                    model = model.module  # Unwrap DDP before deleting
                del model
                for opt in optimizers:
                    del opt
                del optimizers

                # Clear torch._dynamo compilation cache (critical!)
                torch._dynamo.reset()

                import gc

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log0(
                    f"[SOLAR] CUDA memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
                )
                log0(
                    f"[SOLAR] CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
                )

                # Apply stage transition
                model, optimizers, train_loader, step, solar_pattern = (
                    solar_stage2_transition(
                        base_model,
                        solar_pattern,
                        args.solar_dup_type,
                        args,
                        device,
                        train_loader,
                        step,
                        training_time_ms,
                        saved_opt_states=saved_opt_states,
                    )
                )

                # Wrap model for DDP
                if distributed:
                    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

                # Reset timing for stage 2
                training_time_ms = 0.0
                t0 = time.perf_counter()
                solar_transition_done = True
                solar_stage2 = True
                stage2_wallclock_ms = 1000.0 * args.solar_wallclock_seconds
                log0(
                    f"[SOLAR] Stage 2 starting with {len(solar_pattern)} layers, time limit: {args.solar_wallclock_seconds}s"
                )

                # Continue to next iteration
                continue

        scale = lr_mul(step, elapsed_ms, stage2=solar_stage2)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Handle Muon momentum warmup (for all Muon optimizers)
        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            if isinstance(opt, Muon):
                for group in opt.param_groups:
                    group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            if master_process and args.wandb_enabled:
                wandb.log(
                    {"step": step, "train_loss": train_loss.item(), "lr_mul": scale},
                    step=step,
                )

        # Needed to sync whether we've reached the wallclock cap.
        # In stage 2, use stage2_wallclock_ms; otherwise use max_wallclock_ms
        current_wallclock_cap = (
            stage2_wallclock_ms if solar_stage2 else max_wallclock_ms
        )
        reached_cap = (
            current_wallclock_cap is not None
            and approx_training_time_ms >= current_wallclock_cap
        )
        if distributed and current_wallclock_cap is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION
    # -----------------------------

    if master_process:
        if args.solar_enabled and (solar_transition_done or solar_stage2):
            model_pt_path = f"{args.model_basename}_stage2.pt"
            save_dict = {
                "model_type": "SOLAR_DUS",
                "layer_pattern": solar_pattern,
                "dup_type": args.solar_dup_type,
                "base_model_state": base_model.state_dict(),
            }
            torch.save(save_dict, model_pt_path)
        else:
            model_pt_path = f"{args.model_basename}_stage1.pt"
            save_dict = {
                "model": base_model.state_dict(),
                "dataloader_state": {
                    "file_idx": train_loader.stream.file_idx,
                    "pos": train_loader.stream.pos,
                    "step": step,
                },
                "hyperparameters": {
                    "vocab_size": args.vocab_size,
                    "num_layers": args.num_layers,
                    "model_dim": args.model_dim,
                    "num_heads": args.num_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "mlp_mult": args.mlp_mult,
                },
                "optimizer_states_by_layer": extract_optimizer_states_for_dus(
                    base_model, optimizers
                ),
            }
            torch.save(save_dict, model_pt_path)

        model_bytes = os.path.getsize(model_pt_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    if master_process and args.wandb_enabled:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
