"""
Discrete Diffusion Language Model Training (LLADA-like)

This script trains a diffusion-based language model where the model learns to denoise
discrete tokens. The key differences from autoregressive GPT are:
- Training objective: Denoise x_t -> x_0 instead of next-token prediction
- Attention: Bidirectional instead of causal
- Evaluation: Both autoregressive BPB (for comparison) and diffusion ELBO

Reference: D3PM (Structured Denoising Diffusion Models in Discrete State-Spaces)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

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
    torch._inductor.config.max_fusion_size = 4
    GRAD_ACC_DIV = 2


# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get('DATA_PATH', './data/datasets/fineweb10B_sp1024')
    train_files = os.path.join(data_path, 'fineweb_train_*.bin')
    val_files = os.path.join(data_path, 'fineweb_val_*.bin')
    tokenizer_path = os.environ.get('TOKENIZER_PATH', './data/tokenizers/fineweb_1024_bpe.model')
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    seed = int(os.environ.get('SEED', 1337))

    # Weights & Biases.
    wandb_project = os.environ.get('WANDB_PROJECT', 'minimind')
    wandb_enabled = bool(int(os.environ.get('WANDB_ENABLED', '1')))

    # Validation cadence and batch size.
    val_batch_size = int(os.environ.get('VAL_BATCH_SIZE', 524_288))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 1000))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 200))

    # Training length.
    times = int(os.environ.get('TIMES', 1))
    iterations = int(os.environ.get('ITERATIONS', 20000 / times))
    warmdown_iters = int(os.environ.get('WARMDOWN_ITERS', 1200))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', times * 524_288))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 1024))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 1.5))

    # Model shape.
    vocab_size = int(os.environ.get('VOCAB_SIZE', 1024))
    num_layers = int(os.environ.get('NUM_LAYERS', 9))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = int(os.environ.get('MLP_MULT', 2))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))

    # Diffusion-specific parameters.
    num_diffusion_steps = int(os.environ.get('NUM_DIFFUSION_STEPS', 1000))
    noise_type = os.environ.get('NOISE_TYPE', 'uniform')  # 'uniform', 'absorbing'
    time_embed_dim = int(os.environ.get('TIME_EMBED_DIM', 256))  # Internal dim for timestep embedding
    loss_type = os.environ.get('LOSS_TYPE', 'cross_entropy')  # 'cross_entropy', 'vb', 'hybrid'
    # Probability of replacing with random token at t=T (for uniform noise)
    noise_max_prob = float(os.environ.get('NOISE_MAX_PROB', 0.95))
    # Minimum timestep to use during training (0 allows clean token prediction)
    min_train_timestep = int(os.environ.get('MIN_TRAIN_TIMESTEP', 0))
    # Whether to use causal attention in diffusion model (for AR eval compatibility)
    # NOTE: For fair BPB comparison with GPT, causal attention is REQUIRED
    use_causal_attention = bool(int(os.environ.get('USE_CAUSAL_ATTENTION', '1')))
    # Fraction of training steps to use pure AR (next-token) training instead of diffusion
    # This ensures the model learns to predict next tokens, not just reconstruct
    ar_train_fraction = float(os.environ.get('AR_TRAIN_FRACTION', '0.1'))
    # Whether to use simple timestep conditioning (just add embedding, no adaLN)
    # Saves ~5M params but may reduce diffusion quality
    use_simple_timestep = bool(int(os.environ.get('USE_SIMPLE_TIMESTEP', '0')))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.05))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.04))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.04))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.95))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.85))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 500))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.0))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
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
            params = group['params']
            if not params:
                continue
            lr = group['lr']
            momentum = group['momentum']
            backend_steps = group['backend_steps']
            nesterov = group['nesterov']

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
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
# NOISE SCHEDULE FOR DISCRETE DIFFUSION
# -----------------------------


class DiscreteNoiseSchedule:
    """
    Manages the forward diffusion process for discrete tokens.

    Two noise types supported:
    - 'uniform': Tokens are replaced with uniform random samples from vocabulary
    - 'absorbing': Tokens transition to a special absorbing state (EOS token)

    The forward process is:
        q(x_t | x_{t-1}) = (1 - beta_t) * delta(x_t = x_{t-1}) + beta_t * Uniform()
    where beta_t grows from 0 to noise_max_prob over T steps.
    """

    def __init__(
        self,
        num_steps: int,
        vocab_size: int,
        noise_type: str = 'uniform',
        noise_max_prob: float = 0.95,
        device: torch.device | None = None,
    ):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.noise_type = noise_type
        self.noise_max_prob = noise_max_prob
        self.device = device

        # Compute cumulative noise probabilities alpha_bar_t
        # alpha_bar_t = product(1 - beta_s) for s = 1..t
        # We use a cosine schedule similar to improved DDPM
        t = torch.arange(num_steps + 1, dtype=torch.float32)
        s = 0.008
        f_t = torch.cos((t / num_steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]

        # Probability of keeping original token at timestep t
        self.keep_prob = alpha_bar

        # Probability of noise (replacing token) at timestep t
        self.noise_prob = 1.0 - alpha_bar

        # Precompute for device
        self.register_buffer('keep_prob', self.keep_prob)
        self.register_buffer('noise_prob', self.noise_prob)

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Register a buffer that moves with the module."""
        if self.device is not None and not tensor.is_cuda:
            tensor = tensor.to(self.device)
        setattr(self, f'_{name}', tensor)

    def get_keep_prob(self, t: Tensor) -> Tensor:
        """Get probability of keeping original token at timestep t."""
        # t values are in [0, num_steps]
        # Clamp to valid range and index into precomputed probabilities
        t_clamped = torch.clamp(t.long(), 0, self.num_steps)
        return self._keep_prob[t_clamped].to(t.device)

    def get_noise_prob(self, t: Tensor) -> Tensor:
        """Get probability of noise at timestep t."""
        t_clamped = torch.clamp(t.long(), 0, self.num_steps)
        return self._noise_prob[t_clamped].to(t.device)

    def sample_q(
        self,
        x0: Tensor,
        t: Tensor,
        rng: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample from the forward diffusion process q(x_t | x_0).

        Args:
            x0: Clean tokens, shape (batch_size, seq_len)
            t: Timestep, shape (batch_size,) or scalar
            rng: Random number generator

        Returns:
            xt: Noised tokens, shape (batch_size, seq_len)
            noise_mask: Boolean mask of which positions were noised, shape (batch_size, seq_len)
        """
        batch_size, seq_len = x0.shape
        device = x0.device

        # Get noise probability for each timestep
        noise_p = self.get_noise_prob(t)  # (batch_size,) or scalar

        # For each element, decide whether to apply noise
        # Create random values and compare with noise probability
        if rng is None:
            random_vals = torch.rand(batch_size, seq_len, device=device)
        else:
            random_vals = torch.rand(batch_size, seq_len, device=device, generator=rng)

        # Broadcast noise_p to match shape
        if noise_p.dim() == 1:
            noise_p = noise_p.view(-1, 1)

        # Create noise mask: True where noise should be applied
        noise_mask = random_vals < noise_p

        # Start with clean tokens
        xt = x0.clone()

        if self.noise_type == 'uniform':
            # Replace with uniform random tokens
            if noise_mask.any():
                num_noise = noise_mask.sum().item()
                noise_tokens = torch.randint(
                    0, self.vocab_size,
                    (num_noise,),
                    device=device,
                    generator=rng,
                )
                xt[noise_mask] = noise_tokens

        elif self.noise_type == 'absorbing':
            # Replace with EOS token (absorbing state)
            absorbing_token = self.vocab_size - 1  # Use last token as absorbing state
            xt[noise_mask] = absorbing_token

        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        return xt, noise_mask

    def to(self, device: torch.device) -> 'DiscreteNoiseSchedule':
        """Move noise schedule to device."""
        self.device = device
        self._keep_prob = self._keep_prob.to(device)
        self._noise_prob = self._noise_prob.to(device)
        return self


# -----------------------------
# TIMESTEP EMBEDDING
# -----------------------------


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timestep.

    Similar to the original DDPM timestep embedding, but adapted for
    discrete timesteps in language models.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

        # Create frequency matrix
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timestep, shape (batch_size,)

        Returns:
            embedding: Shape (batch_size, embed_dim)
        """
        # t: (batch_size,) -> (batch_size, 1)
        t = t.float()[:, None]
        # args: (batch_size, half_dim)
        args = t * self.freqs.to(t.device)
        # embedding: (batch_size, embed_dim)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class TimestepMLP(nn.Module):
    """MLP to project timestep embedding to model dimension."""

    def __init__(self, time_embed_dim: int, model_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            CastedLinear(time_embed_dim, model_dim),
            nn.SiLU(),
            CastedLinear(model_dim, model_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(t)


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
    """Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """Keep small/control parameters in fp32 even when the model body runs in bf16."""
    CONTROL_TENSOR_NAME_PATTERNS = (
        'attn_scale', 'attn_scales', 'mlp_scale', 'mlp_scales',
        'resid_mix', 'resid_mixes', 'q_gain', 'skip_weight', 'skip_weights',
        'time_mlp', 'adaLN',
    )
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """Caches cos/sin tables per sequence length on the current device."""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        # Clone cached tensors to avoid inference_mode issues during training
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class Attention(nn.Module):
    """Bidirectional attention for diffusion model."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_causal: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('model_dim must be divisible by num_heads')
        if num_heads % num_kv_heads != 0:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError('head_dim must be even for RoPE')
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_causal = use_causal

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
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
            is_causal=self.use_causal,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """relu^2 MLP from the original modded-nanogpt setup."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class DiffusionBlock(nn.Module):
    """
    Transformer block for diffusion model with timestep conditioning.

    Uses adaLN-style conditioning: timestep embedding is used to compute
    scale and shift for normalization layers.

    With use_simple_timestep: Just adds timestep embedding directly (no adaLN).
    Saves parameters but may reduce diffusion quality.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        time_embed_dim: int,
        use_causal: bool = False,
        use_simple_timestep: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_causal)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        # Timestep conditioning
        self.use_simple_timestep = use_simple_timestep
        if not use_simple_timestep:
            # Complex adaLN: ~500K params per block
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                CastedLinear(time_embed_dim, dim * 4),  # For scale/shift of attn and mlp
            )
            self.time_mlp[-1].weight.data.zero_()
            self.time_mlp[-1].bias.data.zero_()

    def forward(self, x: Tensor, x0: Tensor, timestep_emb: Tensor, timestep_mlp_emb: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, dim)
            x0: Initial residual (for residual mix), shape (batch_size, seq_len, dim)
            timestep_emb: Raw timestep embedding, shape (batch_size, time_embed_dim)
            timestep_mlp_emb: Projected timestep embedding (from DiffusionGPT), shape (batch_size, dim)

        Returns:
            Output tensor, shape (batch_size, seq_len, dim)
        """
        # Residual mix
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.use_simple_timestep:
            # Simple mode: just add timestep embedding, no adaLN
            # timestep_mlp_emb is shape (batch, model_dim), broadcast to (batch, seq, dim)
            x = x + timestep_mlp_emb[:, None, :]

            # Standard attention (no timestep conditioning)
            h_norm = self.attn_norm(x)
            attn_out = self.attn(h_norm)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

            # Standard MLP (no timestep conditioning)
            mlp_norm = self.mlp_norm(x)
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_norm)
        else:
            # Complex mode: adaLN with scale/shift
            t_cond = self.time_mlp(timestep_emb)  # (batch_size, dim * 4)
            t_cond = t_cond[:, None, :]  # (batch_size, 1, dim * 4)

            # Split into scale and shift for attention and MLP
            attn_scale, attn_shift, mlp_scale, mlp_shift = t_cond.chunk(4, dim=-1)

            # Attention with timestep conditioning
            h_norm = self.attn_norm(x)
            h_norm = h_norm * (1 + attn_scale.to(dtype=x.dtype)) + attn_shift.to(dtype=x.dtype)
            attn_out = self.attn(h_norm)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

            # MLP with timestep conditioning
            mlp_norm = self.mlp_norm(x)
            mlp_norm = mlp_norm * (1 + mlp_scale.to(dtype=x.dtype)) + mlp_shift.to(dtype=x.dtype)
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_norm)

        return x


class DiffusionGPT(nn.Module):
    """
    Discrete diffusion language model.

    The model learns to denoise tokens by predicting the original token
    given a noised version and the timestep.

    Forward pass:
        x_t: Noised tokens, shape (batch_size, seq_len)
        t: Timestep, shape (batch_size,)
        x0: Clean target tokens, shape (batch_size, seq_len)

    Returns:
        loss: Cross-entropy loss over all positions
    """

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
        time_embed_dim: int,
        use_causal_attention: bool = False,
        use_simple_timestep: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f'logit_softcap must be positive, got {logit_softcap}')
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.use_simple_timestep = use_simple_timestep

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        # In simple mode, we still need time_mlp for adding to token embeddings
        # But blocks won't have their own adaLN
        self.time_mlp = TimestepMLP(time_embed_dim, model_dim)

        # Transformer blocks with timestep conditioning
        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    time_embed_dim,
                    use_causal_attention,
                    use_simple_timestep,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, '_zero_init', False):
                nn.init.zeros_(module.weight)

    def forward(
        self,
        noised_tokens: Tensor,
        timesteps: Tensor,
        target_tokens: Tensor,
    ) -> Tensor:
        """
        Args:
            noised_tokens: Noised tokens x_t, shape (batch_size, seq_len)
            timesteps: Diffusion timestep t, shape (batch_size,)
            target_tokens: Clean target tokens x_0, shape (batch_size, seq_len)

        Returns:
            loss: Cross-entropy loss
        """
        # Token embeddings
        x = self.tok_emb(noised_tokens)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Timestep embedding
        t_emb = self.time_embed(timesteps)  # (batch_size, time_embed_dim)
        t_emb_for_add = self.time_mlp(t_emb)  # (batch_size, model_dim) - for adding to tokens

        # Combine with token embeddings (only in complex mode)
        if not self.use_simple_timestep:
            x = x + t_emb_for_add[:, None, :]

        # Transformer blocks
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            # In simple mode, pass t_emb_for_add (already projected)
            # In complex mode, pass t_emb (raw, block will project it)
            block_t_emb = t_emb_for_add if self.use_simple_timestep else t_emb
            x = self.blocks[i](x, x0, block_t_emb, t_emb_for_add)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_t_emb = t_emb_for_add if self.use_simple_timestep else t_emb
            x = self.blocks[self.num_encoder_layers + i](x, x0, block_t_emb, t_emb_for_add)

        # Final projection
        x = self.final_norm(x).reshape(-1, x.size(-1))

        # Get logits
        targets = target_tokens.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError('lm_head is required when tie_embeddings=False')
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction='mean')


# -----------------------------
# DATA LOADING (from train_gpt.py)
# -----------------------------


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        'CONTROL_TENSOR_NAME_PATTERNS',
        'attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights',
    ).split(',')
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        'INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',
        ','.join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(',')
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix('torch.')
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ('param_count', 'num_tensors', 'num_float_tensors', 'num_nonfloat_tensors', 'baseline_tensor_bytes', 'int8_payload_bytes'),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to('cpu').contiguous()
        stats['param_count'] += int(t.numel())
        stats['num_tensors'] += 1
        stats['baseline_tensor_bytes'] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats['num_nonfloat_tensors'] += 1
            passthrough[name] = t
            stats['int8_payload_bytes'] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats['int8_payload_bytes'] += tensor_nbytes(kept)
            continue

        stats['num_float_tensors'] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {'scheme': 'per_row', 'axis': 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix('torch.')
        stats['int8_payload_bytes'] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        '__quant_format__': 'int8_clean_per_row_v1',
        'quantized': quantized,
        'scales': scales,
        'dtypes': dtypes,
        'passthrough': passthrough,
    }
    if qmeta:
        obj['qmeta'] = qmeta
    if passthrough_orig_dtypes:
        obj['passthrough_orig_dtypes'] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get('qmeta', {})
    passthrough_orig_dtypes = obj.get('passthrough_orig_dtypes', {})
    for name, q in obj['quantized'].items():
        dtype = getattr(torch, obj['dtypes'][name])
        s = obj['scales'][name]
        if qmeta.get(name, {}).get('scheme') == 'per_row' or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj['passthrough'].items():
        out_t = t.detach().to('cpu').contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype('<i4').itemsize
    token_bytes = np.dtype('<u2').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header for {file}')
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f'Shard size mismatch for {file}: expected {expected_size} bytes')
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f'Short read for {file}')
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    """Reads shards sequentially and wraps around forever."""
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f'No files found for pattern: {pattern}')
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
    """Each call consumes a contiguous chunk from the shared token stream."""
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, _seq_len: int, grad_accum_steps: int) -> Tensor:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        return local.to(self.device, non_blocking=True)


# -----------------------------
# TOKENIZER METRICS (from train_gpt.py)
# -----------------------------


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
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
        if piece.startswith('▁'):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode('utf-8'))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f'Validation split is too short for TRAIN_SEQ_LEN={seq_len}')
    return tokens[: usable + 1]


# -----------------------------
# VALIDATION FUNCTIONS
# -----------------------------


def eval_val_autoregressive(
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
    """
    Compute autoregressive BPB using diffusion model.
    This gives the EXACT same metric as train_gpt.py for fair comparison.

    The key insight: evaluate the diffusion model at t=0 (no noise).
    At t=0, the model should predict clean tokens given clean context,
    which is equivalent to autoregressive prediction.
    """
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            'VAL_BATCH_SIZE must provide at least one sequence per rank; '
            f'got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, '
            f'GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}'
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
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)

            # Create zero timestep (t=0 means no noise)
            t_zero = torch.zeros(x.size(0), dtype=torch.long, device=device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, t_zero, y).detach()

            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
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


def eval_val_diffusion_elbo(
    args: Hyperparameters,
    model: nn.Module,
    noise_schedule: DiscreteNoiseSchedule,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
) -> tuple[float, float, dict[int, float]]:
    """
    Compute diffusion ELBO - the true likelihood bound for diffusion models.

    Evaluates loss at multiple timesteps and computes the weighted average.
    """
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    losses_by_t: dict[int, float] = {}
    total_token_count = 0

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x0 = local.reshape(-1, args.train_seq_len)

            # Evaluate at specific timesteps
            for t_val in [1, args.num_diffusion_steps // 4, args.num_diffusion_steps // 2, args.num_diffusion_steps]:
                t = torch.full((x0.size(0),), t_val, dtype=torch.long, device=device)
                xt, _ = noise_schedule.sample_q(x0, t)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    loss_t = model(xt, t, x0).detach()

                if t_val not in losses_by_t:
                    losses_by_t[t_val] = 0.0
                losses_by_t[t_val] += loss_t.item() * x0.numel()

            total_token_count += x0.numel()

    if dist.is_available() and dist.is_initialized():
        for t_val in losses_by_t:
            t_tensor = torch.tensor(losses_by_t[t_val], device=device)
            dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
            losses_by_t[t_val] = t_tensor.item()
        tc_tensor = torch.tensor(total_token_count, device=device)
        dist.all_reduce(tc_tensor, op=dist.ReduceOp.SUM)
        total_token_count = tc_tensor.item()

    # Average losses
    for t_val in losses_by_t:
        losses_by_t[t_val] /= max(total_token_count, 1)

    # Compute average ELBO loss
    avg_elbo_loss = sum(losses_by_t.values()) / max(len(losses_by_t), 1)

    model.train()
    return avg_elbo_loss, avg_elbo_loss / math.log(2.0), losses_by_t


def eval_val_comprehensive(
    args: Hyperparameters,
    model: nn.Module,
    noise_schedule: DiscreteNoiseSchedule,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> dict[str, float]:
    """
    Compute ALL validation metrics for comprehensive comparison.

    Returns:
        {
            'ar_bpb': float,           # Autoregressive BPB (PRIMARY - same as GPT)
            'ar_loss': float,          # Autoregressive loss
            'diffusion_elbo_bpb': float,  # Diffusion ELBO BPB
            'diffusion_elbo_loss': float,
            'loss_t_1': float,
            'loss_t_250': float,
            'loss_t_500': float,
            'loss_t_1000': float,
        }
    """
    # 1. Autoregressive BPB (for fair comparison with GPT)
    ar_loss, ar_bpb = eval_val_autoregressive(
        args, model, rank, world_size, device,
        grad_accum_steps, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )

    # 2. Diffusion ELBO (showcase diffusion strength)
    diffusion_elbo_loss, diffusion_elbo_bpb, losses_by_t = eval_val_diffusion_elbo(
        args, model, noise_schedule, rank, world_size, device,
        grad_accum_steps, val_tokens,
    )

    result = {
        'ar_bpb': ar_bpb,  # PRIMARY METRIC - directly comparable to GPT
        'ar_loss': ar_loss,
        'diffusion_elbo_bpb': diffusion_elbo_bpb,
        'diffusion_elbo_loss': diffusion_elbo_loss,
    }

    # Add per-timestep losses
    for t_val, loss_t in losses_by_t.items():
        result[f'loss_t_{t_val}'] = loss_t

    return result


# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding='utf-8')
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size <= 0:
        raise ValueError(f'WORLD_SIZE must be positive, got {world_size}')
    if 8 % world_size != 0:
        raise ValueError(f'WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral')
    grad_accum_steps = 8 // world_size // GRAD_ACC_DIV
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
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
        os.makedirs('logs', exist_ok=True)
        logfile = f'logs/{args.run_id}.txt'
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, 'a', encoding='utf-8') as f:
                print(msg, file=f)

    log0(code, console=False)
    log0('=' * 100, console=False)
    log0(f'Running Python {sys.version}', console=False)
    log0(f'Running PyTorch {torch.__version__}', console=False)
    log0(
        subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False).stdout,
        console=False,
    )
    log0('=' * 100, console=False)

    if master_process and args.wandb_enabled:
        wandb.init(project=args.wandb_project, name=args.run_id, config=vars(args), save_code=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith('.model'):
        raise ValueError(f'Script only setup for SentencePiece .model file: {args.tokenizer_path}')
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f'VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}')
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob('fineweb_train_*.bin')))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f'val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}')
    log0(f'train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}')
    log0(f'val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}')

    # -----------------------------
    # NOISE SCHEDULE SETUP
    # -----------------------------

    noise_schedule = DiscreteNoiseSchedule(
        num_steps=args.num_diffusion_steps,
        vocab_size=args.vocab_size,
        noise_type=args.noise_type,
        noise_max_prob=args.noise_max_prob,
        device=device,
    )
    log0(f'noise_schedule: type={args.noise_type} steps={args.num_diffusion_steps} max_prob={args.noise_max_prob}')

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = (
        DiffusionGPT(
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
            time_embed_dim=args.time_embed_dim,
            use_causal_attention=args.use_causal_attention,
            use_simple_timestep=args.use_simple_timestep,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    # Skip torch.compile for mixed AR/diffusion training (dtype/shapes cause issues)
    # model: nn.Module = DDP(torch.compile(base_model), ...) if distributed else torch.compile(base_model)
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    # Optimizer setup (similar to train_gpt.py)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]

    # Add timestep embedding parameters
    if hasattr(base_model, 'time_embed'):
        for _name, p in base_model.time_embed.named_parameters():
            scalar_params.append(p)
    if hasattr(base_model, 'time_mlp'):
        for _name, p in base_model.time_mlp.named_parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                scalar_params.append(p)

    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{'params': [base_model.tok_emb.weight], 'lr': token_lr, 'base_lr': token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group['base_lr'] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{'params': scalar_params, 'lr': args.scalar_lr, 'base_lr': args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{'params': [base_model.lm_head.weight], 'lr': args.head_lr, 'base_lr': args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f'model_params:{n_params}')
    log0(f'world_size:{world_size} grad_accum_steps:{grad_accum_steps}')
    log0('sdp_backends:cudnn=False flash=True mem_efficient=False math=False')
    log0(f'attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} causal:{args.use_causal_attention}')
    log0(
        f'tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} '
        f'head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} '
        f'matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}'
    )
    log0(
        f'train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} '
        f'iterations:{args.iterations} warmup_steps:{args.warmup_steps} '
        f'max_wallclock_seconds:{args.max_wallclock_seconds:.3f}'
    )
    log0(f'diffusion_steps:{args.num_diffusion_steps} noise_type:{args.noise_type}')
    log0(f'causal_attention:{args.use_causal_attention} ar_train_fraction:{args.ar_train_fraction} simple_timestep:{args.use_simple_timestep}')
    log0(f'seed:{args.seed}')

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                x0 = x0.reshape(-1, args.train_seq_len)

                use_ar = torch.rand(1).item() < args.ar_train_fraction

                if use_ar and args.use_causal_attention:
                    x_ar = x0[:, :-1].contiguous()
                    y_ar = x0[:, 1:].contiguous()
                    t_ar = torch.zeros(x_ar.size(0), dtype=torch.long, device=device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(x_ar, t_ar, y_ar)
                else:
                    t = torch.randint(
                        args.min_train_timestep,
                        args.num_diffusion_steps + 1,
                        (x0.size(0),),
                        device=device,
                    )
                    xt, _ = noise_schedule.sample_q(x0, t)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(xt, t, x0)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f'warmup_step:{warmup_step + 1}/{args.warmup_steps}')
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            metrics = eval_val_comprehensive(
                args,
                model,
                noise_schedule,
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
                f'step:{step}/{args.iterations} '
                f'ar_bpb:{metrics["ar_bpb"]:.4f} '  # Main comparison metric
                f'diff_bpb:{metrics["diffusion_elbo_bpb"]:.4f} '
                f'train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms'
            )
            if master_process and args.wandb_enabled:
                wandb.log({
                    'step': step,
                    'ar_bpb': metrics['ar_bpb'],
                    'ar_loss': metrics['ar_loss'],
                    'diffusion_elbo_bpb': metrics['diffusion_elbo_bpb'],
                    'diffusion_elbo_loss': metrics['diffusion_elbo_loss'],
                    'train_time_ms': training_time_ms,
                    **{f'loss_t_{k}': v for k, v in metrics.items() if k.startswith('loss_t_')},
                }, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f'stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}')
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            x0 = x0.reshape(-1, args.train_seq_len)

            # Decide: AR training or diffusion training?
            # AR training trains next-token prediction for valid BPB eval
            # Diffusion training trains denoising for the diffusion objective
            use_ar = torch.rand(1).item() < args.ar_train_fraction

            if use_ar and args.use_causal_attention:
                # AR training: predict next token (same as GPT)
                # x[:-1] is context, x[1:] is target
                x_ar = x0[:, :-1].contiguous()  # context: positions 0 to T-2
                y_ar = x0[:, 1:].contiguous()   # targets: positions 1 to T-1
                # Both have shape (batch, seq_len-1), which matches for the loss
                t_ar = torch.zeros(x_ar.size(0), dtype=torch.long, device=device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    loss = model(x_ar, t_ar, y_ar)  # Train on next-token prediction
            else:
                # Diffusion training: denoise x_t -> x_0
                t = torch.randint(
                    args.min_train_timestep,
                    args.num_diffusion_steps + 1,
                    (x0.size(0),),
                    device=device,
                )
                xt, _ = noise_schedule.sample_q(x0, t)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    loss = model(xt, t, x0)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group['momentum'] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(
                f'step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} '
                f'train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms'
            )
            if master_process and args.wandb_enabled:
                wandb.log({'step': step, 'train_loss': train_loss.item(), 'lr_mul': scale}, step=step)

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f'peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB '
        f'reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB'
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        torch.save(base_model.state_dict(), 'final_model.pt')
        model_bytes = os.path.getsize('final_model.pt')
        code_bytes = len(code.encode('utf-8'))
        log0(f'Serialized model: {model_bytes} bytes')
        log0(f'Code size: {code_bytes} bytes')
        log0(f'Total submission size: {model_bytes + code_bytes} bytes')

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open('final_model.int8.ptz', 'wb') as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize('final_model.int8.ptz')
        code_bytes = len(code.encode('utf-8'))
        ratio = quant_stats['baseline_tensor_bytes'] / max(quant_stats['int8_payload_bytes'], 1)
        log0(
            f'Serialized model int8+zlib: {quant_file_bytes} bytes '
            f'(payload:{quant_stats["int8_payload_bytes"]} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)'
        )
        log0(f'Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes')

    if distributed:
        dist.barrier()
    with open('final_model.int8.ptz', 'rb') as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location='cpu')
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_metrics = eval_val_comprehensive(
        args,
        model,
        noise_schedule,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f'final_int8_zlib_roundtrip ar_bpb:{q_metrics["ar_bpb"]:.4f} '
        f'diff_bpb:{q_metrics["diffusion_elbo_bpb"]:.4f} '
        f'eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms'
    )
    log0(f'final_int8_zlib_roundtrip_exact ar_bpb:{q_metrics["ar_bpb"]:.8f} diff_bpb:{q_metrics["diffusion_elbo_bpb"]:.8f}')

    if master_process and args.wandb_enabled:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
