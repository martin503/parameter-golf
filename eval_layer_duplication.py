"""
Evaluate a trained model with layer duplication patterns.

Usage:
    # Single pattern evaluation
    EVAL_LAYER_PATTERN="1,2,3,2,3,4,5,6,7" python eval_layer_duplication.py

    # Auto-search mode (empty EVAL_LAYER_PATTERN triggers search)
    MAX_PARAMS=16_000_000 python eval_layer_duplication.py

    # Custom model path
    MODEL_PATH=my_model.pt MAX_PARAMS=20_000_000 python eval_layer_duplication.py

Patterns use 1-indexed layer indices (1 = first layer).
"""

from __future__ import annotations

import csv
import io
import math
import os
import sys
import time
import zlib
from pathlib import Path

# Increase recompile limit for varying batch sizes during eval
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256

import sentencepiece as spm
import torch
from torch import Tensor

# Add current dir to path to import from train_gpt
sys.path.insert(0, str(Path(__file__).parent))

from train_gpt import (
    GPT,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    load_validation_tokens,
)

# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    # Model and data paths
    model_path = os.environ.get('MODEL_PATH', './final_model.pt')
    csv_path = os.environ.get('CSV_PATH', None)  # If None, auto-generate next to model
    tokenizer_path = os.environ.get('TOKENIZER_PATH', './data/tokenizers/fineweb_1024_bpe.model')

    # Evaluation config
    val_files = os.environ.get('VAL_FILES', './data/datasets/fineweb10B_sp1024/fineweb_val_*.bin')
    val_batch_size = int(os.environ.get('VAL_BATCH_SIZE', 524_288))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 1024))

    # Layer duplication config
    eval_layer_pattern = os.environ.get('EVAL_LAYER_PATTERN', '')  # Empty = auto-search
    max_params = int(os.environ.get('MAX_PARAMS', '16_000_000').replace('_', ''))
    search_type = os.environ.get('SEARCH_TYPE', 'any')  # 'interval' or 'any'
    num_dup_layers = int(os.environ.get('NUM_DUP_LAYERS', '2'))  # For 'any' search: how many layers to duplicate


# -----------------------------
# MODEL CONFIG FROM CHECKPOINT
# -----------------------------


def deduce_model_config(state_dict: dict) -> dict:
    """Deduce model configuration from state_dict."""
    cfg = {}

    # vocab_size and model_dim from tok_emb
    cfg['vocab_size'] = state_dict['tok_emb.weight'].shape[0]
    cfg['model_dim'] = state_dict['tok_emb.weight'].shape[1]

    # num_layers: extract max block index from state_dict keys
    block_indices = set()
    for k in state_dict.keys():
        if k.startswith('blocks.'):
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    cfg['num_layers'] = max(block_indices) + 1 if block_indices else 0

    # num_heads from q_gain shape (first block)
    cfg['num_heads'] = state_dict['blocks.0.attn.q_gain'].shape[0]

    # head_dim
    cfg['head_dim'] = cfg['model_dim'] // cfg['num_heads']

    # num_kv_heads from c_k weight shape
    # c_k weight is (kv_dim, model_dim) where kv_dim = num_kv_heads * head_dim
    kv_dim = state_dict['blocks.0.attn.c_k.weight'].shape[0]
    cfg['num_kv_heads'] = kv_dim // cfg['head_dim']

    # mlp_mult from mlp.fc weight shape
    # fc weight is (hidden, model_dim) where hidden = mlp_mult * model_dim
    hidden = state_dict['blocks.0.mlp.fc.weight'].shape[0]
    cfg['mlp_mult'] = hidden // cfg['model_dim']

    # tie_embeddings: lm_head exists only if not tied
    cfg['tie_embeddings'] = 'lm_head.weight' not in state_dict

    # logit_softcap: not stored in state_dict, use default
    cfg['logit_softcap'] = 30.0

    # rope_base: inv_freq is not persistent, so we can't recover it
    # Using default value
    cfg['rope_base'] = 10000.0

    # qk_gain_init: read from first q_gain value
    cfg['qk_gain_init'] = float(state_dict['blocks.0.attn.q_gain'][0].item())

    # tied_embed_init_std: not recoverible from weights, use default
    cfg['tied_embed_init_std'] = 0.005

    return cfg


# -----------------------------
# LAYER PATTERN GENERATION
# -----------------------------


def parse_layer_pattern(pattern: str, n_layers: int) -> list[int] | None:
    """Parse layer pattern string into list of 0-indexed layer indices."""
    if not pattern or not pattern.strip():
        return None
    indices = [int(x.strip()) for x in pattern.split(',') if x.strip()]
    zero_indexed = [i - 1 for i in indices]  # Convert to 0-indexed
    if any(i < 0 or i >= n_layers for i in zero_indexed):
        raise ValueError(f"Layer indices must be between 1 and {n_layers}")
    return zero_indexed


def generate_interval_duplication_patterns(
    n_layers: int,
    block_params: float,
    total_params: float,
    max_params: int,
) -> list[tuple[str, list[int]]]:
    """
    Generate interval duplication patterns (contiguous block repeated once).

    Rules:
    - First and last layers are never touched
    - A contiguous block from middle layers is repeated once
    - Total params must not exceed max_params

    Args:
        n_layers: Number of trained layers
        block_params: Parameters per transformer block
        total_params: Total parameters in base model
        max_params: Maximum allowed parameters

    Returns:
        List of (pattern_string, zero_indexed_layer_list) tuples
    """
    patterns = []

    # Calculate how many extra layers we can add
    params_budget = max_params - total_params
    max_extra_layers = int(params_budget / block_params)

    if max_extra_layers < 1:
        # Can't add any layers, return only base pattern
        base_pattern = list(range(n_layers))
        patterns.append((",".join(str(i + 1) for i in base_pattern), base_pattern))
        return patterns

    # Only test the maximum block size that fits
    max_block_size = min(n_layers - 2, max_extra_layers)

    block_size = max_block_size
    # For each starting position of the block in middle layers
    # 0-indexed: middle layers are 1 to n_layers-2 (since 0 is first, n_layers-1 is last)
    for start in range(1, n_layers - 1):
        end = start + block_size
        if end > n_layers - 1:
            break  # Block would extend past last layer

        # Pattern: [0, 1, ..., start-1, start...end-1, start...end-1, end, ..., n_layers-1]
        pattern = (
            list(range(0, start)) +           # before block (including first layer)
            list(range(start, end)) +          # block first occurrence
            list(range(start, end)) +          # block repeated
            list(range(end, n_layers))         # after block (including last layer)
        )
        pattern_str = ",".join(str(i + 1) for i in pattern)
        patterns.append((pattern_str, pattern))

    return patterns


def generate_any_duplication_patterns(
    n_layers: int,
    num_dup: int = 2,
) -> list[tuple[str, list[int]]]:
    """
    Generate 'any N layers' duplication patterns.

    Rules:
    - Any N layers can be duplicated right after themselves (including first and last)
    - Duplicates exactly num_dup layers

    Args:
        n_layers: Number of trained layers
        num_dup: Number of layers to duplicate

    Returns:
        List of (pattern_string, zero_indexed_layer_list) tuples
    """
    patterns = []

    # All layers can be duplicated (0-indexed: 0 to n_layers-1)
    all_layers = list(range(n_layers))

    # Generate all combinations of num_dup layers to duplicate
    from itertools import combinations
    for dup_set in combinations(all_layers, num_dup):
        # Build pattern: for each layer, include it twice if it's in dup_set, once otherwise
        pattern = []
        for i in range(n_layers):
            pattern.append(i)
            if i in dup_set:
                pattern.append(i)

        pattern_str = ",".join(str(i + 1) for i in pattern)
        patterns.append((pattern_str, pattern))

    return patterns


# -----------------------------
# MODEL WRAPPER
# -----------------------------


class GPTWithLayerDuplication(torch.nn.Module):
    """GPT wrapper that evaluates with layer duplication."""
    def __init__(self, base_model: GPT, layer_pattern: list[int] | None):
        super().__init__()
        self.base_model = base_model
        self.layer_pattern = layer_pattern if layer_pattern else list(range(len(base_model.blocks)))
        self.eval_layers = len(self.layer_pattern)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.base_model.tok_emb(input_ids)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))
        x0 = x

        blocks_to_use = [self.base_model.blocks[i] for i in self.layer_pattern]

        # Check if using normal skip connection mode
        n = len(self.base_model.blocks)
        has_skip = (self.base_model.num_skip_weights > 0 and
                   self.layer_pattern == list(range(n)))

        if has_skip:
            # Standard mode with skip connections
            skips = []
            for i in range(self.base_model.num_encoder_layers):
                x = self.base_model.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.base_model.num_decoder_layers):
                if skips:
                    x = x + self.base_model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.base_model.blocks[self.base_model.num_encoder_layers + i](x, x0)
        else:
            # Layer duplication mode - just apply blocks in pattern order
            for block in blocks_to_use:
                x = block(x, x0)

        x = self.base_model.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.base_model.tie_embeddings:
            logits_proj = torch.nn.functional.linear(x, self.base_model.tok_emb.weight)
        else:
            logits_proj = self.base_model.lm_head(x)
        logits = self.base_model.logit_softcap * torch.tanh(logits_proj / self.base_model.logit_softcap)
        return torch.nn.functional.cross_entropy(logits.float(), targets, reduction='mean')


# -----------------------------
# EVALUATION
# -----------------------------


def eval_val(
    model: torch.nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    batch_size: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    local_batch_seqs = batch_size // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(0, total_seqs, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()

            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()

    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# MAIN
# -----------------------------


def main() -> None:
    args = Hyperparameters()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from {args.model_path}...')
    if args.model_path.endswith('.ptz'):
        with open(args.model_path, 'rb') as f:
            quant_blob = f.read()
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location='cpu')
        state_dict = dequantize_state_dict_int8(quant_state)
    else:
        state_dict = torch.load(args.model_path, map_location='cpu')
    print('Model loaded.')

    # Deduce model config from checkpoint
    print('Deducing model configuration from checkpoint...')
    cfg = deduce_model_config(state_dict)
    print(f'  vocab_size: {cfg["vocab_size"]}')
    print(f'  num_layers: {cfg["num_layers"]}')
    print(f'  model_dim: {cfg["model_dim"]}')
    print(f'  num_heads: {cfg["num_heads"]}')
    print(f'  num_kv_heads: {cfg["num_kv_heads"]}')
    print(f'  mlp_mult: {cfg["mlp_mult"]}')
    print(f'  tie_embeddings: {cfg["tie_embeddings"]}')

    # Create base model
    print('Creating model...')
    base_model = GPT(
        vocab_size=cfg['vocab_size'],
        num_layers=cfg['num_layers'],
        model_dim=cfg['model_dim'],
        num_heads=cfg['num_heads'],
        num_kv_heads=cfg['num_kv_heads'],
        mlp_mult=cfg['mlp_mult'],
        tie_embeddings=cfg['tie_embeddings'],
        tied_embed_init_std=cfg['tied_embed_init_std'],
        logit_softcap=cfg['logit_softcap'],
        rope_base=cfg['rope_base'],
        qk_gain_init=cfg['qk_gain_init'],
    ).to(device).bfloat16()

    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    block_params = sum(p.numel() for p in base_model.blocks[0].parameters())
    print(f'Base model: {total_params / 1e6:.2f}M params ({cfg["num_layers"]} layers)')
    print(f'Per block: {block_params / 1e6:.2f}M params')
    print(f'Max allowed: {args.max_params / 1e6:.2f}M params')

    # Generate patterns to test
    auto_search = not bool(args.eval_layer_pattern)
    if auto_search:
        print('\nGenerating patterns to test...')
        if args.search_type == 'any':
            patterns_to_test = generate_any_duplication_patterns(cfg['num_layers'], args.num_dup_layers)
        else:  # 'interval'
            patterns_to_test = generate_interval_duplication_patterns(
                cfg['num_layers'], block_params, total_params, args.max_params
            )

        # Random sample if too many patterns
        import random
        if len(patterns_to_test) > 200:
            print(f'Sampling 200 patterns from {len(patterns_to_test)} total...')
            random.seed(42)  # Fixed seed for reproducibility
            patterns_to_test = random.sample(patterns_to_test, 200)

        print(f'Found {len(patterns_to_test)} patterns to test (search_type={args.search_type})')
    else:
        layer_pattern = parse_layer_pattern(args.eval_layer_pattern, cfg['num_layers'])
        if layer_pattern:
            patterns_to_test = [(args.eval_layer_pattern, layer_pattern)]
        else:
            patterns_to_test = [(",".join(str(i + 1) for i in range(cfg['num_layers'])), list(range(cfg['num_layers'])))]

    # Load validation data
    print('\nLoading validation data...')
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, cfg['vocab_size'], device)
    print(f'Loaded {val_tokens.numel()} validation tokens')

    # Results storage
    results = []

    # Evaluate each pattern
    for i, (pat_str, layer_pattern) in enumerate(patterns_to_test):
        print(f'\n[{i+1}/{len(patterns_to_test)}] Testing pattern: {pat_str}')
        print(f'  Layers: {len(layer_pattern)} (trained: {cfg["num_layers"]})')

        # Create wrapped model
        model = GPTWithLayerDuplication(base_model, layer_pattern).to(device)

        # Compile (without fullgraph to avoid recompile issues)
        model = torch.compile(model, dynamic=False)

        # Evaluate
        t0 = time.perf_counter()
        val_loss, val_bpb = eval_val(
            model, val_tokens, args.train_seq_len,
            args.val_batch_size, base_bytes_lut, has_leading_space_lut,
            is_boundary_token_lut, device,
        )
        eval_time = time.perf_counter() - t0

        # Calculate effective params
        effective_layers = len(layer_pattern)
        effective_params = total_params - (cfg['num_layers'] * block_params) + (effective_layers * block_params)

        print(f'  val_loss: {val_loss:.6f}, val_bpb: {val_bpb:.6f}, time: {eval_time:.1f}s')
        print(f'  effective params: {effective_params / 1e6:.2f}M')

        results.append({
            'pattern': pat_str,
            'layers': effective_layers,
            'val_loss': val_loss,
            'val_bpb': val_bpb,
            'time_seconds': eval_time,
            'effective_params_m': effective_params / 1e6,
        })

        # Free compiled model to save memory
        del model
        torch.cuda.empty_cache()

    # Output results
    print('\n' + '=' * 80)
    if auto_search:
        # Sort by val_bpb ascending
        results.sort(key=lambda x: x['val_bpb'])
        print('Results (sorted by val_bpb):')
        print('-' * 80)
        for r in results[:10]:  # Show top 10
            print(f"{r['pattern']:40s} | L:{r['layers']:2d} | P:{r['effective_params_m']:6.2f}M | bpb:{r['val_bpb']:.6f}")
        if len(results) > 10:
            print(f'... and {len(results) - 10} more')

        # Determine CSV path
        if args.csv_path:
            csv_path = args.csv_path
        else:
            model_path_obj = Path(args.model_path)
            csv_path = str(model_path_obj.with_suffix('.layer_dup_results.csv'))

        # Save to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pattern', 'layers', 'val_loss', 'val_bpb', 'time_seconds', 'effective_params_m'])
            writer.writeheader()
            writer.writerows(results)
        print(f'\nResults saved to: {csv_path}')
    else:
        r = results[0]
        print(f'Pattern: {r["pattern"]}')
        print(f'  val_loss: {r["val_loss"]:.6f}')
        print(f'  val_bpb:  {r["val_bpb"]:.6f}')
        print(f'  time: {r["time_seconds"]:.1f}s')
    print('=' * 80)


if __name__ == '__main__':
    main()
