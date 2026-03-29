"""
Evaluate a trained model with symmetric encoder-decoder layer duplication.

By default, the pattern is specified using encoder layer indices (1-indexed), and
the same duplication pattern is applied to both encoder and decoder layers.

With DECODER_ONLY=1, only the decoder layers are duplicated.

Example with 8 total layers (4 encoder + 4 decoder):
    Pattern "1,2,2,3,4" (default) means:
    - Encoder: 1,2,2,3,4
    - Decoder: 5,6,6,7,8
    - Full: 1,2,2,3,4,5,6,6,7,8

    Pattern "1,2,2,3,4" with DECODER_ONLY=1 means:
    - Encoder: 1,2,3,4 (unchanged)
    - Decoder: 5,6,6,7,8 (duplicated)
    - Full: 1,2,3,4,5,6,6,7,8

Usage:
    # Single pattern evaluation (symmetric enc+dec)
    EVAL_ENC_PATTERN="1,2,2,3,4" python eval_enc_dec_duplication.py

    # Single pattern evaluation (decoder only)
    EVAL_ENC_PATTERN="1,2,2,3,4" DECODER_ONLY=1 python eval_enc_dec_duplication.py

    # Auto-search mode (empty EVAL_ENC_PATTERN triggers search)
    # Searches all duplication patterns within MAX_PARAMS budget
    MAX_PARAMS=16_000_000 python eval_enc_dec_duplication.py

    # Auto-search with decoder-only mode
    MAX_PARAMS=16_000_000 DECODER_ONLY=1 python eval_enc_dec_duplication.py
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
from itertools import combinations

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

    # Encoder duplication config
    eval_enc_pattern = os.environ.get('EVAL_ENC_PATTERN', '')  # Empty = auto-search
    max_params = int(os.environ.get('MAX_PARAMS', '16_000_000').replace('_', ''))
    decoder_only = os.environ.get('DECODER_ONLY', '').lower() in ('1', 'true', 'yes')  # Only duplicate decoder


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
    kv_dim = state_dict['blocks.0.attn.c_k.weight'].shape[0]
    cfg['num_kv_heads'] = kv_dim // cfg['head_dim']

    # mlp_mult from mlp.fc weight shape
    hidden = state_dict['blocks.0.mlp.fc.weight'].shape[0]
    cfg['mlp_mult'] = hidden // cfg['model_dim']

    # tie_embeddings: lm_head exists only if not tied
    cfg['tie_embeddings'] = 'lm_head.weight' not in state_dict

    # logit_softcap: not stored in state_dict, use default
    cfg['logit_softcap'] = 30.0

    # rope_base: using default value
    cfg['rope_base'] = 10000.0

    # qk_gain_init: read from first q_gain value
    cfg['qk_gain_init'] = float(state_dict['blocks.0.attn.q_gain'][0].item())

    # tied_embed_init_std: not recoverible from weights, use default
    cfg['tied_embed_init_std'] = 0.005

    return cfg


# -----------------------------
# LAYER PATTERN GENERATION
# -----------------------------


def parse_pattern(pattern: str, n_layers: int, layer_type: str = "encoder") -> list[int] | None:
    """
    Parse layer pattern string into list of 0-indexed layer indices.

    Args:
        pattern: Comma-separated string of 1-indexed layer indices
        n_layers: Number of layers
        layer_type: "encoder" or "decoder" (for error messages)

    Returns:
        List of 0-indexed layer indices, or None for default pattern
    """
    if not pattern or not pattern.strip():
        return None
    indices = [int(x.strip()) for x in pattern.split(',') if x.strip()]
    zero_indexed = [i - 1 for i in indices]  # Convert to 0-indexed
    if any(i < 0 or i >= n_layers for i in zero_indexed):
        raise ValueError(f"{layer_type.capitalize()} layer indices must be between 1 and {n_layers}")
    return zero_indexed


def generate_duplication_patterns(
    n_layers: int,
    max_dup: int,
) -> list[tuple[str, list[int]]]:
    """
    Generate duplication patterns for encoder or decoder layers.

    Rules:
    - Any N layers can be duplicated right after themselves
    - Generate all patterns with 1, 2, ..., max_dup duplications

    Args:
        n_layers: Number of layers (encoder or decoder)
        max_dup: Maximum number of layers to duplicate

    Returns:
        List of (pattern_string, zero_indexed_layer_list) tuples
    """
    patterns = []

    # All layers can be duplicated (0-indexed: 0 to n_layers-1)
    all_layers = list(range(n_layers))

    # Generate all patterns with 1 to max_dup duplications
    for num_dup in range(1, max_dup + 1):
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


class GPTWithEncDecDuplication(torch.nn.Module):
    """GPT wrapper that evaluates with symmetric encoder-decoder layer duplication."""
    def __init__(self, base_model: GPT, enc_pattern: list[int] | None):
        super().__init__()
        self.base_model = base_model
        self.num_enc_layers = base_model.num_encoder_layers
        self.num_dec_layers = base_model.num_decoder_layers

        # Default pattern: use all encoder layers as-is
        if enc_pattern is None:
            self.enc_pattern = list(range(self.num_enc_layers))
        else:
            self.enc_pattern = enc_pattern

        # Build decoder pattern by adding offset to encoder pattern
        self.dec_pattern = [i + self.num_enc_layers for i in self.enc_pattern]

        # Full layer pattern for the model
        self.layer_pattern = self.enc_pattern + self.dec_pattern

        self.eval_enc_layers = len(self.enc_pattern)
        self.eval_dec_layers = len(self.dec_pattern)
        self.eval_total_layers = len(self.layer_pattern)

        # Set skip weights count before building mapping
        self.num_skip_weights = base_model.num_skip_weights
        # Build skip connection mapping
        # Maps decoder position -> skip weight index to use
        self.dec_to_skip_idx = self._build_skip_mapping()

    def _build_skip_mapping(self) -> list[int]:
        """
        Build mapping from encoder pattern position to skip weight index.

        Original skip connection pattern (from train_gpt.py):
        - skips.append(encoder[i]) for i in range(num_enc_layers)
        - For decoder at position i: skip_weights[i] * skips.pop()
        - Since skips.pop() returns in reverse order:
          * decoder[0] gets skip_weights[0] * enc[last]
          * decoder[1] gets skip_weights[1] * enc[last-1]
          * ...
          * decoder[last] gets skip_weights[last] * enc[0]

        So the skip weight index is the decoder position i, and the
        source encoder is enc[num_enc_layers - 1 - i].

        With duplication, if decoder is at position i (0-indexed from first decoder),
        we want to use skip_weights[min(i, num_skip_weights-1)] and connect to
        the "i-th from last" encoder (which may be duplicated).

        We map: decoder_position → which encoder position (in pattern) → skip weight index
        """
        n_enc = len(self.enc_pattern)
        n_dec = len(self.dec_pattern)
        n_skip = self.num_skip_weights

        mapping = []
        for dec_pos in range(n_dec):
            # Which encoder position does this decoder connect to?
            # Original: enc[n_enc_layers - 1 - dec_pos]
            # With pattern: same relative position from the end
            enc_pos = n_enc - 1 - dec_pos
            if enc_pos < 0:
                enc_pos = 0  # If decoder is longer, connect to first encoder

            # Now, which skip weight index for this encoder position?
            # The skip weight index is based on the order from the end:
            # enc[last] uses skip_weights[0], enc[last-1] uses skip_weights[1], etc.
            # So: skip_idx = (n_enc - 1) - enc_pos
            skip_idx = (n_enc - 1) - enc_pos
            # But we also need to cap at num_skip_weights - 1
            skip_idx = min(skip_idx, n_skip - 1)
            mapping.append(skip_idx)

        return mapping

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.base_model.tok_emb(input_ids)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))
        x0 = x

        # Get encoder blocks
        enc_blocks = [self.base_model.blocks[i] for i in self.enc_pattern]
        # Get decoder blocks
        dec_blocks = [self.base_model.blocks[i] for i in self.dec_pattern]

        # Encoder pass - collect skip connections
        skips = []
        for block in enc_blocks:
            x = block(x, x0)
            skips.append(x)

        # Decoder pass - use skip connections
        # Decoder position i connects to encoder output from position (len(enc) - 1 - i)
        # using skip weight determined by the mapping
        for dec_pos, block in enumerate(dec_blocks):
            if self.num_skip_weights > 0:
                # Which encoder position does this decoder connect to?
                enc_pos = len(skips) - 1 - dec_pos
                if enc_pos >= 0:
                    skip_idx = self.dec_to_skip_idx[dec_pos] if dec_pos < len(self.dec_to_skip_idx) else dec_pos
                    skip_idx = min(skip_idx, self.num_skip_weights - 1)
                    skip_weight = self.base_model.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                    x = x + skip_weight * skips[enc_pos]

            x = block(x, x0)

        x = self.base_model.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.base_model.tie_embeddings:
            logits_proj = torch.nn.functional.linear(x, self.base_model.tok_emb.weight)
        else:
            logits_proj = self.base_model.lm_head(x)
        logits = self.base_model.logit_softcap * torch.tanh(logits_proj / self.base_model.logit_softcap)
        return torch.nn.functional.cross_entropy(logits.float(), targets, reduction='mean')


class GPTWithDecoderDuplication(torch.nn.Module):
    """GPT wrapper that duplicates only decoder layers, keeping encoder unchanged.

    The pattern is specified using decoder layer indices (1-indexed), where
    1 refers to the first decoder layer.
    """
    def __init__(self, base_model: GPT, dec_pattern: list[int] | None):
        super().__init__()
        self.base_model = base_model
        self.num_enc_layers = base_model.num_encoder_layers
        self.num_dec_layers = base_model.num_decoder_layers

        # Encoder is always unchanged
        self.enc_pattern = list(range(self.num_enc_layers))

        # Default pattern: use all decoder layers as-is
        if dec_pattern is None:
            self.dec_pattern_rel = list(range(self.num_dec_layers))
        else:
            self.dec_pattern_rel = dec_pattern  # 0-indexed relative to decoder

        # Convert relative decoder indices to absolute block indices
        # Decoder starts at index num_enc_layers in the blocks list
        self.dec_pattern = [i + self.num_enc_layers for i in self.dec_pattern_rel]

        # Full layer pattern for the model
        self.layer_pattern = self.enc_pattern + self.dec_pattern

        self.eval_enc_layers = len(self.enc_pattern)
        self.eval_dec_layers = len(self.dec_pattern)
        self.eval_total_layers = len(self.layer_pattern)

        # Set skip weights count before building mapping
        self.num_skip_weights = base_model.num_skip_weights
        # Build skip connection mapping
        # Maps decoder position -> skip weight index to use
        self.dec_to_skip_idx = self._build_skip_mapping()

    def _build_skip_mapping(self) -> list[int]:
        """
        Build mapping from decoder position to skip weight index.

        In decoder-only mode, encoder is unchanged but decoder may have more layers.
        We need to map each decoder position to the appropriate encoder skip.

        Original skip pattern: decoder[i] connects to enc[num_enc_layers - 1 - i]
        via skip_weights[i].

        With duplicated decoder layers, we need to figure out which "original"
        decoder position each duplicated decoder corresponds to.
        """
        n_dec = len(self.dec_pattern_rel)
        n_skip = self.num_skip_weights

        mapping = []
        for dec_pos in range(n_dec):
            # Which original decoder layer does this position correspond to?
            # Since we're duplicating decoder layers, the pattern might have duplicates
            # We need to map to the unique original decoder index
            dec_layer_idx = self.dec_pattern_rel[dec_pos]

            # The skip weight index is based on the original decoder layer index
            # Original: decoder[i] uses skip_weights[i] (with i from 0 to num_dec_layers-1)
            # But the connection is to enc[num_enc_layers - 1 - i]

            # So the skip weight index is the decoder layer index (capped at available skips)
            skip_idx = min(dec_layer_idx, n_skip - 1)
            mapping.append(skip_idx)

        return mapping

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.base_model.tok_emb(input_ids)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))
        x0 = x

        # Get encoder blocks (unchanged)
        enc_blocks = [self.base_model.blocks[i] for i in self.enc_pattern]
        # Get decoder blocks (possibly duplicated)
        dec_blocks = [self.base_model.blocks[i] for i in self.dec_pattern]

        # Encoder pass - collect skip connections
        skips = []
        for block in enc_blocks:
            x = block(x, x0)
            skips.append(x)

        # Decoder pass - use skip connections
        # Map decoder position to encoder skip (from the end, in reverse order)
        for dec_pos, block in enumerate(dec_blocks):
            if self.num_skip_weights > 0:
                # Which encoder layer does this decoder connect to?
                # Original: decoder[i] connects to enc[num_enc_layers - 1 - i]
                # But here, dec_layer_idx tells us which "original" decoder this is
                dec_layer_idx = self.dec_pattern_rel[dec_pos]
                enc_pos = self.num_enc_layers - 1 - dec_layer_idx
                if enc_pos >= 0:
                    skip_idx = self.dec_to_skip_idx[dec_pos] if dec_pos < len(self.dec_to_skip_idx) else dec_pos
                    skip_idx = min(skip_idx, self.num_skip_weights - 1)
                    skip_weight = self.base_model.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                    x = x + skip_weight * skips[enc_pos]

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
        checkpoint = torch.load(args.model_path, map_location='cpu')
        # Handle train_solar.py stage1 checkpoint format (nested 'model' key)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
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

    # Get encoder/decoder layer counts
    n_enc_layers = base_model.num_encoder_layers
    n_dec_layers = base_model.num_decoder_layers
    decoder_only = args.decoder_only
    print(f'Architecture: {n_enc_layers} encoder + {n_dec_layers} decoder layers')
    print(f'Mode: {"Decoder-only duplication" if decoder_only else "Symmetric (enc+dec) duplication"}')

    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    block_params = sum(p.numel() for p in base_model.blocks[0].parameters())
    print(f'Base model: {total_params / 1e6:.2f}M params ({cfg["num_layers"]} layers)')
    print(f'Per block: {block_params / 1e6:.2f}M params')
    print(f'Max allowed: {args.max_params / 1e6:.2f}M params')

    # Generate patterns to test
    auto_search = not bool(args.eval_enc_pattern)

    if auto_search:
        # Calculate max duplications based on parameter budget
        params_budget = args.max_params - total_params

        if decoder_only:
            # Each duplicated decoder layer adds 1 block
            max_extra_blocks = int(params_budget / block_params)
            max_dup_layers = min(max_extra_blocks, n_dec_layers)
            layer_type = "decoder"
            n_layers = n_dec_layers
        else:
            # Each duplicated encoder layer adds 2 blocks (1 enc + 1 dec)
            max_extra_blocks = int(params_budget / (2 * block_params))
            max_dup_layers = min(max_extra_blocks, n_enc_layers)
            layer_type = "encoder"
            n_layers = n_enc_layers

        print(f'\nGenerating {layer_type} duplication patterns to test...')
        print(f'  Mode: {"Decoder-only" if decoder_only else "Symmetric (enc+dec)"}')
        print(f'  Parameter budget: {params_budget / 1e6:.2f}M')
        print(f'  Can duplicate up to {max_dup_layers} {layer_type} layers')

        if max_dup_layers < 1:
            print('  No duplication possible within parameter budget')
            patterns_to_test = []
        else:
            patterns_to_test = generate_duplication_patterns(n_layers, max_dup_layers)
            print(f'  Found {len(patterns_to_test)} patterns (within param budget)')

        # Random sample if too many patterns
        import random
        if len(patterns_to_test) > 200:
            print(f'Sampling 200 patterns from {len(patterns_to_test)} total...')
            random.seed(42)
            patterns_to_test = random.sample(patterns_to_test, 200)

        if not patterns_to_test:
            print('\nNo valid patterns found. Exiting.')
            return
    else:
        if decoder_only:
            layer_pattern = parse_pattern(args.eval_enc_pattern, n_dec_layers, "decoder")
            if layer_pattern:
                patterns_to_test = [(args.eval_enc_pattern, layer_pattern)]
            else:
                patterns_to_test = [(",".join(str(i + 1) for i in range(n_dec_layers)), list(range(n_dec_layers)))]
        else:
            layer_pattern = parse_pattern(args.eval_enc_pattern, n_enc_layers, "encoder")
            if layer_pattern:
                patterns_to_test = [(args.eval_enc_pattern, layer_pattern)]
            else:
                patterns_to_test = [(",".join(str(i + 1) for i in range(n_enc_layers)), list(range(n_enc_layers)))]

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
        if decoder_only:
            # Decoder-only mode: encoder unchanged, decoder duplicated
            enc_pattern = list(range(n_enc_layers))
            dec_pattern_rel = layer_pattern  # Pattern is relative to decoder
            dec_pattern = [i + n_enc_layers for i in dec_pattern_rel]
            full_pattern = enc_pattern + dec_pattern
            full_pattern_str = ",".join(str(i + 1) for i in full_pattern)

            print(f'\n[{i+1}/{len(patterns_to_test)}] Testing dec pattern: {pat_str}')
            print(f'  Full pattern: {full_pattern_str}')
            print(f'  Encoder layers: {len(enc_pattern)}, Decoder layers: {len(dec_pattern)}')

            # Create wrapped model (decoder-only)
            model = GPTWithDecoderDuplication(base_model, dec_pattern_rel).to(device)
        else:
            # Symmetric mode: both encoder and decoder duplicated
            enc_pattern = layer_pattern
            dec_pattern = [i + n_enc_layers for i in enc_pattern]
            full_pattern = enc_pattern + dec_pattern
            full_pattern_str = ",".join(str(i + 1) for i in full_pattern)

            print(f'\n[{i+1}/{len(patterns_to_test)}] Testing enc pattern: {pat_str}')
            print(f'  Full pattern: {full_pattern_str}')
            print(f'  Encoder layers: {len(enc_pattern)}, Decoder layers: {len(dec_pattern)}')

            # Create wrapped model (symmetric)
            model = GPTWithEncDecDuplication(base_model, enc_pattern).to(device)

        # Compile
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
        total_layers = len(enc_pattern) + len(dec_pattern)
        effective_params = total_params - (cfg['num_layers'] * block_params) + (total_layers * block_params)

        print(f'  val_loss: {val_loss:.6f}, val_bpb: {val_bpb:.6f}, time: {eval_time:.1f}s')
        print(f'  effective params: {effective_params / 1e6:.2f}M')

        results.append({
            'pattern': pat_str,
            'full_pattern': full_pattern_str,
            'enc_layers': len(enc_pattern),
            'dec_layers': len(dec_pattern),
            'total_layers': total_layers,
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
            print(f"{r['pattern']:25s} | L:{r['enc_layers']:2d}+{r['dec_layers']:2d} | P:{r['effective_params_m']:6.2f}M | bpb:{r['val_bpb']:.6f}")
        if len(results) > 10:
            print(f'... and {len(results) - 10} more')

        # Determine CSV path
        if args.csv_path:
            csv_path = args.csv_path
        else:
            model_path_obj = Path(args.model_path)
            suffix = '.dec_dup_results.csv' if decoder_only else '.enc_dec_dup_results.csv'
            csv_path = str(model_path_obj.with_suffix(suffix))

        # Save to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pattern', 'full_pattern', 'enc_layers', 'dec_layers',
                                                     'total_layers', 'val_loss', 'val_bpb', 'time_seconds', 'effective_params_m'])
            writer.writeheader()
            writer.writerows(results)
        print(f'\nResults saved to: {csv_path}')
    else:
        r = results[0]
        print(f'Pattern: {r["pattern"]}')
        print(f'Full pattern:     {r["full_pattern"]}')
        print(f'  val_loss: {r["val_loss"]:.6f}')
        print(f'  val_bpb:  {r["val_bpb"]:.6f}')
        print(f'  time: {r["time_seconds"]:.1f}s')
    print('=' * 80)


if __name__ == '__main__':
    main()
