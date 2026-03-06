from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import  Cache, DynamicCache, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin, ALL_CACHE_NAMES, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# RoPE correction helpers for KV-cache retention after pruning
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate halves — same convention as HuggingFace RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_correction(
    shift: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin tensors for RoPE position-shift correction.

    When cached keys need their RoPE positions decreased by *shift* (i.e. they
    move from position p to p − shift), we apply a rotation of angle −shift·θᵢ
    to each dimension pair i.

    In the HF ``apply_rotary_pos_emb`` convention
        k_new = k · cos_corr + rotate_half(k) · sin_corr
    this translates to:
        cos_corr = cos(shift · θ)   (cos is even, so cos(−α) = cos(α))
        sin_corr = −sin(shift · θ)  (sin is odd,  so sin(−α) = −sin(α))

    Returns ``(cos_corr, sin_corr)`` each of shape ``(head_dim,)``.
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = shift * inv_freq                        # (half_dim,)
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = -torch.sin(angles).to(dtype)          # negative for backward shift
    # Duplicate to full head_dim (matching HF's ``emb = cat(freqs, freqs)``)
    cos_corr = torch.cat([cos_vals, cos_vals])       # (head_dim,)
    sin_corr = torch.cat([sin_vals, sin_vals])       # (head_dim,)
    return cos_corr, sin_corr


def _apply_rope_correction(
    keys: torch.Tensor,
    cos_corr: torch.Tensor,
    sin_corr: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE position correction to cached keys.

    Uses the same rotation convention as HuggingFace:
        k_corrected = k · cos + rotate_half(k) · sin

    Args:
        keys:     shape ``(..., seq, head_dim)``
        cos_corr: shape ``(head_dim,)``
        sin_corr: shape ``(head_dim,)``  (already negated for backward shift)
    """
    return keys * cos_corr + _rotate_half(keys) * sin_corr


# ---------------------------------------------------------------------------
# KV-cache manipulation
# ---------------------------------------------------------------------------

def _retain_and_prune_kv_cache(
    cache: DynamicCache,
    prune_map: dict[int, Tuple[int, int]],
    batch_size: int,
    old_seq_len: int,
    head_dim: int,
    rope_theta: float,
    prune_aware: bool,
) -> int:
    """Modify a DynamicCache in-place after a pruning event.

    For each pruned batch element the function:
      1. Removes KV entries for the pruned token range
         ``(thought_pos, solution_pos]``.
      2. *(prune-aware only)* Applies a RoPE correction rotation to keys
         whose positions shifted left.
      3. Removes the **last** KV entry so that the generation loop naturally
         re-processes the final token on the next iteration (needed to
         produce fresh logits that reflect the pruned context).

    For non-pruned batch elements the last entry is also removed so the
    entire batch can proceed uniformly through a single-token forward pass.

    Compatible with both transformers ≥5.x (``cache.layers[i].keys/values``)
    and older versions (``cache.key_cache[i]``).

    Returns the new cache sequence length (``max_new_seq``).
    """
    # Detect API version: transformers ≥5 uses cache.layers, older uses cache.key_cache
    use_layers_api = hasattr(cache, 'layers')
    if use_layers_api:
        num_layers = len(cache.layers)
    elif hasattr(cache, 'key_cache'):
        num_layers = len(cache.key_cache)  # type: ignore[attr-defined]
    else:
        return 0

    if num_layers == 0:
        return 0

    if use_layers_api:
        first_keys = cache.layers[0].keys if len(cache.layers) > 0 else None
    else:
        first_keys = cache.key_cache[0] if len(cache.key_cache) > 0 else None  # type: ignore[attr-defined]
    if first_keys is None:
        return 0
    device = first_keys.device
    dtype = first_keys.dtype

    # Pre-compute per-element keep indices and RoPE corrections
    keep_indices: list[torch.Tensor] = []
    rope_corrections: list[Tuple[torch.Tensor, torch.Tensor] | None] = []
    # Index in the *kept* tensor where the "after block" keys start
    split_points: list[int] = []

    for b in range(batch_size):
        if b in prune_map:
            thought_pos, solution_pos = prune_map[b]
            shift = solution_pos - thought_pos  # number of positions removed

            # Keep [0..thought_pos] ∪ [solution_pos+1..old_seq_len-2]
            # (exclude old_seq_len-1 = the last entry, to force re-processing)
            before = torch.arange(0, thought_pos + 1, device=device)
            after = torch.arange(solution_pos + 1, old_seq_len - 1, device=device)
            keep = torch.cat([before, after])
            keep_indices.append(keep)

            split_at = thought_pos + 1  # where "after" entries start in kept
            split_points.append(split_at)

            if prune_aware and shift > 0:
                cos_corr, sin_corr = _build_rope_correction(
                    shift, head_dim, rope_theta, device, dtype
                )
                rope_corrections.append((cos_corr, sin_corr))
            else:
                rope_corrections.append(None)
        else:
            # Non-pruned: keep all except the last entry
            keep = torch.arange(0, old_seq_len - 1, device=device)
            keep_indices.append(keep)
            split_points.append(old_seq_len)  # no split needed
            rope_corrections.append(None)

    max_new_seq = max(k.shape[0] for k in keep_indices)

    for layer_idx in range(num_layers):
        if use_layers_api:
            old_keys = cache.layers[layer_idx].keys    # (batch, heads, seq, dim)
            old_vals = cache.layers[layer_idx].values
        else:
            old_keys = cache.key_cache[layer_idx]      # type: ignore[attr-defined]
            old_vals = cache.value_cache[layer_idx]     # type: ignore[attr-defined]
        num_heads = old_keys.shape[1]

        new_keys = torch.zeros(
            batch_size, num_heads, max_new_seq, head_dim,
            dtype=dtype, device=device,
        )
        new_vals = torch.zeros_like(new_keys)

        for b in range(batch_size):
            keep = keep_indices[b]
            kept_k = old_keys[b, :, keep, :]   # (heads, kept_len, dim)
            kept_v = old_vals[b, :, keep, :]

            # Apply RoPE correction to shifted keys
            corr = rope_corrections[b]
            if corr is not None:
                sp = split_points[b]
                kept_k[:, sp:, :] = _apply_rope_correction(
                    kept_k[:, sp:, :], corr[0], corr[1]
                )

            n = keep.shape[0]
            new_keys[b, :, :n, :] = kept_k
            new_vals[b, :, :n, :] = kept_v

        if use_layers_api:
            cache.layers[layer_idx].keys = new_keys
            cache.layers[layer_idx].values = new_vals
        else:
            cache.key_cache[layer_idx] = new_keys      # type: ignore[attr-defined]
            cache.value_cache[layer_idx] = new_vals     # type: ignore[attr-defined]

    if hasattr(cache, '_seen_tokens'):
        cache._seen_tokens = max_new_seq
    return max_new_seq


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _prepare_inputs_for_generation(
    model,
    input_ids: torch.LongTensor,
    past_key_values: Cache | None = None,
    attention_mask: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    is_first_iteration: bool | None = False,
    **kwargs,
):
    model_inputs = GenerationMixin.prepare_inputs_for_generation(
        model,
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        is_first_iteration=is_first_iteration,
        **kwargs
    )

    return model_inputs

def _update_model_kwargs_for_generation(
    model,
    outputs: ModelOutput,
    model_kwargs: dict[str, Any],
    is_encoder_decoder: bool = False,
    num_new_tokens: int = 1,
) -> dict[str, Any]:
    model_kwargs = GenerationMixin._update_model_kwargs_for_generation(
        model,
        outputs,
        model_kwargs,
        is_encoder_decoder=is_encoder_decoder,
        num_new_tokens=num_new_tokens
    )

    # HF's _update_model_kwargs_for_generation does not manage position_ids.
    # For prune-agnostic mode we track position_ids explicitly after each
    # prune event so that RoPE positions match training.  Extend the tensor
    # by one (next position = last + 1) so that prepare_inputs_for_generation
    # can slice the correct value for the next decode step.
    if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
        pos = model_kwargs["position_ids"]
        model_kwargs["position_ids"] = torch.cat([pos, pos[:, -1:] + 1], dim=-1)

    return model_kwargs


def _prune_model_inputs(
    model,
    prune_input_candidates: Sequence[int],
    prune_input_locations: Sequence[Sequence[Tuple[int, int, int]]],
    input_ids: torch.LongTensor,
    prune_aware: bool,
    model_kwargs: dict[str, Any],
    retain_kv_cache: bool = True,
) -> Tuple[torch.LongTensor, dict[str, Any]]:
    """Prune input sequences after a ``[RETURN]`` token is generated.

    When ``retain_kv_cache=True`` (default) and batch size is 1, the function
    retains the KV cache, removes entries for pruned positions, and applies a
    RoPE correction to shifted keys.  This reduces the post-prune cost from a
    full O(N) re-prefill to a single-token O(1) forward pass.

    Falls back to the original full-discard strategy when ``retain_kv_cache``
    is False or when batching makes cache surgery impractical.
    """
    is_prune_agnostic = not prune_aware

    batch_size = input_ids.shape[0]
    device = input_ids.device

    if is_prune_agnostic:
        # Construct position_ids from model_kwargs if already tracked,
        # otherwise build contiguous positions (correct before first prune).
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
        else:
            position_ids = torch.arange(
                input_ids.shape[1], device=device
            ).unsqueeze(0).expand(batch_size, -1)
    else:
        position_ids = None

    # Map batch index → (thought_pos, solution_pos) for valid prune targets
    prune_map: dict[int, Tuple[int, int]] = {}
    for cand_idx, batch_idx in enumerate(prune_input_candidates):
        for thought_pos, solution_pos, _return_pos in prune_input_locations[cand_idx]:
            if solution_pos is not None:
                prune_map[batch_idx] = (thought_pos, solution_pos)

    if not prune_map:
        return input_ids, model_kwargs

    # ------------------------------------------------------------------
    # Decide whether to retain the KV cache
    # ------------------------------------------------------------------
    cache = model_kwargs.get('past_key_values', None)
    use_layers_api = hasattr(cache, 'layers')
    cache_populated = (
        (use_layers_api and len(cache.layers) > 0) if cache is not None
        else False
    ) or (
        hasattr(cache, 'key_cache') and len(cache.key_cache) > 0  # type: ignore[union-attr]
    ) if cache is not None else False
    can_retain_cache = (
        retain_kv_cache
        and cache is not None
        and isinstance(cache, DynamicCache)
        and cache_populated
    )

    # ------------------------------------------------------------------
    # Build pruned rows per batch element (same as before)
    # ------------------------------------------------------------------
    new_rows: list[torch.Tensor] = []
    if is_prune_agnostic:
        new_position_rows: list[torch.Tensor] = list(
            position_ids[b] for b in range(batch_size)  # type: ignore
        )

    for b in range(batch_size):
        if b in prune_map:
            thought_pos, solution_pos = prune_map[b]
            new_rows.append(torch.cat((input_ids[b, :thought_pos + 1], input_ids[b, solution_pos + 1:])))
            if is_prune_agnostic:
                new_position_rows[b] = torch.cat((position_ids[b, :thought_pos + 1], position_ids[b, solution_pos + 1:])) # type: ignore
        else:
            new_rows.append(input_ids[b])

    # Pad to uniform length
    max_len = max(r.shape[0] for r in new_rows)
    pad_id = getattr(model.config, 'pad_token_id', None)
    if pad_id is None:
        pad_id = getattr(model.config, 'eos_token_id', 0)

    new_input_ids = torch.full((batch_size, max_len), pad_id, dtype=input_ids.dtype, device=device)
    new_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b, r in enumerate(new_rows):
        new_input_ids[b, :r.shape[0]] = r
        new_attention_mask[b, :r.shape[0]] = 1

    if is_prune_agnostic:
        new_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for b, p in enumerate(new_position_rows):
            new_position_ids[b, :p.shape[0]] = p
        model_kwargs['position_ids'] = new_position_ids
    else:
        model_kwargs['position_ids'] = None

    model_kwargs['attention_mask'] = new_attention_mask

    # ------------------------------------------------------------------
    # KV cache handling
    # ------------------------------------------------------------------
    if can_retain_cache:
        if hasattr(cache, 'layers') and len(cache.layers) > 0:
            old_seq_len = cache.layers[0].keys.shape[2]
        else:
            old_seq_len = cache.key_cache[0].shape[2]  # type: ignore[union-attr]
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        rope_theta = getattr(model.config, 'rope_theta', 10000.0)

        new_cache_seq = _retain_and_prune_kv_cache(
            cache=cache,  # type: ignore[arg-type]
            prune_map=prune_map,
            batch_size=batch_size,
            old_seq_len=old_seq_len,
            head_dim=head_dim,
            rope_theta=rope_theta,
            prune_aware=prune_aware,
        )

        # The cache now has new_cache_seq entries (= max_len − 1).
        # Set cache_position so the next forward pass processes just
        # the last token of the pruned sequence (single-token decode).
        model_kwargs['cache_position'] = torch.tensor(
            [max_len - 1], dtype=torch.int64, device=device,
        )
    else:
        # Fallback: discard the KV cache and re-prefill from scratch
        old_cache = model_kwargs.pop('past_key_values', None)
        del old_cache

        # Reset cache_position to cover the full pruned sequence so that
        # prepare_inputs_for_generation treats the next forward as a prefill
        model_kwargs['cache_position'] = torch.arange(max_len, dtype=torch.int64, device=device)

    if position_ids is not None:
        del position_ids

    return new_input_ids, model_kwargs # type: ignore



def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    prune_aware: bool = False,
    retain_kv_cache: bool = True,
    **model_kwargs,
):
    """Generate sequences using argmax or sampling from model logits.
    This function implements the core generation loop for token-by-token sequence generation.
    It supports both deterministic (argmax) and stochastic (sampling) token selection, and includes
    special handling for pruning sequences based on custom tokens ([THOUGHT], [SOLUTION], [RETURN]).
    Args:
        model: The language model used for generation.
        input_ids (torch.LongTensor): Initial input token IDs of shape (batch_size, seq_len).
        logits_processor (LogitsProcessorList): List of processors to apply to logits before sampling.
        stopping_criteria (StoppingCriteriaList): List of criteria to determine when to stop generation.
        generation_config (GenerationConfig): Configuration object containing generation parameters.
        processing_class (Optional[PreTrainedTokenizerBase]): Tokenizer for token-id conversion. Defaults to None.
        synced_gpus (bool): Whether GPUs are synchronized for multi-device generation. Defaults to False.
        streamer (Optional[BaseStreamer]): Optional streamer to output tokens during generation. Defaults to None.
        prune_aware (bool): When True, positions are renumbered after pruning (contiguous). Defaults to False.
        retain_kv_cache (bool): When True, retains and surgically modifies the KV cache after
            pruning instead of discarding it.  Reduces post-prune cost from O(N) re-prefill to
            O(1) single-token forward pass.  Defaults to True.
        **model_kwargs: Additional keyword arguments passed to the model forward pass.
    Returns:
        torch.LongTensor: Generated sequences of shape (batch_size, seq_len) including input and generated tokens.
    Notes:
        - In prune-aware mode with retain_kv_cache=True, after pruning the KV cache is retained
          with RoPE-corrected keys for shifted positions. Only the last token is re-processed.
        - In prune-aware mode with retain_kv_cache=False, the KV cache is discarded and the full
          pruned sequence is re-prefilled from scratch.
        - In prune-agnostic mode, position_ids are tracked explicitly to maintain original RoPE
          positions for surviving tokens.
        - Handles batch generation with unfinished sequences tracking.
        - Manages KV-cache through model_kwargs for efficient decoding.
    """

    thought_token_id = processing_class.convert_tokens_to_ids("[THOUGHT]")
    solution_token_id = processing_class.convert_tokens_to_ids("[SOLUTION]")
    return_token_id = processing_class.convert_tokens_to_ids("[RETURN]")

    pad_token_id = generation_config._pad_token_tensor # type: ignore
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    batch_size = input_ids.shape[0]
    this_peer_finished: bool = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    stacks = [[] for _ in range(batch_size)]

    model_forward = (
        model.get_compiled_call(generation_config.compile_config)
        if GenerationMixin._valid_auto_compile_criteria(model, model_kwargs, generation_config)
        else model.__call__
    )

    # Assisted generation completes the prefill stage in candidate generator so that
    # we don't have several `prefill` calls in one generation loop. Skip `_prefill` for assistants
    if not generation_config.is_assistant:
        outputs = GenerationMixin._prefill(model, input_ids, generation_config, model_kwargs)
        prefill_consumed = False
    else:
        model_kwargs = GenerationMixin._get_initial_cache_position(model, input_ids.shape[1], input_ids.device, model_kwargs)
        prefill_consumed = True

    while GenerationMixin._has_unfinished_sequences(model, this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed:
            model_inputs = _prepare_inputs_for_generation(model, input_ids, **model_kwargs)
            with GenerationMixin._optimize_model_for_decode(model):
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True
        model_kwargs = _update_model_kwargs_for_generation(
            model,
            outputs, # type: ignore The value is always initialized
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device) # type: ignore

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) # type: ignore

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        last_token = next_tokens.view(-1)
        pos = input_ids.shape[1] - 1
        for b in (last_token == thought_token_id).nonzero(as_tuple=True)[0].tolist():
            stacks[b].append([pos, None, None])

        for b in (last_token == solution_token_id).nonzero(as_tuple=True)[0].tolist():
            if stacks[b]:
                stacks[b][-1][1] = pos

        for b in (last_token == return_token_id).nonzero(as_tuple=True)[0].tolist():
            if stacks[b]:
                stacks[b][-1][2] = pos

        return_indices = (last_token == return_token_id).nonzero(as_tuple=True)[0].tolist()
        prune_candidates = [idx for idx in return_indices if stacks[idx]]

        if prune_candidates:
            input_ids, model_kwargs = _prune_model_inputs(
                model,
                prune_input_candidates=prune_candidates,
                prune_input_locations=[[stacks[b].pop()] for b in prune_candidates],
                input_ids=input_ids,
                prune_aware=prune_aware,
                model_kwargs=model_kwargs,
                retain_kv_cache=retain_kv_cache,
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores) # type: ignore
        this_peer_finished = bool(unfinished_sequences.max() == 0)

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs # type: ignore

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]
        if model.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
    else:
        return input_ids


def generate(model, processing_class = None, retain_kv_cache = True, **kwargs):
    """Custom generate method for Hierarchical Chain of Thought.

    Args:
        model: The language model.
        processing_class: Tokenizer / processing class.
        retain_kv_cache: When True (default), retains the KV cache after pruning
            with surgical RoPE correction instead of discarding it.  This turns
            each post-prune step from a full O(N) re-prefill into an O(1)
            single-token decode.
        **kwargs: Forwarded to ``model.generate``.
    """

    return model.generate(
        custom_generate=_sample,
        processing_class=processing_class,
        retain_kv_cache=retain_kv_cache,
        **kwargs
    )
