from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn
from trl.trainer.utils import get_config_model_id
from transformers import  AutoProcessor, Cache, DynamicCache, LogitsProcessorList, ProcessorMixin, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin, ALL_CACHE_NAMES, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# KV-cache manipulation
# ---------------------------------------------------------------------------

def _retain_and_prune_kv_cache(
    cache: DynamicCache,
    prune_map: dict[int, Tuple[int, int]],
    batch_size: int,
    old_seq_len: int,
) -> int:
    """Modify a DynamicCache in-place after a pruning event.

    For each pruned batch element the function keeps only the prefix
    ``[0..thought_pos]`` in the cache.  Both thought and solution tokens are
    removed.  The caller sets ``cache_position`` so the next forward pass
    re-processes the solution tokens against the cached prefix.

    For non-pruned batch elements all entries are kept.

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
    head_dim = first_keys.shape[-1]

    # Compute per-element prefix length as plain ints (no GPU tensors needed).
    # The kept positions are always contiguous [0..N) by construction, so we
    # only need the length, not explicit index tensors.
    prefix_lengths: list[int] = []
    for b in range(batch_size):
        if b in prune_map:
            thought_pos, _solution_pos = prune_map[b]
            prefix_lengths.append(thought_pos + 1)
        else:
            prefix_lengths.append(old_seq_len)

    max_new_seq = max(prefix_lengths)

    # Fast path: all batch elements keep the same contiguous prefix length.
    # Common case (batch_size=1, or all pruned to the same point).
    all_same_prefix = len(set(prefix_lengths)) == 1

    # Pre-compute gather indices once (used across all layers) for the
    # heterogeneous fallback path only.
    if not all_same_prefix:
        idx = torch.zeros(batch_size, max_new_seq, dtype=torch.long, device=device)
        for b in range(batch_size):
            n = prefix_lengths[b]
            idx[b, :n] = torch.arange(n, device=device)

    for layer_idx in range(num_layers):
        if use_layers_api:
            old_keys = cache.layers[layer_idx].keys    # (batch, heads, seq, dim)
            old_vals = cache.layers[layer_idx].values
        else:
            old_keys = cache.key_cache[layer_idx]      # type: ignore[attr-defined]
            old_vals = cache.value_cache[layer_idx]     # type: ignore[attr-defined]

        if all_same_prefix:
            # Single slice — no allocation, just a contiguous copy
            new_keys = old_keys[:, :, :prefix_lengths[0], :].contiguous()
            new_vals = old_vals[:, :, :prefix_lengths[0], :].contiguous()
        else:
            # Vectorized gather across batch and heads
            num_heads = old_keys.shape[1]
            idx_expanded = idx[:, None, :, None].expand(-1, num_heads, -1, head_dim)
            new_keys = torch.gather(old_keys, 2, idx_expanded)
            new_vals = torch.gather(old_vals, 2, idx_expanded)
            # Zero out padding positions beyond each element's prefix.
            # Without this, gather copies position-0 values into padding slots,
            # which FA2 would attend to (it ignores attention masks).
            for b in range(batch_size):
                n = prefix_lengths[b]
                if n < max_new_seq:
                    new_keys[b, :, n:, :] = 0
                    new_vals[b, :, n:, :] = 0

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
    # After a prune event, cache_position covers only the solution tokens
    # that need re-processing (fewer than input_ids).  The standard
    # prepare_inputs_for_generation expects them to match, so pre-slice
    # input_ids to the tokens at cache_position.  This also handles the
    # normal decode case where cache_position is a single element.
    if (
        cache_position is not None
        and past_key_values is not None
        and input_ids.shape[1] != cache_position.shape[0]
    ):
        input_ids = input_ids[:, cache_position]

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
    use_pos_ids_buf: bool = False,
) -> dict[str, Any]:
    model_kwargs = GenerationMixin._update_model_kwargs_for_generation(
        model,
        outputs,
        model_kwargs,
        is_encoder_decoder=is_encoder_decoder,
        num_new_tokens=num_new_tokens
    )

    # HF's _update concatenates new positions onto cache_position, which is
    # correct for the initial prefill→decode transition but breaks after a
    # prune: the multi-element cache_position from _prune_model_inputs gets
    # carried forward, causing prepare_inputs_for_generation to re-select
    # already-cached tokens on every subsequent step.  DynamicCache.update()
    # appends these duplicates, inflating the cache until old_seq_len exceeds
    # the pruned input length and torch.arange crashes.  Fix: always keep
    # only the last element (the next decode position).
    if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:]

    # HF's _update_model_kwargs_for_generation does not manage position_ids.
    # For prune-agnostic mode we track position_ids explicitly after each
    # prune event so that RoPE positions match training.
    # When use_pos_ids_buf=True, _sample manages position_ids via a
    # pre-allocated buffer, so skip the O(n) concatenation here.
    if (
        "position_ids" in model_kwargs
        and model_kwargs["position_ids"] is not None
        and not use_pos_ids_buf
    ):
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

    When ``retain_kv_cache=True`` (default), the function retains the
    prefix (tokens before the thought block) in the KV cache and sets
    ``cache_position`` so the next forward pass re-processes the solution
    tokens against the cached prefix.  Cost: O(k) where k = number of
    solution tokens.

    When ``retain_kv_cache=False``, the entire cache is discarded and a
    full re-prefill is performed.  Cost: O(N).
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
    if cache is None:
        cache_populated = False
    elif use_layers_api:
        cache_populated = len(cache.layers) > 0
    elif hasattr(cache, 'key_cache'):
        cache_populated = len(cache.key_cache) > 0  # type: ignore[union-attr]
    else:
        cache_populated = False
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

    # Build pruned input_ids, attention_mask, and position_ids
    if batch_size == 1:
        # Fast path: no inter-batch padding needed
        new_input_ids = new_rows[0].unsqueeze(0)
        max_len = new_input_ids.shape[1]
        model_kwargs['attention_mask'] = torch.ones(1, max_len, dtype=torch.long, device=device)
        if is_prune_agnostic:
            model_kwargs['position_ids'] = new_position_rows[0].unsqueeze(0)
        else:
            model_kwargs['position_ids'] = None
    else:
        # Pad to uniform length across batch
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

        new_cache_seq = _retain_and_prune_kv_cache(
            cache=cache,  # type: ignore[arg-type]
            prune_map=prune_map,
            batch_size=batch_size,
            old_seq_len=old_seq_len,
        )

        # Cache has only the prefix (a tokens).  Set cache_position so
        # the next forward pass re-processes the solution tokens (c tokens)
        # against the cached prefix.  Cost: O(k) where k = solution length.
        model_kwargs['cache_position'] = torch.arange(
            new_cache_seq, max_len, dtype=torch.int64, device=device,
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


# This code has been copied over from transformers library utils.py and updated to add the 
# code specific to context pruning. This code only works on decoder models even though
# at places there are references to is_encoder_decoder check.
def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    thought_token_id: int,
    solution_token_id: int,
    return_token_id: int,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    prune_aware: bool = True,
    retain_kv_cache: bool = True,
    return_unpruned_output: bool = False,
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
        thought_token_id (int): Token ID for [THOUGHT].
        solution_token_id (int): Token ID for [SOLUTION].
        return_token_id (int): Token ID for [RETURN].
        synced_gpus (bool): Whether GPUs are synchronized for multi-device generation. Defaults to False.
        streamer (Optional[BaseStreamer]): Optional streamer to output tokens during generation. Defaults to None.
        prune_aware (bool): When True, positions are renumbered after pruning (contiguous). Defaults to True.
        retain_kv_cache (bool): When True (default), retains the prefix in the KV cache
            after pruning and re-processes only the solution tokens.  Cost: O(k).
            When False, discards the cache entirely and re-prefills.  Cost: O(N).
        **model_kwargs: Additional keyword arguments passed to the model forward pass.
    Returns:
        torch.LongTensor: Generated sequences of shape (batch_size, seq_len) including input and generated tokens.
    Notes:
        - With retain_kv_cache=True, the prefix is cached and solution tokens are
          re-processed via a standard forward pass.
        - With retain_kv_cache=False, the KV cache is discarded and the full
          pruned sequence is re-prefilled from scratch.
        - In prune-agnostic mode, position_ids are tracked explicitly to maintain original RoPE
          positions for surviving tokens.
        - Handles batch generation with unfinished sequences tracking.
        - Manages KV-cache through model_kwargs for efficient decoding.
    """

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

    if return_unpruned_output:
        unpruned_ids = [input_ids[b].tolist() for b in range(batch_size)]

    # Use eager-mode forward for all steps.  torch.compile with
    # mode="reduce-overhead" uses CUDAGraphs which record a separate graph
    # for every distinct input shape.  During decode the KV-cache sequence
    # length grows each step, producing a new shape every token — this
    # causes excessive graph-recording overhead and no speedup.
    model_forward = model.__call__

    # Pre-allocate input_ids buffer to avoid O(n²) copies from torch.cat
    # on every token step.  input_ids becomes a view into this buffer.
    _pad_id_scalar = pad_token_id.item() if isinstance(pad_token_id, torch.Tensor) else pad_token_id
    _max_new = generation_config.max_new_tokens if generation_config.max_new_tokens is not None else generation_config.max_length
    _buf_len = input_ids.shape[1] + _max_new
    _ids_buf = torch.full(
        (batch_size, _buf_len), _pad_id_scalar,
        dtype=input_ids.dtype, device=input_ids.device,
    )
    _ids_buf[:, :input_ids.shape[1]] = input_ids
    _cur_len = input_ids.shape[1]
    input_ids = _ids_buf[:, :_cur_len]

    # Pre-allocate position_ids buffer for prune-agnostic mode to avoid
    # O(n²) torch.cat growth in _update_model_kwargs_for_generation.
    _pos_ids_buf = None
    if not prune_aware:
        _pos_ids_buf = torch.zeros(
            (batch_size, _buf_len), dtype=torch.long, device=input_ids.device,
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
            use_pos_ids_buf=(_pos_ids_buf is not None),
        )

        # Extend position_ids via buffer instead of torch.cat (prune-agnostic mode)
        if _pos_ids_buf is not None and model_kwargs.get("position_ids") is not None:
            pos_ids = model_kwargs["position_ids"]
            plen = pos_ids.shape[1]
            _pos_ids_buf[:, :plen] = pos_ids
            _pos_ids_buf[:, plen] = _pos_ids_buf[:, plen - 1] + 1
            model_kwargs["position_ids"] = _pos_ids_buf[:, :plen + 1]

        if synced_gpus and this_peer_finished:
            continue

        # .float() severs the reference to outputs.logits (avoids keeping the
        # full logits tensor alive) and converts to float32 in a single op.
        # The redundant device= kwarg is removed (logits are already on the
        # correct device).  The del outputs below provides additional safety.
        next_token_logits = outputs.logits[:, -1, :].float() # type: ignore

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
        # Write into pre-allocated buffer instead of O(n²) torch.cat
        _ids_buf[:, _cur_len] = next_tokens
        _cur_len += 1
        input_ids = _ids_buf[:, :_cur_len]

        # Optimization: This is done to avoid the cpu() call unless needed
        _next_cpu = None
        def _get_next_cpu():
            nonlocal _next_cpu
            if _next_cpu is None:
                _next_cpu = next_tokens.cpu()
            return _next_cpu
        
        if streamer is not None:
            streamer.put(_get_next_cpu())

        pos = input_ids.shape[1] - 1

        if return_unpruned_output:
            _cpu = _get_next_cpu()
            for b in range(batch_size):
                unpruned_ids[b].append(_cpu[b].item())

        # Optimization: This comparison happens on GPU and we only sync once
        # during the if check to see if there is any special token
        _is_thought = next_tokens == thought_token_id
        _is_solution = next_tokens == solution_token_id
        _is_return = next_tokens == return_token_id
        _any_special = (_is_thought | _is_solution | _is_return).any()
        
        if _any_special.item():
            for b in _is_thought.nonzero(as_tuple=True)[0].tolist():
                if stacks[b]:
                    stacks[b][-1][1] = pos

            for b in _is_solution.nonzero(as_tuple=True)[0].tolist():
                if stacks[b]:
                    stacks[b][-1][1] = pos

            for b in _is_return.nonzero(as_tuple=True)[0].tolist():
                if stacks[b]:
                    stacks[b][-1][2] = pos

            return_indices = _is_return.nonzero(as_tuple=True)[0].tolist()
            prune_candidates = [idx for idx in return_indices if stacks[idx]]
        else:
            prune_candidates = []

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
            # Copy pruned result back into the pre-allocated buffer.
            # No need to zero the tail — it was initialized to pad_id and
            # input_ids is always sliced to _cur_len so stale data is never read.
            _cur_len = input_ids.shape[1]
            _ids_buf[:, :_cur_len] = input_ids
            input_ids = _ids_buf[:, :_cur_len]

            # Sync position_ids into buffer after pruning (prune-agnostic mode)
            if _pos_ids_buf is not None and model_kwargs.get("position_ids") is not None:
                pos_ids = model_kwargs["position_ids"]
                _pos_ids_buf[:, :pos_ids.shape[1]] = pos_ids
                model_kwargs["position_ids"] = _pos_ids_buf[:, :pos_ids.shape[1]]

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores) # type: ignore
        this_peer_finished = not unfinished_sequences.any()

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs # type: ignore

    if streamer is not None:
        streamer.end()

    if return_unpruned_output:
        max_len = max(len(ids) for ids in unpruned_ids)
        pad_id = pad_token_id.item() if isinstance(pad_token_id, torch.Tensor) else pad_token_id
        unpruned_tensor = torch.full(
            (batch_size, max_len), pad_id,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        for b, ids in enumerate(unpruned_ids):
            unpruned_tensor[b, :len(ids)] = torch.tensor(ids, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = unpruned_tensor

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


def generate(
    model, 
    tokenizer: PreTrainedTokenizerBase, 
    retain_kv_cache: bool = True, 
    return_unpruned_output: bool = False, 
    prune_aware: bool = True,
    **kwargs
):
    """Custom generate method for Hierarchical Chain of Thought.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        retain_kv_cache: When True (default), retains the prefix in the KV
            cache after pruning and re-processes solution tokens.  When False,
            discards the cache entirely.
        **kwargs: Forwarded to ``model.generate``.
    """
    custom_generate = kwargs.pop('custom_generate', _sample)
    thought_token_id = tokenizer.convert_tokens_to_ids('[THOUGHT]')
    solution_token_id = tokenizer.convert_tokens_to_ids('[SOLUTION]')
    return_token_id = tokenizer.convert_tokens_to_ids('[RETURN]')

    return GenerationMixin.generate(
        model,
        custom_generate=custom_generate,
        thought_token_id=thought_token_id,
        solution_token_id=solution_token_id,
        return_token_id=return_token_id,
        retain_kv_cache=retain_kv_cache,
        return_unpruned_output=return_unpruned_output,
        prune_aware=prune_aware,
        **kwargs
    )
