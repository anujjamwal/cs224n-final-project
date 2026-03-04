from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoProcessor, Cache, LogitsProcessorList, PreTrainedModel, ProcessorMixin, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin, ALL_CACHE_NAMES, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedTokenizerBase
from trl.trainer.utils import get_config_model_id


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
) -> Tuple[torch.LongTensor, dict[str, Any]]:
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

    # Build pruned rows per batch element
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

    # Discard the KV cache — the next iteration will re-prefill from scratch
    old_cache = model_kwargs.pop('past_key_values', None)
    del old_cache
    del position_ids

    # Reset cache_position to cover the full pruned sequence so that
    # prepare_inputs_for_generation treats the next forward as a prefill
    model_kwargs['cache_position'] = torch.arange(max_len, dtype=torch.int64, device=device)
    model_kwargs['attention_mask'] = new_attention_mask

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
        **model_kwargs: Additional keyword arguments passed to the model forward pass.
    Returns:
        torch.LongTensor: Generated sequences of shape (batch_size, seq_len) including input and generated tokens.
    Notes:
        - In prune-aware mode, model location_ids are cleared. This allows the the model to regenerate the
          location_ids and fill the KV Cache. In prune agnostic mode, we retain the location_ids of tokens
          after pruning.
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

        # Store scores, attentions and hidden_states when required
        # if return_dict_in_generate:
        #     if output_scores:
        #         scores += (next_token_scores,)
        #     if output_logits:
        #         raw_logits += (next_token_logits,)
        #     if output_attentions:
        #         decoder_attentions += (
        #             (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
        #         )
        #         if model.config.is_encoder_decoder:
        #             cross_attentions += (outputs.cross_attentions,)

        #     if output_hidden_states:
        #         decoder_hidden_states += (
        #             (outputs.decoder_hidden_states,)
        #             if model.config.is_encoder_decoder
        #             else (outputs.hidden_states,)
        #         )

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
                model_kwargs=model_kwargs
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


def generate(model, processing_class = None, **kwargs):
    """Custom generate method for Hierarchical Chain of Thought"""

    return model.generate(
        custom_generate=_sample,
        processing_class=processing_class,
        **kwargs
    )
