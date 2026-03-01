from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import BaseStreamer, Cache, GenerationConfig, LogitsProcessorList, PreTrainedModel, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.utils.generic import ModelOutput
from transformers import PreTrainedTokenizerBase

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

    return model_kwargs


def _prune_model_inputs(
    model,
    prune_input_candidates: Sequence[int],
    prune_input_locations: Sequence[Sequence[Tuple[int, int, int]]],
    input_ids: torch.LongTensor,
    model_kwargs: dict[str, Any]
) -> Tuple[torch.LongTensor, dict[str, Any]]:

    # Prune input_ids[idx] at locations to torch.cat((tokens[b, :thought_pos+1], tokens[b, solution_pos+1:]))
    # Prune location_ids at the same locations
    # Prune attention_mask at the locations
    # Reshape the tensors after prune

    return input_ids, model_kwargs


def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    """Generate the sequence from model using argmax
    
    Algorithm:
    1. Get parameters from model_kwargs for position_ids etc    
    2. Loop until we hit a stopping criteria
        2.1 Generate the token
        2.2 Update the stack if one of 3 tokens
        2.3 If we got the return token
            2.3.1 Prune the input_ids
            2.3.2 Update the position_ids
            2.3.3 Update or clear the cache
    """
    tokenizer: PreTrainedTokenizerBase = model_kwargs.get("tokenizer") # type: ignore
    thought_token_id = tokenizer.convert_tokens_to_ids("[THOUGHT]")
    solution_token_id = tokenizer.convert_tokens_to_ids("[SOLUTION]")
    return_token_id = tokenizer.convert_tokens_to_ids("[RETURN]")

    pad_token_id = generation_config._pad_token_tensor # type: ignore
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    scores = ()
    
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
        #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
        #         )
        #         if self.config.is_encoder_decoder:
        #             cross_attentions += (outputs.cross_attentions,)

        #     if output_hidden_states:
        #         decoder_hidden_states += (
        #             (outputs.decoder_hidden_states,)
        #             if self.config.is_encoder_decoder
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
        
        last_token = next_tokens.squeeze(-1)
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
                model_kwargs=model_kwargs
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores) # type: ignore
        this_peer_finished = bool(unfinished_sequences.max() == 0)

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs # type: ignore

    if streamer is not None:
        streamer.end()

    # if return_dict_in_generate:
    #     cache = None
    #     if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
    #         cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
    #         cache = model_kwargs[cache_key]
    #     if self.config.is_encoder_decoder:
    #         return GenerateEncoderDecoderOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             logits=raw_logits,
    #             encoder_attentions=encoder_attentions,
    #             encoder_hidden_states=encoder_hidden_states,
    #             decoder_attentions=decoder_attentions,
    #             cross_attentions=cross_attentions,
    #             decoder_hidden_states=decoder_hidden_states,
    #             past_key_values=cache,
    #         )
    #     else:
    #         return GenerateDecoderOnlyOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             logits=raw_logits,
    #             attentions=decoder_attentions,
    #             hidden_states=decoder_hidden_states,
    #             past_key_values=cache,
    #         )
    # else:
    #     return input_ids
    return input_ids


def generate(model, **kwargs):
    """Custom generate method for Hierarchical Chain of Thought"""

    return model.generate(
        custom_generate=_sample,
        **kwargs
    )
