import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput

from .masks import extract_cot_blocks


@dataclass
class HCotGenerateOutput(GenerateDecoderOnlyOutput):
    """Generation output with per-record token processing statistics.

    Per-record fields (lists of length batch_size):
        total_tokens_processed: sum of seq_len fed to the model at each forward pass
        output_tokens: actual (non-padded) output length for each record
        prune_events: number of prune operations for each record
        tokens_pruned: total tokens removed by pruning for each record

    Scalar fields:
        prompt_tokens: input sequence length (same for all records)
        generated_tokens: number of forward pass steps (same for all records)
        wall_time_seconds: wall clock time for the entire generate call
    """
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    total_tokens_processed: Optional[list] = None
    output_tokens: Optional[list] = None
    prune_events: Optional[list] = None
    tokens_pruned: Optional[list] = None
    wall_time_seconds: Optional[float] = None

def generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    return_token_id,
    solution_token_id,
    thought_token_id,
    eos_token_id,
    **model_kwargs
):
    start_time = time.perf_counter()

    batch_size = input_ids.shape[0]
    tokens = input_ids.clone()
    prompt_tokens = input_ids.shape[1]
    total_tokens_processed = [0] * batch_size
    num_generated = 0
    prune_events = [0] * batch_size
    tokens_pruned = [0] * batch_size

    # Track sub thoughts – one stack per batch element
    stacks = [[] for _ in range(batch_size)]

    # Track which batch elements are still generating
    attention_mask = model_kwargs.pop('attention_mask')
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    max_token_length = model_kwargs.get("max_new_tokens", stopping_criteria[0].max_length)

    while tokens.shape[1] < max_token_length:
        for b in range(batch_size):
            total_tokens_processed[b] += tokens.shape[1]
        num_generated += 1
        outputs = model(tokens, attention_mask=attention_mask, **model_kwargs)
        logits = outputs.logits
        next_token_logits = logits_processor(tokens, logits[:, -1, :])

        next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
        tokens = torch.cat((tokens, next_tokens), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1)

        # Check for EOS
        unfinished = unfinished & (next_tokens.squeeze(-1) != eos_token_id)
        if not unfinished.any():
            break

        # Honour all stopping criteria (e.g. MaxTimeCriteria, custom callbacks)
        if stopping_criteria(tokens, next_token_logits).all():
            break
        
        sq = next_tokens.squeeze(-1)
        pos = tokens.shape[1] - 1  # index of the just-appended token

        # Track [THOUGHT] and [SOLUTION] tokens — sparse iteration over active elements only
        for b in (sq == thought_token_id).nonzero(as_tuple=True)[0].tolist():
            stacks[b].append([pos, None])

        for b in (sq == solution_token_id).nonzero(as_tuple=True)[0].tolist():
            if stacks[b]:
                stacks[b][-1][1] = pos

        # Prune CoT per batch element that generated [RETURN]
        return_indices = (sq == return_token_id).nonzero(as_tuple=True)[0].tolist()
        if return_indices:
            new_rows = []
            pruned_any = False
            for b in range(batch_size):
                if b in return_indices and stacks[b]:
                    thought_pos, solution_pos = stacks[b].pop()
                    if solution_pos is not None:
                        prune_events[b] += 1
                        tokens_pruned[b] += solution_pos - thought_pos
                        new_rows.append(torch.cat((tokens[b, :thought_pos], tokens[b, solution_pos:])))
                        pruned_any = True
                        continue
                new_rows.append(tokens[b])

            if pruned_any:
                max_len = max(r.shape[0] for r in new_rows)
                tokens = torch.full((batch_size, max_len), eos_token_id, dtype=tokens.dtype, device=tokens.device)
                attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
                for b, r in enumerate(new_rows):
                    tokens[b, :r.shape[0]] = r
                    attention_mask[b, :r.shape[0]] = 1

    wall_time = time.perf_counter() - start_time
    output_tokens = attention_mask.sum(dim=1).tolist()
    return HCotGenerateOutput(
        sequences=tokens,
        prompt_tokens=prompt_tokens,
        generated_tokens=num_generated,
        total_tokens_processed=total_tokens_processed,
        output_tokens=output_tokens,
        prune_events=prune_events,
        tokens_pruned=tokens_pruned,
        wall_time_seconds=wall_time,
    )


def generate_with_mask(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    return_token_id,
    solution_token_id,
    thought_token_id,
    eos_token_id,
    min_token_length,
    **model_kwargs
):
    """Generate with hierarchical masking below min_token_length, pruning above.

    Until the sequence reaches min_token_length tokens, completed
    [THOUGHT]...[SOLUTION]...[RETURN] blocks are hidden via a 4-D attention
    mask (identical to MaterialisedMaskMixin) instead of being pruned away.
    Once the sequence is long enough, new blocks are pruned as in ``generate``.
    """

    def _build_4d_mask(tokens_tensor, padding_mask_1d):
        """Build a 4-D float attention mask with hierarchical CoT blocking."""
        batch_size, seq_len = tokens_tensor.shape
        device = tokens_tensor.device

        batch_blocks = extract_cot_blocks(
            tokens_tensor, thought_token_id, solution_token_id, return_token_id
        )

        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        masks = []
        for b in range(batch_size):
            mask = causal.clone()
            for thought_pos, solution_pos, return_pos in batch_blocks[b]:
                mask[return_pos + 1:, thought_pos:solution_pos] = False
            mask = mask & padding_mask_1d[b].bool().unsqueeze(0)
            masks.append(mask)

        mask_tensor = torch.stack(masks).unsqueeze(1)
        float_mask = torch.zeros(
            batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device
        )
        float_mask.masked_fill_(~mask_tensor, torch.finfo(torch.bfloat16).min)
        return float_mask

    start_time = time.perf_counter()

    batch_size = input_ids.shape[0]
    tokens = input_ids.clone()
    prompt_tokens = input_ids.shape[1]
    total_tokens_processed = [0] * batch_size
    num_generated = 0
    prune_events = [0] * batch_size
    tokens_pruned = [0] * batch_size

    padding_mask = model_kwargs.pop('attention_mask')
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    has_active_blocks = False
    stacks = [[] for _ in range(batch_size)]
    max_token_length = model_kwargs.get("max_new_tokens", stopping_criteria[0].max_length)

    while tokens.shape[1] < max_token_length:
        for b in range(batch_size):
            total_tokens_processed[b] += tokens.shape[1]
        num_generated += 1

        if has_active_blocks:
            attn_mask = _build_4d_mask(tokens, padding_mask)
        else:
            attn_mask = padding_mask

        outputs = model(tokens, attention_mask=attn_mask, **model_kwargs)
        logits = outputs.logits
        next_token_logits = logits_processor(tokens, logits[:, -1, :])

        next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
        tokens = torch.cat((tokens, next_tokens), dim=-1)
        padding_mask = torch.cat((padding_mask, torch.ones_like(next_tokens)), dim=-1)

        # Check for EOS
        unfinished = unfinished & (next_tokens.squeeze(-1) != eos_token_id)
        if not unfinished.any():
            break

        # Honour all stopping criteria (e.g. MaxTimeCriteria, custom callbacks)
        if stopping_criteria(tokens, next_token_logits).all():
            break

        sq = next_tokens.squeeze(-1)
        pos = tokens.shape[1] - 1  # index of the just-appended token

        # Track [THOUGHT] and [SOLUTION] tokens — sparse iteration over active elements only
        for b in (sq == thought_token_id).nonzero(as_tuple=True)[0].tolist():
            stacks[b].append([pos, None])

        for b in (sq == solution_token_id).nonzero(as_tuple=True)[0].tolist():
            if stacks[b]:
                stacks[b][-1][1] = pos

        # Handle [RETURN] token
        return_indices = (sq == return_token_id).nonzero(as_tuple=True)[0].tolist()
        if return_indices:
            if tokens.shape[1] < min_token_length:
                # Below threshold: mask the block instead of pruning
                has_active_blocks = True
            else:
                # Above threshold: prune the block
                new_rows = []
                pruned_any = False
                for b in range(batch_size):
                    if b in return_indices and stacks[b]:
                        thought_pos, solution_pos = stacks[b].pop()
                        if solution_pos is not None:
                            prune_events[b] += 1
                            tokens_pruned[b] += solution_pos - thought_pos
                            new_rows.append(torch.cat((tokens[b, :thought_pos], tokens[b, solution_pos:])))
                            pruned_any = True
                            continue
                    new_rows.append(tokens[b])

                if pruned_any:
                    max_len = max(r.shape[0] for r in new_rows)
                    tokens = torch.full((batch_size, max_len), eos_token_id, dtype=tokens.dtype, device=tokens.device)
                    padding_mask = torch.zeros((batch_size, max_len), dtype=padding_mask.dtype, device=padding_mask.device)
                    for b, r in enumerate(new_rows):
                        tokens[b, :r.shape[0]] = r
                        padding_mask[b, :r.shape[0]] = 1

                    # Check if earlier masked blocks still remain
                    remaining = extract_cot_blocks(
                        tokens, thought_token_id, solution_token_id, return_token_id
                    )
                    has_active_blocks = any(len(b) > 0 for b in remaining)

    wall_time = time.perf_counter() - start_time
    output_tokens = padding_mask.sum(dim=1).tolist()
    return HCotGenerateOutput(
        sequences=tokens,
        prompt_tokens=prompt_tokens,
        generated_tokens=num_generated,
        total_tokens_processed=total_tokens_processed,
        output_tokens=output_tokens,
        prune_events=prune_events,
        tokens_pruned=tokens_pruned,
        wall_time_seconds=wall_time,
    )


def generate_standard(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    **model_kwargs
):
    """Wrapper around HF's standard generate that returns HCotGenerateOutput with stats.

    Computes total_tokens_processed analytically assuming no KV-cache:
    each step i processes (prompt_len + i) tokens, so the total is
    sum_{i=1}^{k} (n + i) = k*n + k*(k+1)/2  where n = prompt length, k = generated tokens.
    """
    start_time = time.perf_counter()

    prompt_tokens = input_ids.shape[1]
    use_cache = model_kwargs.get("use_cache", True)

    output = model.generate(
        input_ids=input_ids,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        **model_kwargs,
    )

    wall_time = time.perf_counter() - start_time

    # output is a plain LongTensor from HF generate (return_dict_in_generate defaults to False)
    if hasattr(output, 'sequences'):
        sequences = output.sequences
    else:
        sequences = output

    batch_size = sequences.shape[0]
    output_len = sequences.shape[1]
    generated_tokens = output_len - prompt_tokens

    if use_cache:
        # With KV-cache, each step processes only 1 new token (after the first full-prompt pass)
        per_record_total = prompt_tokens + generated_tokens
    else:
        # Without KV-cache, step i processes (prompt_tokens + i) tokens
        # sum_{i=1}^{k} (n + i) = k*n + k*(k+1)//2
        k = generated_tokens
        n = prompt_tokens
        per_record_total = k * n + k * (k + 1) // 2

    return HCotGenerateOutput(
        sequences=sequences,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_tokens_processed=[per_record_total] * batch_size,
        output_tokens=[output_len] * batch_size,
        prune_events=[0] * batch_size,
        tokens_pruned=[0] * batch_size,
        wall_time_seconds=wall_time,
    )
