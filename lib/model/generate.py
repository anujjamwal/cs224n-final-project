import torch
from transformers import LogitsProcessorList, StoppingCriteriaList
from .masks import extract_cot_blocks

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
    def _prune_cot_single(seq):
        """Prune the most recently completed [THOUGHT]...[SOLUTION]...[RETURN] block.

        Uses stack-based matching (consistent with extract_cot_blocks in masks.py)
        to correctly handle nested CoT structures. Only prunes the block closed
        by the trailing [RETURN] token (the last token in seq).
        """
        # Stack-based scan to find all complete blocks
        stack = []  # entries: [thought_pos, solution_pos]
        blocks = []  # complete (thought_pos, solution_pos, return_pos) triplets
        for i, tok in enumerate(seq):
            if tok == thought_token_id:
                stack.append([i, None])
            elif tok == solution_token_id and stack:
                stack[-1][1] = i
            elif tok == return_token_id and stack:
                thought_pos, solution_pos = stack.pop()
                if solution_pos is not None:
                    blocks.append((thought_pos, solution_pos, i))

        if not blocks:
            return seq

        # Prune the last completed block (the one just closed by the trailing [RETURN])
        thought_idx, solution_idx, _ = blocks[-1]
        print(f"Pruning COT on {thought_idx} - {solution_idx}")
        return seq[:thought_idx] + seq[solution_idx:]

    batch_size = input_ids.shape[0]
    tokens = input_ids.clone()
    # Track which batch elements are still generating
    attention_mask = model_kwargs.pop('attention_mask')
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    max_token_length = model_kwargs.get("max_new_tokens", stopping_criteria[0].max_length)

    while tokens.shape[1] < max_token_length:
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

        # Prune CoT per batch element that generated [RETURN]
        return_mask = next_tokens.squeeze(-1) == return_token_id
        if return_mask.any():
            print("FOUND token initiate prune")
            token_lists = [tokens[b].tolist() for b in range(batch_size)]
            for b in range(batch_size):
                if return_mask[b]:
                    token_lists[b] = _prune_cot_single(token_lists[b])

            # Pad to equal length and rebuild tensors
            max_len = max(len(s) for s in token_lists)
            padded = []
            new_mask = []
            for s in token_lists:
                pad_len = max_len - len(s)
                padded.append(s + [eos_token_id] * pad_len)
                new_mask.append([1] * len(s) + [0] * pad_len)

            tokens = torch.tensor(padded, dtype=tokens.dtype, device=tokens.device)
            attention_mask = torch.tensor(new_mask, dtype=attention_mask.dtype, device=attention_mask.device)
  
    return tokens


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

    def _prune_cot_single(seq):
        stack = []
        blocks = []
        for i, tok in enumerate(seq):
            if tok == thought_token_id:
                stack.append([i, None])
            elif tok == solution_token_id and stack:
                stack[-1][1] = i
            elif tok == return_token_id and stack:
                thought_pos, solution_pos = stack.pop()
                if solution_pos is not None:
                    blocks.append((thought_pos, solution_pos, i))
        if not blocks:
            return seq
        thought_idx, solution_idx, _ = blocks[-1]
        return seq[:thought_idx] + seq[solution_idx:]

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

    batch_size = input_ids.shape[0]
    tokens = input_ids.clone()
    padding_mask = model_kwargs.pop('attention_mask')
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    has_active_blocks = False
    max_token_length = model_kwargs.get("max_new_tokens", stopping_criteria[0].max_length)

    while tokens.shape[1] < max_token_length:
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

        # Handle [RETURN] token
        return_mask = next_tokens.squeeze(-1) == return_token_id
        if return_mask.any():
            if tokens.shape[1] < min_token_length:
                # Below threshold: mask the block instead of pruning
                has_active_blocks = True
            else:
                # Above threshold: prune the block
                token_lists = [tokens[b].tolist() for b in range(batch_size)]
                for b in range(batch_size):
                    if return_mask[b]:
                        token_lists[b] = _prune_cot_single(token_lists[b])

                max_len = max(len(s) for s in token_lists)
                padded = []
                new_mask = []
                for s in token_lists:
                    pad_len = max_len - len(s)
                    padded.append(s + [eos_token_id] * pad_len)
                    new_mask.append([1] * len(s) + [0] * pad_len)

                tokens = torch.tensor(padded, dtype=tokens.dtype, device=tokens.device)
                padding_mask = torch.tensor(new_mask, dtype=padding_mask.dtype, device=padding_mask.device)

                # Check if earlier masked blocks still remain
                remaining = extract_cot_blocks(
                    tokens, thought_token_id, solution_token_id, return_token_id
                )
                has_active_blocks = any(len(b) > 0 for b in remaining)

    return tokens
