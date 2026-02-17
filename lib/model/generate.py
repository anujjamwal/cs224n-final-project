import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

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
        """Prune a single 1-D token sequence in-place (as a list).

        Finds the last [THOUGHT] and its matching [SOLUTION], removes everything
        between them (inclusive of [THOUGHT], exclusive of [SOLUTION]).
        """
        # Find the last [THOUGHT] token
        thought_idx = None
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] == thought_token_id:
                thought_idx = i
                break

        if thought_idx is None:
            return seq

        # Find the last [SOLUTION] token after that [THOUGHT]
        solution_idx = None
        for i in range(len(seq) - 1, thought_idx, -1):
            if seq[i] == solution_token_id:
                solution_idx = i
                break

        if solution_idx is None:
            return seq

        return seq[:thought_idx] + seq[solution_idx:]

    batch_size = input_ids.shape[0]
    tokens = input_ids.clone()
    # Track which batch elements are still generating
    attention_mask = model_kwargs.pop('attention_mask')
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

    while tokens.shape[1] < stopping_criteria[0].max_length:
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

        # Prune CoT per batch element that generated [RETURN]
        return_mask = next_tokens.squeeze(-1) == return_token_id
        if return_mask.any():
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

