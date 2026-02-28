import torch
from torch import nn


def extract_cot_blocks(input_ids, thought_token_id, solution_token_id, return_token_id):
    """Find [THOUGHT]-[SOLUTION]-[RETURN] triplets in each batch element.

    Returns a list (length = batch_size) of lists of (thought_pos, solution_pos,
    return_pos) tuples.
    """
    batch_size = input_ids.shape[0]
    batch_blocks = []
    for b in range(batch_size):
        tokens = input_ids[b].tolist()
        stack, blocks = [], []
        for i, tok in enumerate(tokens):
            if tok == thought_token_id:
                stack.append([i, None])
            elif tok == solution_token_id and stack:
                stack[-1][1] = i
            elif tok == return_token_id and stack:
                thought_pos, solution_pos = stack.pop()
                if solution_pos is not None:
                    blocks.append((thought_pos, solution_pos, i))
        batch_blocks.append(blocks)
    return batch_blocks


def build_min_blocked_q(input_ids, batch_blocks):
    """Build the min_blocked_q tensor used by both mask paths.

    For each key position k, stores the first query index that is blocked from
    attending to k.  ``seq_len`` means "never blocked" (acts as +infinity).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    min_blocked_q = torch.full(
        (batch_size, seq_len), seq_len, dtype=torch.long, device=device
    )
    for b, blocks in enumerate(batch_blocks):
        for thought_pos, solution_pos, return_pos in blocks:
            # By reduction rule we maintain [THOUGHT] and [RETURN] tokens.
            span = min_blocked_q[b, thought_pos+1:solution_pos+1]
            threshold = torch.full_like(span, return_pos + 1)
            min_blocked_q[b, thought_pos+1:solution_pos+1] = torch.minimum(span, threshold)
    return min_blocked_q


def build_position_ids(input_ids, batch_blocks, padding_mask=None):
    """Build position_ids that simulate contiguous positions after pruning.

    During training the hierarchical mask hides the reasoning span
    (thought_pos+1 … solution_pos) from tokens after the corresponding
    [RETURN].  But the default sequential position_ids still include the
    hidden span, creating a train-test mismatch: at inference the pruned
    sequence has contiguous positions, so the RoPE relative distances
    differ from what the model saw during training.

    This function assigns position_ids as if each masked reasoning span
    has been physically removed.  Tokens **inside** the span keep their
    original positions (they still attend to each other during training).
    Tokens from solution_pos+1 onward are shifted back by the span length
    so their RoPE distances to visible tokens match what they would be
    after actual pruning.

    For multiple blocks, shifts accumulate in sequence order.

    When ``padding_mask`` is provided (1 = real token, 0 = pad), the base
    positions are derived from its cumulative sum so that padded tokens
    (whether left- or right-padded) do not consume position slots.
    """
    batch_size, seq_len = input_ids.shape

    if padding_mask is not None:
        position_ids = padding_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)
    else:
        position_ids = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1).clone()

    for b in range(batch_size):
        sorted_blocks = sorted(batch_blocks[b], key=lambda x: x[2])
        for thought_pos, solution_pos, return_pos in sorted_blocks:
            span_len = solution_pos - thought_pos
            position_ids[b, solution_pos + 1:] -= span_len

    return position_ids


class FlexAttentionMaskMixin:
    """Mixin that builds a hierarchical attention mask via FlexAttention.

    Requires ``torch.nn.attention.flex_attention.create_block_mask``.
    Uses O(seq_len) memory via block-sparse bitmaps.
    """

    def _build_hierarchical_mask(self, input_ids, padding_mask=None):
        from torch.nn.attention.flex_attention import create_block_mask

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        batch_blocks = extract_cot_blocks(
            input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
        )
        mbq = build_min_blocked_q(input_ids, batch_blocks)

        # create_block_mask rounds Q_LEN/KV_LEN up to a multiple of the block
        # size (default 128).  Pad mbq so out-of-bounds kv_idx values map to
        # seq_len (= "never blocked"), and pad the padding_mask with 0 (masked).
        BLOCK = 128
        padded_len = ((seq_len + BLOCK - 1) // BLOCK) * BLOCK
        if padded_len > seq_len:
            pad_cols = padded_len - seq_len
            mbq = torch.nn.functional.pad(mbq, (0, pad_cols), value=seq_len)
            if padding_mask is not None:
                padding_mask = torch.nn.functional.pad(padding_mask, (0, pad_cols), value=0)

        _mbq = mbq
        _pm = padding_mask

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            not_pruned = q_idx < _mbq[b, kv_idx]
            if _pm is not None:
                return causal & not_pruned & _pm[b, kv_idx].bool()
            return causal & not_pruned

        return create_block_mask(
            mask_mod,
            B=batch_size, H=None,
            Q_LEN=seq_len, KV_LEN=seq_len,
            device=device,
        )


class MaterialisedMaskMixin:
    """Mixin that builds a materialised 4-D float attention mask.

    Falls back to an explicit (batch, 1, seq_len, seq_len) mask tensor.
    Only feasible for short sequences due to O(seq_len^2) memory.
    """

    def _build_hierarchical_mask(self, input_ids, padding_mask=None):
        mask, _ = self._build_hierarchical_mask_and_position_ids(input_ids, padding_mask)
        return mask

    def _build_hierarchical_mask_and_position_ids(self, input_ids, padding_mask=None):
        """Return ``(attention_mask, position_ids)`` for training.

        The attention mask is the standard 4-D materialised causal mask with
        reasoning spans blocked.  The position_ids tensor assigns contiguous
        positions as if those spans had been pruned, aligning training-time
        RoPE distances with inference-time distances after actual pruning.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        batch_blocks = extract_cot_blocks(
            input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
        )

        # Create the default mask with lower triangle
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        masks = []
        for b in range(batch_size):
            mask = causal.clone()
            for thought_pos, solution_pos, return_pos in batch_blocks[b]:
                # By reduction rule we maintain [THOUGHT] and [RETURN] tokens.
                mask[return_pos + 1:, thought_pos+1:solution_pos+1] = False
            if padding_mask is not None:
                mask = mask & padding_mask[b].bool().unsqueeze(0)
            masks.append(mask)

        mask_tensor = torch.stack(masks).unsqueeze(1)
        float_mask = torch.zeros(
            batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device
        )
        float_mask.masked_fill_(~mask_tensor, torch.finfo(torch.bfloat16).min)

        position_ids = build_position_ids(input_ids, batch_blocks, padding_mask)
        return float_mask, position_ids
