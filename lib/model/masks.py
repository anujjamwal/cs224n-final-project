import torch
from torch import nn
import utils


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


class FlexAttentionMaskMixin:
    """Mixin that builds a hierarchical attention mask via FlexAttention.

    Requires ``torch.nn.attention.flex_attention.create_block_mask``.
    Uses O(seq_len) memory via block-sparse bitmaps.
    """

    def _build_hierarchical_mask(self, input_ids, padding_mask=None):
        from torch.nn.attention.flex_attention import create_block_mask

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        batch_blocks = utils.find_cot_blocks(
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

    def _build_hierarchical_mask(self, input_ids, padding_mask):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        batch_blocks = utils.find_cot_blocks(
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

            mask = mask & padding_mask[b].bool().unsqueeze(0)
            masks.append(mask)

        mask_tensor = torch.stack(masks).unsqueeze(1)
        float_mask = torch.zeros(
            batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device
        )
        float_mask.masked_fill_(~mask_tensor, torch.finfo(torch.bfloat16).min)
        return float_mask
