import torch


def find_cot_blocks(input_ids, start_token_id, mid_token_id, end_token_id):
    """Find the COT blocks in batched inputs

    The blocks are structured as <start-token> .. <mid-token> .. <end-token>
    """

    batch_size = input_ids.shape[0]
    batch_blocks = []
    for b in range(batch_size):
        tokens = input_ids[b].tolist()
        stack, blocks = [], []
        for i, tok in enumerate(tokens):
            if tok == start_token_id:
                stack.append([i, None])
            elif tok == mid_token_id and stack:
                stack[-1][1] = i
            elif tok == end_token_id and stack:
                thought_pos, solution_pos = stack.pop()
                if solution_pos is not None:
                    blocks.append((thought_pos, solution_pos, i))
        batch_blocks.append(blocks)
    return batch_blocks


def build_stages(input_ids_b, labels_b, attention_mask_b, blocks):
    """Build staged pruned sequences for a single batch element.

    Simulates inference execution where thought content is physically pruned
    at each [RETURN] token. Each stage represents a forward pass the model
    would do after a pruning event.

    Stage 0: full sequence, loss for tokens up to first [RETURN]
    Stage k: k blocks pruned, loss for tokens between k-th and (k+1)-th [RETURN]
    Last stage: all blocks pruned, loss for tokens after last [RETURN]

    Args:
        input_ids_b: (seq_len,) tensor of token IDs (single element, no batch dim)
        labels_b: (seq_len,) tensor of labels
        attention_mask_b: (seq_len,) tensor of attention mask (1=real, 0=pad)
        blocks: list of (thought_pos, solution_pos, return_pos) tuples

    Returns:
        List of (pruned_input_ids, pruned_labels, pruned_attention_mask) tuples.
        Stages with no loss-contributing tokens are omitted.
    """
    sorted_blocks = sorted(blocks, key=lambda b: b[2])  # sort by return_pos
    seq_len = input_ids_b.shape[0]
    n_blocks = len(sorted_blocks)
    device = input_ids_b.device

    stages = []

    for stage_idx in range(n_blocks + 1):
        # Positions to remove: thought content from all blocks pruned before this stage
        remove_positions: set[int] = set()
        for k in range(stage_idx):
            tp, sp, _rp = sorted_blocks[k]
            for i in range(tp + 1, sp + 1):
                remove_positions.add(i)

        # Keep non-removed, non-padding positions
        kept = [p for p in range(seq_len)
                if p not in remove_positions and attention_mask_b[p]]

        if not kept:
            continue

        kept_t = torch.tensor(kept, dtype=torch.long, device=device)
        pruned_ids = input_ids_b[kept_t]
        pruned_mask = torch.ones(len(kept), dtype=attention_mask_b.dtype, device=device)

        # Loss range in original positions
        loss_start = 0 if stage_idx == 0 else sorted_blocks[stage_idx - 1][2] + 1
        loss_end = sorted_blocks[stage_idx][2] if stage_idx < n_blocks else seq_len - 1

        # Labels: keep original label only within the loss range for this stage
        pruned_labels = torch.full_like(pruned_ids, -100)
        for j, orig_p in enumerate(kept):
            if loss_start <= orig_p <= loss_end:
                pruned_labels[j] = labels_b[orig_p]

        if (pruned_labels != -100).any():
            stages.append((pruned_ids, pruned_labels, pruned_mask))

    return stages
