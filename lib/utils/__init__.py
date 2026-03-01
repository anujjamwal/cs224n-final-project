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
