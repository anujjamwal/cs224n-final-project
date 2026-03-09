"""Reward functions to use in GRPO Trainer."""

from dataclasses import dataclass
from typing import Tuple

from trainer import THOUGHT_TOKEN, SOLUTION_TOKEN, RETURN_TOKEN
from eval.benchmarks import extract_boxed_last, check_answer_math_verify


def _get_completion_text(completion) -> str:
    """Extract plain text from a TRL completion (conversational format)."""
    if isinstance(completion, list):
        # conversational: [{role: ..., content: ...}]
        return completion[0]["content"] if completion else ""
    return str(completion)

def correctness_reward(completions, expected_answer, **kwargs) -> list[float]:
    """1.0 if the \\boxed{} answer matches the expected answer, else 0.0.

    Uses ``math-verify`` for robust mathematical comparison with a
    normalised-string fallback.
    """
    rewards = []
    for completion, answer in zip(completions, expected_answer):
        text = _get_completion_text(completion)
        predicted = extract_boxed_last(text)
        correct = check_answer_math_verify(predicted, str(answer))
        rewards.append(1.0 if correct else 0.0)
    return rewards

def build_syntax_reward(tokenizer, threshold: int = 4):
    thought_token_id = tokenizer.convert_tokens_to_ids('[THOUGHT]')
    solution_token_id = tokenizer.convert_tokens_to_ids('[SOLUTION]')
    return_token_id = tokenizer.convert_tokens_to_ids('[RETURN]')
    think_start_token_id = tokenizer.convert_tokens_to_ids('<think>')
    think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')
    def syntax_reward(completion_ids, **kwargs) -> list[float]:
        rewards = [0.0] * len(completion_ids)

        for i, completion in enumerate(completion_ids):
            stack = []
            block_count = 0
            for idx, token in enumerate(completion):
                if token == thought_token_id:
                    if stack and stack[-1][1]:
                        block_count += 1
                    else:
                        stack.append([idx, None])
                elif token == solution_token_id:
                    if stack and not stack[-1][1]:
                        stack[-1][1] = idx
                    else:
                        block_count += 1
                elif token == return_token_id and stack:
                    tid, sid = stack.pop()
                    if sid:
                      rewards[i] += 1
                    block_count += 1
                elif token == think_start_token_id:
                    block_count += 1
                    if not stack:
                        rewards[i] += 1
                elif token == think_end_token_id:
                    block_count += 1
                    if not stack:
                        rewards[i] += 1
            if block_count < 1:
                rewards[i] = 0
            else:
                count = block_count - rewards[i]
                rewards[i] /= block_count
                rewards[i] = rewards[i] ** count

        return rewards
    return syntax_reward

def build_short_thought_reward(tokenizer, threshold: int = 4):
    thought_token_id = tokenizer.convert_tokens_to_ids('[THOUGHT]')
    solution_token_id = tokenizer.convert_tokens_to_ids('[SOLUTION]')
    return_token_id = tokenizer.convert_tokens_to_ids('[RETURN]')
    def short_thoughts_reward(completion_ids, **kwargs) -> list[float]:
        rewards = [0.0] * len(completion_ids)

        for i, completion in enumerate(completion_ids):
            stack = []
            block_count = 0
            for idx, token in enumerate(completion):
                if token == thought_token_id:
                    stack.append([idx, None, 0])
                elif token == solution_token_id and stack:
                    stack[-1][1] = idx
                elif token == return_token_id and stack:
                    tid, sid, pruned = stack.pop()
                    if sid:
                        block_count += 1
                        if sid - tid - pruned < threshold:
                            rewards[i] += 1
                        if stack:
                            stack[-1][2] += (sid - tid)

            if block_count < 1:
                rewards[i] = 0
            else:
                count = block_count - rewards[i]
                rewards[i] /= block_count
                rewards[i] = rewards[i] ** count

        return rewards
  
    return short_thoughts_reward

def build_depth_reward(tokenizer, threshold: int = 4):
    thought_token_id = tokenizer.convert_tokens_to_ids('[THOUGHT]')
    return_token_id = tokenizer.convert_tokens_to_ids('[RETURN]')
    def depth_reward(completion_ids, **kwargs) -> list[float]:
        rewards = [0.0] * len(completion_ids)
        for i, completion in enumerate(completion_ids):
            max_depth = 0
            depth = 0
            for token in completion:
                if token == thought_token_id:
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif token == return_token_id:
                    depth -= 1

            if max_depth <= threshold:
                rewards[i] = 1
            else:
                rewards[i] = max(0.0, 1.0 - (max_depth - threshold) / 2)
        return rewards

    return depth_reward

def build_compression_reward(tokenizer):
    thought_token_id = tokenizer.convert_tokens_to_ids('[THOUGHT]')
    solution_token_id = tokenizer.convert_tokens_to_ids('[SOLUTION]')
    return_token_id = tokenizer.convert_tokens_to_ids('[RETURN]')
    def compression_reward(completion_ids, **kwargs) -> list[float]:
        rewards = [0.0] * len(completion_ids)

        for i, completion in enumerate(completion_ids):
            stack = []
            total_thought = 0
            total_solution = 0
            for idx, token in enumerate(completion):
                if token == thought_token_id:
                    stack.append([idx, None, 0])
                elif token == solution_token_id and stack:
                    stack[-1][1] = idx
                elif token == return_token_id and stack:
                    tid, sid, pruned = stack.pop()
                    if sid:
                        total_thought += sid - tid - pruned
                        total_solution += idx - sid
                        if stack:
                            stack[-1][2] += (sid - tid)
            if total_thought < 1:
                rewards[i] = 0
            else:
                ratio = total_solution / total_thought
                if 0.1 <= ratio <= 0.5:
                    rewards[i] = 1.0
                elif ratio < 0.1:
                    # Linear decay from 1.0 at 0.1 to 0.0 at 0.0
                    rewards[i] = max(0.0, ratio / 0.1)
                else:
                    # Linear decay from 1.0 at 0.5 to 0.0 at 1.0
                    rewards[i] = max(0.0, 1.0 - (ratio - 0.5) / 0.5)

        return rewards
  
    return compression_reward

def format_reward(completions, **kwargs) -> list[float]:
    rewards = [0.0] * len(completions)
    for i, completion in enumerate(completions):
        text = _get_completion_text(completion)
        think_end_idx = text.rfind('</think>')
        think_start_idx = text.find('<think>')
        boxed_idx = text.rfind('\\boxed{')
        first_thought_idx = text.find('[THOUGHT]')
        last_return_idx = text.rfind('[RETURN]')
        if think_start_idx > -1 and think_end_idx > -1 and think_start_idx < think_end_idx:
            rewards[i] += 1

        if boxed_idx > -1:
            rewards[i] += 1

        
        if boxed_idx > think_end_idx and think_end_idx > -1:
            rewards[i] += 1
        
        if text.rfind("\\boxed{") > text.rfind('</think>'):
            rewards[i] += 1

        if first_thought_idx > -1 and \
            first_thought_idx > think_start_idx and \
            last_return_idx > -1 and \
            last_return_idx < think_end_idx:
            rewards[i] += 1

        rewards[i] /= 5
    return rewards
