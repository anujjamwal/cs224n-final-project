from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

import utils

def convert_to_trl(example, think_key="hierarchical_cot", output_key="expected_answer", question_key="question"):
    prompt = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."
    assistant_content = f"<think>\n{example[think_key]}\n</think>\n\\boxed{{{example[output_key]}}}"
    
    return {
        "prompt": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": example[question_key]},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def prepare_prune_aware(
    batch, 
    tokenizer: PreTrainedTokenizer, 
    thought_token="[THOUGHT]", 
    return_token="[RETURN]", 
    solution_token="[SOLUTION]",
    mask=-100
) -> dict[str, Any]:
    """Take a batch of input sequences and breaks them into stages.
    
    In order to perform prune aware training, we need to split the dataset
    into batches based on the location of [RETURN] tokens. In order to
    ensure that the model doesn't over index on the loss for the already
    seen tokens, we set their label to -100 so the loss is not computed on
    these."""

    new_input_ids = []
    new_attention_masks = []
    new_labels = []

    for i in range(len(batch["prompt"])):
        # Convert to messages ensuring completion is at the end of list
        messages = batch["prompt"][i] + batch["completion"][i]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True
        )

        input_ids = torch.tensor(tokenized['input_ids'])
        attention_mask = torch.tensor(tokenized['attention_mask'])
        labels = input_ids.clone()

        # Mask prompt
        prompt_tokenized = tokenizer.apply_chat_template(
            batch["prompt"][i],
            tokenize=True,
            add_generation_prompt=True
        )
        prompt_len = len(prompt_tokenized)
        labels[:prompt_len] = mask

        # Prepare for stages from the hierarchical COT
        batch_blocks = utils.find_cot_blocks(
            input_ids.unsqueeze(0),
            tokenizer.convert_tokens_to_ids(thought_token),
            tokenizer.convert_tokens_to_ids(solution_token),
            tokenizer.convert_tokens_to_ids(return_token)
        )[0]

        stages = utils.build_stages(input_ids, labels, attention_mask, batch_blocks)

        for stage_ids, stage_labs, stage_mask in stages:
            new_input_ids.append(stage_ids.tolist())
            new_attention_masks.append(stage_mask.tolist())
            new_labels.append(stage_labs.tolist())

    return {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_masks,
        "labels": new_labels
    }
        