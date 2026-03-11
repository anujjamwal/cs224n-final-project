
import argparse
import logging
import os
import re

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types

import tenacity


logger = logging.getLogger(__name__)

PERSONA = "You are a research assistant that is helping me prepare a dataset for training a new kind of reasoning LLM."

PROMPT = """
## Background
Let me give you some background on my research. Traditionally, CoT are treated as long append-only structure which keeps growing
as the LLM attemps to solve the problem at hand. However, this structure is very inefficient and resource heavy due to all the
redundant information in the thoughts.

If we comb through the chain of thought, we realise that the LLM actually  reasons in a hierarchical manner where it breaks down
the problem into subproblems, solves one problem at a time and then combines the solutions as it moved up the hierarchy of
reasoning. The chain of thought is a flattened out version of this hierarchical reasoning process. Each step in the hierarchy of
reasoning can be represented as below, hereto referred to as `sub-cot`.

```
[THOUGHT] chain_of_thought [SOLUTION] solution [RETURN]
```

where:
- [THOUGHT], [SOLUTION] and [RETURN] are special tokens that mark the beginning of the chain of thought, the solution and the
return to the parent problem respectively.
- chain_of_thought: the reasoning process that leads to the solution of this sub problem.
- solution: the final answer to this sub problem and a concise summary of the step that captures the key idea or insight.

Complete hierarchical chain of thought can look something like below. Ignore the indentation. These obey the following rules
- There is always a root sub-cot
- `sub-cot` can only be added to [THOUGHT] section of parent
- Tree can be any depth deep
- sub-cot must always be complete, that is, both the sections must be present. 

```
[THOUGHT] <root thoughts> 
    [THOUGHT] sub problem 1
        [THOUGHT]  sub problem 1.1 [SOLUTION] solution 1.1 [RETURN]
        [THOUGHT]  sub problem 1.2
            [THOUGHT]  sub problem 1.2.1 [SOLUTION] solution 1.2.1 [RETURN]
            ...
        [SOLUTION] solution 1.2 [RETURN]
        ...
    [SOLUTION] solution 1 [RETURN]
    ...
    [THOUGHT sub problem n]
        ...
    [SOLUTION] solution n [RETURN]
[SOLUTION] solution [RETURN]
```

Once the LLM has solved a subproblem, it can discard the chain of thought and replace it with the `solution` which should be
sufficient for the future steps. This way, we can prune the chain of thought at inference time allowing the LLM to think very
deeply when needed and utilize the solutions to subproblems without being contrained by the length context window. Alternatively, when the LLM thinks that the current chain of thought is redundant, it can "backtrack" by producing `solution`
calling out what it has tried and successively moving up the thought branch until it wants to begin solving the new branch of thoughts. 

In this model, the autoregressive loop works with the following pseudocode:

```
def generate(input):
    tokens = tokenize(input)
    while not done:
        next_token = model(tokens)

        if next_token == [RETURN]:
            # we have reached the end of a step in the hierarchy of reasoning
            # we can prune the chain of thought and replace it with the solution
            tokens, chain_of_thought = remove_chain_of_thought_tokens(tokens)
            tokens.append(next_token)

```

## Process

I see the data preparation task as two step activity. We have already completed first stage of activity where we break the COT into hierarchical COT without performing backtracking.

## Task

You will be given a problem statement, a hierarchical chain of thought and the final solution. Your task is to review the provided hierarchical chain of thought, find redundant branches of reasoning especially where the model realises half way that it should change course and simulate backtracking. It might require you to wrap the branches of reasoning in nested thought
blocks or just produce solution blocks for exisiting thought blocks to successively "backtrack" along the chain of reasoning. It is very important to remember that we are segmenting the chain of thought not writing a new one.

Follow these strict rules:
1. Use thinking as the foundation for your segmentation. Do not replace or rewrite the content of the original chain of thought.
2. Preserve the original wording. Do not paraphase, rewrite or reorder the content of the chain of thought as you segment those
into chain of thoughts for the subproblems.
3. Do not add headings or any other text that is not present in the original chain of thought. The only addition you can make is
to add the special tokens and to the solution section if the original chain of thought doesn't have a clear wording for it.
4. Ensure the final solution has the solution in the format requested in the prompt template. This will be in the last few lines.
5. Prefer deeper chain of thoughts but limit upto 5 levels deep.

At the end produce a summary of the steps and verify that the generated hierarchical cot is valid.

## Output Format

Output the produced hierarchical COT wrapped in <hierarchical-cot> </hierarchical-cot> tags.
"""
INPUT="""
## Inputs

### Prompt Template

Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\\n\\n{{problem}}

### Problem Statement

{problem_statement}

### Final Solution

{final_solution}

### Hierarchical Chain of Thought

{chain_of_thought}
"""

MAX_TOKENS = 16000
THINKING_BUDGET = 10000
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs-cleanup")
DEFAULT_PARALLELISM = int(os.environ.get("PARALLELISM", "4"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleanup the redundant chain of thought and course corrections"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier passed to the API or CLI",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index into the source dataset (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of source records to consider",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=DEFAULT_PARALLELISM,
        help=f"Parallel workers for CLI method (default: {DEFAULT_PARALLELISM})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="anujjamwal/OpenMathReasoning-Sampled-Hierarchical-Cot",
        help=f"The huggingface dataset with hcot",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="anujjamwal/OpenMathReasoning-Sampled-Hierarchical-Cot-cleaned",
        help=f"The target huggingface dataset to push to",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to cache LLM outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def parse_result(result):
    match = re.search(r'<hierarchical-cot>(.*?)</hierarchical-cot>', result, re.DOTALL)
    if match:
        hierarchical_cot = match.group(1).strip()
    else:
        hierarchical_cot = result
    return hierarchical_cot, result


def _is_gemini_retryable_error(e):
    err_str = str(e)
    return "429" in err_str or "ResourceExhausted" in err_str


@tenacity.retry(retry=tenacity.retry_if_exception(_is_gemini_retryable_error),
                wait=tenacity.wait_exponential_jitter(initial=30, jitter=30),
                stop=tenacity.stop_after_attempt(10))
def call_gemini(prompt, model, max_tokens=MAX_TOKENS):
    if genai is None:
        raise ImportError("The 'google-genai' package is not installed. Please install it with 'pip install google-genai' to use Gemini models via API.")
    client = genai.Client()
    
    response = client.models.generate_content(
        model=model,
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=PERSONA),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=PROMPT),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH
            )
        )
    )
    return response.text


def cleanup_chain_of_thought(problem_statement, chain_of_thought, final_solution, model=None, output_file=None):
    if output_file and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                logger.info(f"Found cached content in {output_file}. Skipping API invocation.")
                return parse_result(content)

    if model is None:
        model = os.environ.get("MODEL", "claude-sonnet-4-6")

    prompt = INPUT.format(
        problem_statement=problem_statement,
        chain_of_thought=chain_of_thought,
        final_solution=final_solution,
    )
    result = call_gemini(prompt, model=model)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result or "")

    return parse_result(result)


def process_with_api(examples, model, parallelism, output_dir):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    def _segment(x, idx):
        output_file = os.path.join(output_dir, f"example_{x["id"]}.txt")
        x["hcot_model"] = model
        try:
            x["hierarchical_cot"], x["hierarchical_cot_raw"] = cleanup_chain_of_thought(
                x["question"], x["hierarchical_cot"], x["expected_answer"], model=model, output_file=output_file
            )
        except Exception as e:
            logger.error("Failed to segment problem %r (index %s): %s", x["question"], x["id"], e)
            x["hierarchical_cot"], x["hierarchical_cot_raw"] = "", ""
        
        return x

    return examples.map(_segment, with_indices=True, num_proc=parallelism)


def main():
    load_dotenv()
    args = parse_args()

    source = load_dataset(args.source, split="train", streaming=True)
    records = list(source.skip(args.offset).take(args.limit))
    source = Dataset.from_list(records)

    newly_processed = process_with_api(source, args.model, args.parallelism, args.output_dir)
    newly_processed.push_to_hub(args.target)


if __name__ == "__main__":
    main()