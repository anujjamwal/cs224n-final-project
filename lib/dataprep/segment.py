import logging
import anthropic
import os
import subprocess
import time
import concurrent.futures
import tenacity
import re
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

CLAUDE_PROMPT = """You are a research assistant that is helping me prepare a dataset for training a new kind of reasoning LLM.

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
deeply when needed and utilize the solutions to subproblems without being contrained by the length context window.

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

## Task

You will be given a problem statement, a chain of thought and the final solution. Your task is to segment the chain of thought
into steps in the hierarchical reasoning in the format mentioned above. Remember we are segmenting the chain of thought not
writing a new one.

Follow these strict rules:
1. Use thinking as the foundation for your segmentation. Do not replace or rewrite the content of the original chain of thought.
2. Preserve the original wording. Do not paraphase, rewrite or reorder the content of the chain of thought as you segment those
into chain of thoughts for the subproblems.
3. Do not add headings or any other text that is not present in the original chain of thought. The only addition you can make is
to add the special tokens and to the solution section if the original chain of thought doesn't have a clear wording for it.
4. Ensure the final solution has the solution in the format requested in the prompt template. This will be in the last few lines.

At the end produce a summary of the steps and verify that the generated hierarchical cot is valid.

## Output Format

Output the produced hierarchical COT wrapped in <hierarchical-cot> </hierarchical-cot> tags.
"""
CLAUDE_INPUT="""
## Inputs

### Prompt Template

Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\\n\\n{{problem}}

### Problem Statement

{problem_statement}

### Final Solution

{final_solution}

### Chain of Thought

{chain_of_thought}
"""

MAX_TOKENS = 16000
THINKING_BUDGET = 10000


def segment_chain_of_thought(problem_statement, chain_of_thought, final_solution, model=None):
    if model is None:
        model = os.environ.get("MODEL", "claude-sonnet-4-6")
    prompt = CLAUDE_INPUT.format(
        problem_statement=problem_statement,
        chain_of_thought=chain_of_thought,
        final_solution=final_solution,
    )
    result = call_llm_api(prompt, model=model)
    return parse_result(result)


# Keep old name as alias for backwards compatibility
segment_chain_of_thought_with_claude = segment_chain_of_thought


def call_llm_api(prompt, model):
    """Route to the correct LLM backend based on model name prefix."""
    if model.startswith("gemini-"):
        return call_gemini(prompt, model=model)
    elif model.startswith("claude-"):
        return call_claude(prompt, model=model)
    else:
        raise ValueError(f"Unknown model prefix in {model!r}. Model must start with 'claude-' or 'gemini-'.")


@tenacity.retry(retry=tenacity.retry_if_exception(anthropic.RateLimitError),
                wait=tenacity.wait_exponential_jitter(),
                stop=tenacity.stop_after_attempt(5))
def call_claude(prompt, model, max_tokens=MAX_TOKENS, thinking_budget=THINKING_BUDGET):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={
            "type": "enabled",
            "budget_tokens": thinking_budget,
        },
        messages=[
            {"role": "user", "content": CLAUDE_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


@tenacity.retry(retry=tenacity.retry_if_exception(lambda e: "429" in str(e) or "ResourceExhausted" in str(e)),
                wait=tenacity.wait_exponential_jitter(),
                stop=tenacity.stop_after_attempt(5))
def call_gemini(prompt, model, max_tokens=MAX_TOKENS):
    client = genai.Client()
    
    response = client.models.generate_content(
        model=model,
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH"
            )
        )
    )
    return response.text


# --- Claude CLI-based implementation ---

ALLOWED_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
}

CLI_MAX_RETRIES = int(os.environ.get("CLI_MAX_RETRIES", "5"))
CLI_TIMEOUT = int(os.environ.get("CLI_TIMEOUT", "900"))
DEFAULT_PARALLELISM = int(os.environ.get("PARALLELISM", "4"))


def call_llm_cli(prompt, model=None, output_file=None, max_retries=CLI_MAX_RETRIES, timeout=CLI_TIMEOUT):
    """Invoke the claude CLI to process a prompt. The system prompt and user prompt
    are concatenated and passed via stdin using the -p (print) flag.

    stdout and stderr are streamed directly to file pipes (output_file and
    output_file + ".err") when output_file is provided, otherwise to
    subprocess.PIPE. The subprocess inherits the current process environment
    so the claude CLI can access its stored login/session credentials.

    Safety notes:
    - The prompt is passed via stdin (not a CLI arg), so no shell escaping is needed.
    - The command is given as a list with shell=False (default), so the shell never
      interprets it; shlex.quote() would be wrong and harmful in that context.
    - The model is checked against ALLOWED_MODELS to catch misconfigured env variables
      before they reach the subprocess call.
    """
    if model is None:
        model = os.environ.get("CLAUDE_CLI_MODEL", "claude-opus-4-6")

    if model.startswith("gemini-"):
        raise ValueError(f"CLI method is not supported for Gemini models ({model!r}). Use --method api instead.")

    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown model {model!r}. Must be one of: {sorted(ALLOWED_MODELS)}")

    full_prompt = CLAUDE_PROMPT + "\n\n" + prompt

    for attempt in range(max_retries):
        stdout_fh = stderr_fh = None
        try:
            if output_file is not None:
                stdout_fh = open(output_file, "w", encoding="utf-8")
                stderr_fh = open(output_file + ".err", "w", encoding="utf-8")
                stdout_arg, stderr_arg = stdout_fh, stderr_fh
            else:
                stdout_arg, stderr_arg = subprocess.PIPE, subprocess.PIPE

            result = subprocess.run(
                ["claude", "-p", "--model", model],
                input=full_prompt,
                stdout=stdout_arg,
                stderr=stderr_arg,
                text=True,
                timeout=timeout,
                env=os.environ,
            )

            if result.returncode == 0:
                if output_file is not None:
                    stdout_fh.flush()
                    stdout_fh.close()
                    stderr_fh.close()
                    stdout_fh = stderr_fh = None
                    with open(output_file, encoding="utf-8") as f:
                        return f.read().strip()
                return result.stdout.strip()

            stderr_msg = f"see {output_file}.err" if output_file else result.stderr.strip()
            logger.warning(
                "Claude CLI non-zero exit (attempt %d/%d): %s",
                attempt + 1, max_retries, stderr_msg,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Claude CLI timed out (attempt %d/%d)", attempt + 1, max_retries)
        except Exception as e:
            logger.warning("Claude CLI exception (attempt %d/%d): %s", attempt + 1, max_retries, e)
        finally:
            if stdout_fh is not None:
                stdout_fh.close()
            if stderr_fh is not None:
                stderr_fh.close()

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Claude CLI failed after {max_retries} attempts")


def segment_chain_of_thought_with_claude_cli(problem_statement, chain_of_thought, final_solution, model=None, output_file=None):
    """Segment a single example using the Claude CLI."""
    prompt = CLAUDE_INPUT.format(
        problem_statement=problem_statement,
        chain_of_thought=chain_of_thought,
        final_solution=final_solution,
    )
    
    result = call_llm_cli(prompt, model=model, output_file=output_file)
    return result


def parse_result(result):
    match = re.search(r'<hierarchical-cot>(.*?)</hierarchical-cot>', result, re.DOTALL)
    if match:
        hierarchical_cot = match.group(1).strip()
    else:
        hierarchical_cot = result
    return hierarchical_cot, result


CLI_OUTPUT_DIR = os.environ.get("CLI_OUTPUT_DIR", "cli_outputs")


def process_examples_parallel(examples, parallelism=DEFAULT_PARALLELISM, model=None):
    """Process a list of examples in parallel using the Claude CLI.

    Each element of *examples* should be a dict with keys:
        - problem_statement
        - chain_of_thought
        - final_solution

    Each CLI call writes its output to CLI_OUTPUT_DIR/example_{idx}.txt.

    Returns a list of (example, result_or_exception) tuples in the same order
    as the input.
    """
    os.makedirs(CLI_OUTPUT_DIR, exist_ok=True)
    results = [None] * len(examples)

    def _process(idx, example):
        output_file = os.path.join(CLI_OUTPUT_DIR, f"example_{idx}.txt")

        if output_file and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    logger.info(f"Found content for {idx} in {output_file}. Skipping model invocation")
                    return idx, parse_result(content)
                else:
                    logger.warn(f"Found empty content for {idx} in {output_file}. Skipping model invocation")
                    return idx, ("", "")

        try:
            logger.info("Running CLI for examples %d", idx)
            output = segment_chain_of_thought_with_claude_cli(
                problem_statement=example["problem_statement"],
                chain_of_thought=example["chain_of_thought"],
                final_solution=example["final_solution"],
                model=model,
                output_file=output_file,
            )
            return idx, parse_result(output)
        except Exception as e:
            logger.error("Failed to process example %d: %s", idx, e)
            return idx, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {executor.submit(_process, i, ex): i for i, ex in enumerate(examples)}
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results
