import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

THOUGHT_TOKEN='[THOUGHT]'
SOLUTION_TOKEN='[SOLUTION]'
RETURN_TOKEN='[RETURN]'

DEFAULT_SEED = {
    THOUGHT_TOKEN:  "subproblem start",
    SOLUTION_TOKEN: "summary of solution",
    RETURN_TOKEN:   "subproblem return",
    "<think>":    "thinking start",
    "</think>":   "thinking complete",
}

def prepare_base_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, token_seed=DEFAULT_SEED):
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": tokenizer.all_special_tokens + 
                [THOUGHT_TOKEN, SOLUTION_TOKEN, RETURN_TOKEN] + 
                ['<think>','</think>']
        }
    )

    tokenizer.chat_template = """\
    {%- if messages[0]['role'] == 'system' -%}
    <|im_start|>system
    {{ messages[0]['content'] | trim }}<|im_end|>
    {%- else -%}
    <|im_start|>system
    <|im_end|>
    {%- endif -%}
    {%- for message in messages -%}
    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant') -%}
    <|im_start|>{{ message.role }}
    {%- if message['role'] == 'assistant' %}
    {% generation %}{{ message['content'] | trim }}<|im_end|>{% endgeneration %}
    {%- else %}
    {{ message['content'] | trim }}<|im_end|>
    {%- endif %}
    {%- endif -%}
    {%- endfor -%}
    {%- if add_generation_prompt %}
    <|im_start|>assistant
    {%- endif -%}
    """

    model.resize_token_embeddings(len(tokenizer))

    token_seed_dict = DEFAULT_SEED
    if token_seed:
        token_seed_dict = token_seed_dict | token_seed
    
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        for special_tok, seed_word in token_seed_dict.items():
            # Get the ID of the newly added special token
            special_id = tokenizer.convert_tokens_to_ids(special_tok)
            # Tokenize the seed word — may produce multiple sub-tokens
            seed_ids = tokenizer.encode(seed_word, add_special_tokens=False)
            # Average the embeddings of the seed sub-tokens
            seed_embeds = embed_layer.weight[seed_ids]          # type: ignore # (n_subtokens, hidden_dim)
            avg_embed = seed_embeds.mean(dim=0)                 # (hidden_dim,)
            # Write into the embedding table
            embed_layer.weight[special_id] = avg_embed # type: ignore
            logger.info(f"  {special_tok:12s} (id={special_id}) <- avg of '{seed_word}' tokens {seed_ids}")

        # Also copy input embeddings to the output (lm_head) if they are NOT tied,
        # so the model can predict these tokens from the start.
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != embed_layer.weight.data_ptr(): # type: ignore
            for special_tok, seed_word in token_seed_dict.items():
                special_id = tokenizer.convert_tokens_to_ids(special_tok)
                lm_head.weight[special_id] = embed_layer.weight[special_id] # type: ignore

    return model, tokenizer
