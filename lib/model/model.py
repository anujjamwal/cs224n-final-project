import torch
from torch import nn
from transformers import GenerationMixin

from .masks import FlexAttentionMaskMixin, MaterialisedMaskMixin
from .generate import generate as hierarchical_generate

THOUGHT_TOKEN = "[THOUGHT]"
SOLUTION_TOKEN = "[SOLUTION]"
RETURN_TOKEN = "[RETURN]"
SPECIAL_TOKENS = [THOUGHT_TOKEN, SOLUTION_TOKEN, RETURN_TOKEN]


class _CausalLMModelBase(nn.Module):
    """Base class with shared init, build, forward, generate, and prune logic.

    Subclasses (or mixins) must provide ``_build_hierarchical_mask``.
    """

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.thought_token_id = tokenizer.convert_tokens_to_ids(THOUGHT_TOKEN)
        self.solution_token_id = tokenizer.convert_tokens_to_ids(SOLUTION_TOKEN)
        self.return_token_id = tokenizer.convert_tokens_to_ids(RETURN_TOKEN)

    @classmethod
    def build(cls, model, tokenizer):
        """
        Build the instance from given base model and tokenizer. As a part of the build
        the tokenizer is expanded to add additional tokens.

        :param model: Base model CausalLLM that is wrapped by this Model
        :param tokenizer: Corresponding tokenizer that is used with the base model
        :return: Instance of Model
        """
        if tokenizer.pad_token is None:
          tokenizer.pad_token = tokenizer.eos_token

        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        print(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")
        print(f"New vocab size: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

        return cls(model, tokenizer)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass. Uses hierarchical attention masking only during training;
        passes the original attention_mask through during eval."""

        if self.training:
            mask = self._build_hierarchical_mask(input_ids, padding_mask=attention_mask)
        else:
            mask = attention_mask

        return self.model(
            input_ids=input_ids,
            attention_mask=mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Custom autoregressive generation with hierarchical CoT pruning.

        When [RETURN] is generated, the chain of thought (tokens between the
        last [THOUGHT] and its corresponding [SOLUTION]) is pruned from the
        context, keeping only the solution summary. This reduces context length
        while preserving key information for subsequent reasoning steps.

        Returns:
            output_ids: final token sequence
            total_tokens_generated: total new tokens produced (including pruned)
            peak_context_length: max context window size seen during generation
        """
        
        return GenerationMixin.generate(
            self,
            *args,
            custom_generate=hierarchical_generate,
            **kwargs
        )


class CausalLMModelWithFlexAttention(FlexAttentionMaskMixin, _CausalLMModelBase):
    """Hierarchical CoT model using FlexAttention block masks (O(seq_len) memory)."""


class CausalLMModelWithMaterialisedMask(MaterialisedMaskMixin, _CausalLMModelBase):
    """Hierarchical CoT model using materialised 4-D float masks (O(seq_len^2) memory)."""


# Default: keep backward-compatible name that tries FlexAttention first
CausalLMModelWithHierarchicalCot = CausalLMModelWithMaterialisedMask
