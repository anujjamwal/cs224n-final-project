from dataclasses import field
from typing import Any, Callable
import torch
from torch import nn
from transformers import Trainer, AutoProcessor, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.processing_utils import ProcessorMixin
from trl import trainer
from trl.trainer.utils import get_config_model_id
import utils

class HCotSFTConfig(trainer.sft_config.SFTConfig):
    thought_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for marking start of thought."
        },
    )
    solution_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for marking start of thought."
        },
    )
    return_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for marking start of thought."
        },
    )


class HCotSFTTrainer(trainer.sft_trainer.SFTTrainer):
    def __init__(
        self, 
        model: PreTrainedModel,
        args: HCotSFTConfig | trainer.sft_config.SFTConfig | TrainingArguments | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        **kwargs
    ):
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))
        
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer # type: ignore
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if hasattr(args, 'thought_token'):
            thought_token = getattr(args, 'thought_token', '[THOUGHT]')
        else:
            thought_token = '[THOUGHT]'

        if hasattr(args, 'thought_token'):
            solution_token = getattr(args, 'solution_token', '[SOLUTION]')
        else:
            solution_token = '[SOLUTION]'

        if hasattr(args, 'return_token'):
            return_token = getattr(args, 'return_token', '[RETURN]')
        else:
            return_token = '[RETURN]'

        self.thought_token_id = tokenizer.convert_tokens_to_ids(thought_token)
        self.solution_token_id = tokenizer.convert_tokens_to_ids(solution_token)
        self.return_token_id = tokenizer.convert_tokens_to_ids(return_token)

        super().__init__(model, args=args, processing_class=processing_class, **kwargs)
    

    def _prepare_attention_mask(self, inputs: dict[str, torch.Tensor|Any]):
        input_ids = inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        batch_blocks = utils.find_cot_blocks(
            input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
        )

        # Create the default mask with lower triangle
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool)
        )

        padding_mask = inputs['attention_mask']
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

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:

        inputs['attention_mask'] = self._prepare_attention_mask(model, inputs)

        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        return (loss, outputs) if return_outputs else loss
