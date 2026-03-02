from typing import Any
import torch
from torch import nn
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.processing_utils import ProcessorMixin
from trl import trainer
import utils


class HCotSFTTrainer(trainer.sft_trainer.SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: trainer.sft_config.SFTConfig | TrainingArguments | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        prune_aware: bool = False,
        **kwargs
    ):
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer # type: ignore
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        self.thought_token_id = tokenizer.convert_tokens_to_ids("[THOUGHT]")
        self.solution_token_id = tokenizer.convert_tokens_to_ids("[SOLUTION]")
        self.return_token_id = tokenizer.convert_tokens_to_ids("[RETURN]")
        self.prune_aware = prune_aware

        super().__init__(model, args=args, processing_class=processing_class, **kwargs)


    def _prepare_attention_mask(
        self,
        inputs: dict[str, torch.Tensor | Any],
        batch_blocks: list[list[tuple[int, int, int]]] | None = None,
    ):
        input_ids = inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if batch_blocks is None:
            batch_blocks = utils.find_cot_blocks(
                input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
            )

        # Create the default mask with lower triangle
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
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

    def _build_pruned_inputs(
        self,
        inputs: dict[str, torch.Tensor | Any],
        batch_blocks: list[list[tuple[int, int, int]]],
    ) -> dict[str, torch.Tensor | Any]:
        """Create pruned copies of input_ids/labels with thinking tokens removed.

        For each block ``(thought_pos, solution_pos, return_pos)`` the tokens in
        the range ``[thought_pos+1, solution_pos]`` (inclusive) are removed.  This
        matches what happens during inference pruning and the attention-mask hiding
        range.  The resulting sequences get standard causal attention with
        contiguous position IDs so the model learns to operate on a pruned context.
        """
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        pruned_rows: list[torch.Tensor] = []
        pruned_labels: list[torch.Tensor] = []
        for b in range(batch_size):
            keep = torch.ones(seq_len, dtype=torch.bool, device=device)
            for thought_pos, solution_pos, _return_pos in batch_blocks[b]:
                keep[thought_pos + 1 : solution_pos + 1] = False
            pruned_rows.append(input_ids[b][keep])
            pruned_labels.append(labels[b][keep])

        max_len = max(r.shape[0] for r in pruned_rows)
        pad_id = self.processing_class.pad_token_id  # type: ignore
        new_input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=input_ids.dtype, device=device
        )
        new_labels = torch.full(
            (batch_size, max_len), -100, dtype=labels.dtype, device=device
        )
        new_attn = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )
        for b, (r, l) in enumerate(zip(pruned_rows, pruned_labels)):
            new_input_ids[b, : r.shape[0]] = r
            new_labels[b, : l.shape[0]] = l
            new_attn[b, : r.shape[0]] = 1

        return {
            'input_ids': new_input_ids,
            'labels': new_labels,
            'attention_mask': new_attn,
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:

        batch_blocks = utils.find_cot_blocks(
            inputs['input_ids'], self.thought_token_id,
            self.solution_token_id, self.return_token_id,
        )

        # Pass 1: full sequence with hierarchical attention mask
        inputs['attention_mask'] = self._prepare_attention_mask(inputs, batch_blocks)
        full_loss, outputs = Trainer.compute_loss(
            self, model, inputs, return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if self.prune_aware:
            # Pass 2: pruned sequence with standard causal attention
            pruned_inputs = self._build_pruned_inputs(inputs, batch_blocks)
            pruned_loss, _ = Trainer.compute_loss(
                self, model, pruned_inputs, return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            loss = full_loss + pruned_loss
        else:
            loss = full_loss

        return (loss, outputs) if return_outputs else loss
