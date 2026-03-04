from dataclasses import field
from typing import Any
import torch
from torch import nn
from transformers import Trainer, AutoProcessor, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.processing_utils import ProcessorMixin
from trl import trainer
from trl.trainer.utils import get_config_model_id
import utils


class HCotSFTTrainer(trainer.sft_trainer.SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: trainer.sft_config.SFTConfig | TrainingArguments | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        prune_aware: bool = True,
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


    def _prepare_attention_mask(self, inputs: dict[str, torch.Tensor|Any]):
        input_ids = inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

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

    def _compute_loss_staged(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Staged pruning training: multiple forward passes mirroring inference.

        Each pruning event (at [RETURN]) creates a new stage where thought
        content is physically removed and the model does a fresh forward on
        the pruned sequence with contiguous positions — exactly matching
        inference behaviour.
        """
        input_ids = inputs['input_ids']
        labels = inputs.get('labels', input_ids)
        attention_mask = inputs['attention_mask']
        batch_size = input_ids.shape[0]
        device = input_ids.device

        batch_blocks = utils.find_cot_blocks(
            input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
        )

        # Build all stages for every batch element
        all_stages = []
        for b in range(batch_size):
            stages = utils.build_stages(
                input_ids[b], labels[b], attention_mask[b], batch_blocks[b]
            )
            all_stages.append(stages)

        max_num_stages = max(len(s) for s in all_stages)

        total_loss = None
        first_outputs = None
        pad_id = getattr(self.processing_class, 'pad_token_id', 0) or 0

        for stage_idx in range(max_num_stages):
            # Collect sequences from batch elements that participate in this stage
            participating = []
            for b in range(batch_size):
                if stage_idx < len(all_stages[b]):
                    participating.append(b)

            if not participating:
                continue

            # Pad to uniform length and stack into a batch
            max_len = max(all_stages[b][stage_idx][0].shape[0] for b in participating)
            n = len(participating)

            batch_ids = torch.full((n, max_len), pad_id, dtype=input_ids.dtype, device=device)
            batch_labels = torch.full((n, max_len), -100, dtype=input_ids.dtype, device=device)
            batch_mask = torch.zeros((n, max_len), dtype=attention_mask.dtype, device=device)

            for local_idx, b_idx in enumerate(participating):
                s_ids, s_labels, s_mask = all_stages[b_idx][stage_idx]

                batch_ids[local_idx, :s_ids.shape[0]] = s_ids
                batch_labels[local_idx, :s_labels.shape[0]] = s_labels
                batch_mask[local_idx, :s_mask.shape[0]] = s_mask

            stage_inputs: dict[str, torch.Tensor | Any] = {
                'input_ids': batch_ids,
                'attention_mask': batch_mask,
                'labels': batch_labels,
            }
            stage_loss, stage_outputs = Trainer.compute_loss(
                self, model, stage_inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch,
            )

            total_loss = stage_loss if total_loss is None else total_loss + stage_loss
            if first_outputs is None:
                first_outputs = stage_outputs

        if total_loss is None:
            # Edge case: no stages produced (shouldn't happen with valid data)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return (total_loss, first_outputs) if return_outputs else total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:

        if self.prune_aware:
            return self._compute_loss_staged(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Fallback: single-pass with hierarchical attention mask
        inputs['attention_mask'] = self._prepare_attention_mask(inputs)

        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        return (loss, outputs) if return_outputs else loss
