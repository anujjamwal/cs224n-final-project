from typing import Any, Callable
import torch
from torch import nn
from transformers import Trainer
# from trl import GrpoTrainer


class HCotTrainer(Trainer):
    def __init__(self, attention_mask_func: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attention_mask_func = attention_mask_func
      
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        padding_mask = inputs.get('attention_mask', None)
        attention_mask = self._attention_mask_func(input_ids=inputs["input_ids"], padding_mask=padding_mask)
        inputs['attention_mask'] = attention_mask
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


# class HCotGrpoTrainer(GrpoTrainer):
#     def __init__(self, attention_mask_func: Callable, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._attention_mask_func = attention_mask_func
    
#     def compute_loss(
#         self,
#         model: nn.Module,
#         inputs: dict[str, torch.Tensor | Any],
#         return_outputs: bool = False,
#         num_items_in_batch: torch.Tensor | int | None = None,
#     ) -> torch.Tensor | tuple[torch.Tensor, Any]:
#         padding_mask = inputs.get('attention_mask', None)
#         attention_mask = self._attention_mask_func(input_ids=inputs["input_ids"], padding_mask=padding_mask)
#         inputs['attention_mask'] = attention_mask
#         return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

# For the GRPO traininer, the compute_loss is custom function.
# Explore the use of data collator in Trainer to pass the correct mask to the compute loss method
# 
# train
#   _inner_training_loop
#       train_dataloader = self.get_train_dataloader() 
#       training_step(model, inputs, num_items_in_batch)
#           self._prepare_inputs(inputs)
#           self.compute_loss

#  GRPO Trainer has a custom compute_loss function. We can still override that function.