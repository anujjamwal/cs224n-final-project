import torch
from torch import nn
from transformers import GenerationMixin

from .masks import FlexAttentionMaskMixin, MaterialisedMaskMixin
from .generate import generate as hierarchical_generate

THOUGHT_TOKEN = "[THOUGHT]"
SOLUTION_TOKEN = "[SOLUTION]"
RETURN_TOKEN = "[RETURN]"
SPECIAL_TOKENS = [THOUGHT_TOKEN, SOLUTION_TOKEN, RETURN_TOKEN]
