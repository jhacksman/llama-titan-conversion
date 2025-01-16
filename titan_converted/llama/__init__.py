"""
LLaMA model implementation with Titans architecture modifications.
This package contains the base LLaMA implementation that will be extended
with Titans' three-component memory system.
"""

from .model import ModelArgs, Transformer, RMSNorm, Attention, TransformerBlock
from .generation import Llama, Dialog, Message, Role

__all__ = [
    'ModelArgs',
    'Transformer',
    'RMSNorm',
    'Attention',
    'TransformerBlock',
    'Llama',
    'Dialog',
    'Message',
    'Role'
]
