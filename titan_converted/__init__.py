"""
Titans Architecture Implementation for LLaMA 7B 3.3

This package implements the Titans architecture using LLaMA 7B 3.3 as the base model,
focusing on the three-component memory system while optimizing for specific hardware
constraints (3x NVIDIA RTX 3090 GPUs, 64GB total VRAM).
"""

from .titan_model import TitanTransformer, TitanModelArgs
from .memory_modules import MemoryConfig, MemoryManager, LongTermMemory, PersistentMemory

__all__ = [
    'TitanTransformer',
    'TitanModelArgs',
    'MemoryConfig',
    'MemoryManager',
    'LongTermMemory',
    'PersistentMemory',
]
