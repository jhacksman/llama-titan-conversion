"""
Memory modules for Titans architecture implementation.

This package provides the memory components and utilities for implementing
the Titans architecture with DeepSeek's MoE-based routing system.
"""

from .memory_config import MemoryConfig
from .memory_modules import (
    CoreMemory,
    LongTermMemory,
    PersistentMemory,
    MemoryManager
)
from .memory_utils import (
    calculate_memory_requirements,
    optimize_memory_distribution,
    MemoryOptimizer
)

__all__ = [
    'MemoryConfig',
    'CoreMemory',
    'LongTermMemory',
    'PersistentMemory',
    'MemoryManager',
    'calculate_memory_requirements',
    'optimize_memory_distribution',
    'MemoryOptimizer'
]
