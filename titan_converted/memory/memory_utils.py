"""
Utility functions for Titans memory management and optimization.

This module provides helper functions for managing memory allocation,
VRAM distribution, and optimization across multiple GPUs.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

from .memory_config import MemoryConfig
import torch
import torch.nn as nn
import torch.distributed as dist


def calculate_memory_requirements(
    batch_size: int,
    seq_length: int,
    dim: int,
    memory_length: int,
    dtype_size: int = 2  # fp16/bf16 = 2 bytes
) -> Dict[str, int]:
    """
    Calculate memory requirements for different components.
    
    Args:
        batch_size: Batch size for processing
        seq_length: Sequence length to process
        dim: Model dimension size
        memory_length: Length of memory banks
        dtype_size: Size of data type in bytes
    
    Returns:
        Dict containing memory requirements for each component
    """
    # Core attention memory
    core_size = (
        batch_size * seq_length * dim * 4 * dtype_size +  # Activations
        dim * dim * 4 * dtype_size * 3 +                 # QKV projections
        batch_size * seq_length * dim * 2 * dtype_size   # KV cache
    )
    
    # Long-term memory
    long_term_size = (
        dim * memory_length * dtype_size +              # Memory bank
        batch_size * dim * 4 * dtype_size +            # Access mechanisms
        batch_size * seq_length * dim * dtype_size     # Retrieved context
    )
    
    # Persistent memory
    persistent_size = (
        dim * (memory_length // 2) * dtype_size +      # Knowledge base
        batch_size * dim * 4 * dtype_size +            # Access mechanisms
        batch_size * seq_length * dim * dtype_size     # Retrieved knowledge
    )
    
    return {
        "core_attention": core_size,
        "long_term_memory": long_term_size,
        "persistent_memory": persistent_size,
        "total": core_size + long_term_size + persistent_size
    }


def optimize_memory_distribution(
    total_vram: int,
    n_gpus: int,
    batch_size: int,
    seq_length: int,
    config: 'MemoryConfig'
) -> Dict[str, Dict[str, Union[int, Optional[int]]]]:
    """
    Calculate optimal memory distribution across available GPUs.
    
    Args:
        total_vram: Total available VRAM in bytes
        n_gpus: Number of available GPUs
        batch_size: Batch size for processing
        seq_length: Sequence length to process
        config: Memory configuration object
    
    Returns:
        Dict containing memory allocation strategy for each component
    """
    # Calculate memory requirements
    requirements = calculate_memory_requirements(
        batch_size=batch_size,
        seq_length=seq_length,
        dim=config.dim,
        memory_length=config.max_memory_length
    )
    
    # Verify total requirements against budget
    if requirements["total"] > total_vram:
        raise ValueError(
            f"Memory requirements ({requirements['total'] / 1e9:.2f}GB) "
            f"exceed budget ({total_vram / 1e9:.2f}GB)"
        )
    
    # Target ~10GB per component with flexibility
    target_per_component = config.vram_target_per_component
    minimum_per_component = config.vram_minimum_per_component
    
    # Calculate available memory after accounting for shared resources
    shared_memory = batch_size * seq_length * config.dim * 4  # Activations
    available_memory = total_vram - shared_memory
    
    # Distribute remaining memory among components
    if available_memory >= 3 * target_per_component:
        # Ideal case: each component gets target allocation
        memory_per_component = target_per_component
    else:
        # Constrained case: distribute evenly with minimum guarantee
        memory_per_component = max(
            minimum_per_component,
            available_memory // 3
        )
    
    return {
        "core_module": {
            "vram_allocated": int(memory_per_component)  # Share VRAM across GPUs
        },
        "long_term": {
            "vram_allocated": int(memory_per_component)
        },
        "persistent": {
            "vram_allocated": int(memory_per_component)
        }
    }


class MemoryOptimizer:
    """
    Optimizer for managing memory usage across components.
    Implements memory-saving techniques including:
    1. Gradient checkpointing
    2. Mixed precision training
    3. Activation recomputation
    4. Memory-efficient attention
    """
    def __init__(self, config: 'MemoryConfig'):
        self.config = config
        self.memory_stats = {
            "allocated": 0,
            "cached": 0,
            "peak": 0
        }
    
    def _enable_checkpointing(self, model: nn.Module) -> None:
        """Enable gradient checkpointing for transformer layers."""
        if hasattr(model, "gradient_checkpointing"):
            model.gradient_checkpointing = True
        
        # Enable for all transformer layers
        for module in model.modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                if hasattr(module, "checkpoint_core_attention"):
                    module.checkpoint_core_attention = True
    
    def _optimize_attention(self, model: nn.Module) -> None:
        """Implement memory-efficient attention patterns."""
        for module in model.modules():
            if hasattr(module, "use_flash_attention"):
                module.use_flash_attention = True
    
    def _monitor_memory(self) -> None:
        """Update total memory statistics across all GPUs."""
        total_allocated = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
        total_cached = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count()))
        total_peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        
        self.memory_stats["allocated"] = total_allocated
        self.memory_stats["cached"] = total_cached
        self.memory_stats["peak"] = total_peak
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Optimize model memory usage.
        
        Args:
            model: Model to optimize
        
        Returns:
            Optimized model
        """
        # Enable memory-saving techniques
        if self.config.use_checkpointing:
            self._enable_checkpointing(model)
        
        if self.config.use_flash_attention:
            self._optimize_attention(model)
        
        # Monitor initial memory state
        self._monitor_memory()
        
        # Verify we're within budget
        if self.memory_stats["peak"] > self.config.total_vram_budget:
            raise RuntimeError(
                f"Peak memory usage ({self.memory_stats['peak'] / 1e9:.2f}GB) "
                f"exceeds budget ({self.config.total_vram_budget / 1e9:.2f}GB)"
            )
        
        return model
