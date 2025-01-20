"""
Configuration for Titans memory modules.

This module defines the configuration parameters for the three-component memory system:
1. Core Module: Modified attention mechanism
2. Long-term Memory: Neural memory for historical context
3. Persistent Memory: Task-specific knowledge storage

The configuration ensures efficient VRAM usage across multiple GPUs while maintaining
model performance and supporting extended context windows.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import torch


@dataclass
class MemoryConfig:
    """Configuration parameters for memory modules."""
    # Model dimensions
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_attention_heads: int = 32
    max_sequence_length: int = 2097152  # Support for 2M+ context
    
    # Memory-specific parameters
    memory_dim: int = 4096
    num_memory_heads: int = 8
    max_memory_length: int = 1000000
    memory_update_interval: int = 100
    
    # VRAM management
    vram_target_per_component: int = 10 * (1024 ** 3)  # Target 10GB per component
    vram_minimum_per_component: int = 5 * (1024 ** 3)  # Minimum 5GB per component
    total_vram_budget: int = 64 * (1024 ** 3)  # Total 64GB VRAM
    num_gpus: int = 3
    
    # Memory routing
    num_memory_experts: int = 8
    num_activated_experts: int = 4
    routing_algorithm: Literal["topk", "hash", "learned"] = "topk"
    
    # Optimization
    use_checkpointing: bool = True
    use_flash_attention: bool = True
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    
    # Component-specific settings
    core_module: dict = {}
    long_term_memory: dict = {}
    persistent_memory: dict = {}
    
    def __post_init__(self):
        """Initialize component-specific configurations."""
        self.core_module = {
            "dim": self.hidden_dim,
            "num_heads": self.num_attention_heads,
            "head_dim": self.hidden_dim // self.num_attention_heads,
            "max_sequence_length": self.max_sequence_length,
            "use_flash_attention": self.use_flash_attention
        }
        
        self.long_term_memory = {
            "memory_dim": self.memory_dim,
            "num_heads": self.num_memory_heads,
            "max_memory_length": self.max_memory_length,
            "update_interval": self.memory_update_interval
        }
        
        self.persistent_memory = {
            "memory_dim": self.memory_dim,
            "num_heads": self.num_memory_heads,
            "max_memory_length": self.max_memory_length // 2,  # Smaller persistent memory
            "num_experts": self.num_memory_experts
        }
    
    def validate_vram_budget(self) -> bool:
        """
        Validate that memory configuration fits within VRAM budget.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Calculate approximate VRAM requirements
        core_vram = (
            self.hidden_dim * self.max_sequence_length * 4 +  # Activations
            self.hidden_dim * self.hidden_dim * 4 * 3 +      # QKV projections
            self.hidden_dim * self.max_sequence_length * 2    # KV cache
        )
        
        long_term_vram = (
            self.memory_dim * self.max_memory_length * 2 +   # Memory bank
            self.memory_dim * self.hidden_dim * 4            # Projections
        )
        
        persistent_vram = (
            self.memory_dim * (self.max_memory_length // 2) * 2 +  # Memory bank
            self.memory_dim * self.hidden_dim * 4                  # Projections
        )
        
        total_vram = core_vram + long_term_vram + persistent_vram
        
        # Check if configuration fits budget
        return total_vram <= self.total_vram_budget
    
    def optimize_for_hardware(self) -> None:
        """Optimize configuration for available hardware."""
        if not self.validate_vram_budget():
            # Adjust memory lengths to fit budget
            reduction_factor = 0.8  # Reduce by 20%
            self.max_memory_length = int(self.max_memory_length * reduction_factor)
            self.persistent_memory["max_memory_length"] = self.max_memory_length // 2
            
            # Enable memory-saving features
            self.use_checkpointing = True
            self.use_flash_attention = True
            
            # Verify again
            if not self.validate_vram_budget():
                raise ValueError("Cannot fit memory configuration within VRAM budget")
