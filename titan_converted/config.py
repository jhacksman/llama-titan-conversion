"""
Configuration for DeepSeek-Titans integration.

This module provides configuration classes that merge:
1. DeepSeek's MoE configuration
2. Titans memory parameters
3. Hardware-specific settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import torch


@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    total_vram: int = 64 * (1024 ** 3)  # 64GB shared VRAM
    num_gpus: int = 3  # 3x NVIDIA RTX 3090
    
    def __post_init__(self):
        # Verify minimum VRAM requirements (5GB per component)
        min_required = 15 * (1024 ** 3)  # 5GB * 3 components
        if self.total_vram < min_required:
            raise ValueError(
                f"Insufficient VRAM: {self.total_vram/1e9:.1f}GB < "
                f"{min_required/1e9:.1f}GB minimum"
            )


from .memory.memory_config import MemoryConfig  # Import the consolidated MemoryConfig





@dataclass
class ModelConfig:
    """Model architecture configuration."""
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 2097152  # Support 2M+ context
    multiple_of: int = 256
    norm_eps: float = 1e-5
    
    # Dropout settings
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Activation function
    activation_fn: Literal["gelu", "silu"] = "silu"
    
    # Position embeddings for extended context
    pos_embedding: Literal["rotary"] = "rotary"  # DeepSeek-R1 uses rotary
    rope_theta: float = 10000.0  # Base for rotary embeddings
    rope_scaling: float = 1.0  # For context extension
    rotary_dim: Optional[int] = None  # Will be set to dim//n_heads
    
    def __post_init__(self):
        if self.rotary_dim is None:
            self.rotary_dim = self.dim // self.n_heads


@dataclass
class DeepSeekTitanConfig:
    """
    Combined configuration for DeepSeek-Titans integration.
    
    This class merges:
    1. Hardware configuration
    2. Memory configuration
    3. MoE configuration
    4. Model architecture configuration
    """
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Convert hardware config if needed
        if isinstance(self.hardware, dict):
            self.hardware = HardwareConfig(**self.hardware)
            
        # Validate VRAM budget
        total_mem_target = self.memory.vram_target_per_component * 3  # Three components
        if total_mem_target > self.hardware.total_vram:
            raise ValueError(
                f"Memory target ({total_mem_target / 1e9:.2f}GB) "
                f"exceeds total VRAM ({self.hardware.total_vram / 1e9:.2f}GB)"
            )
        
        # Validate memory configuration
        if not self.memory.validate_vram_budget():
            self.memory.optimize_for_hardware()
        
        # Validate sequence length
        if self.model.max_seq_len > 2_097_152:  # 2M tokens
            raise ValueError(
                f"Maximum sequence length ({self.model.max_seq_len}) "
                "exceeds supported length (2,097,152)"
            )
        
        return True
    
    def optimize_for_hardware(self) -> None:
        """Optimize configuration for available hardware."""
        # Adjust memory allocation if needed
        total_mem_target = self.memory.vram_target_per_component * 3
        if total_mem_target > self.hardware.total_vram:
            # Scale down memory targets
            scale = self.hardware.total_vram / total_mem_target
            self.memory.vram_target_per_component = int(self.memory.vram_target_per_component * scale)
            
            # Ensure minimum requirements
            if self.memory.vram_target_per_component < self.memory.vram_minimum_per_component:
                raise ValueError(
                    "Cannot meet minimum memory requirements "
                    f"({self.memory.vram_minimum_per_component / 1e9:.2f}GB per component)"
                )
        
        # Enable memory optimization features
        self.memory.use_checkpointing = True
        self.memory.use_flash_attention = True
        
        # Adjust memory configuration
        if not self.memory.validate_vram_budget():
            self.memory.optimize_for_hardware()


def create_config(**kwargs) -> DeepSeekTitanConfig:
    """
    Create a configuration instance with optional overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        DeepSeekTitanConfig: Initialized configuration
    """
    # Convert nested dictionaries to proper config objects
    if 'hardware' in kwargs:
        if isinstance(kwargs['hardware'], dict):
            kwargs['hardware'] = HardwareConfig(**kwargs['hardware'])
    else:
        kwargs['hardware'] = HardwareConfig()
        
    if 'memory' in kwargs:
        if isinstance(kwargs['memory'], dict):
            kwargs['memory'] = MemoryConfig(**kwargs['memory'])
    else:
        kwargs['memory'] = MemoryConfig()
        
    if 'model' in kwargs:
        if isinstance(kwargs['model'], dict):
            kwargs['model'] = ModelConfig(**kwargs['model'])
    else:
        kwargs['model'] = ModelConfig()
        
    # Create main config
    config = DeepSeekTitanConfig(**kwargs)
    
    # Validate and optimize
    config.validate()
    config.optimize_for_hardware()
    
    return config
