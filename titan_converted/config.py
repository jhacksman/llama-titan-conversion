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
    total_vram: int = 64 * (1024 ** 3)  # 64GB total VRAM
    num_gpus: int = 3  # 3x NVIDIA RTX 3090
    vram_per_gpu: int = field(init=False)
    
    def __post_init__(self):
        self.vram_per_gpu = self.total_vram // self.num_gpus


@dataclass
class MemoryConfig:
    """Memory-specific configuration."""
    # Component sizes
    long_term_mem_size: int = 1_000_000
    persistent_mem_size: int = 500_000
    
    # VRAM allocation (per component)
    vram_target: int = 10 * (1024 ** 3)  # Target 10GB
    vram_minimum: int = 5 * (1024 ** 3)  # Minimum 5GB
    
    # Memory features
    use_flash_attention: bool = True
    use_checkpointing: bool = True
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"


@dataclass
class MoEConfig:
    """MoE-specific configuration."""
    num_experts: int = 128
    num_experts_per_token: int = 6
    expert_capacity: int = 128
    expert_dim: int = 4096
    expert_stride: int = 2
    num_expert_groups: int = 2
    router_aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.01
    expert_dropout: float = 0.1


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
    
    # Position embeddings
    pos_embedding: Literal["rotary", "alibi", "relative"] = "rotary"
    rotary_dim: Optional[int] = None
    
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
    moe: MoEConfig = field(default_factory=MoEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate VRAM budget
        total_mem_target = self.memory.vram_target * 3  # Three components
        if total_mem_target > self.hardware.total_vram:
            raise ValueError(
                f"Memory target ({total_mem_target / 1e9:.2f}GB) "
                f"exceeds total VRAM ({self.hardware.total_vram / 1e9:.2f}GB)"
            )
        
        # Validate expert configuration
        experts_per_gpu = self.moe.num_experts // self.hardware.num_gpus
        if experts_per_gpu * self.hardware.num_gpus != self.moe.num_experts:
            raise ValueError(
                f"Number of experts ({self.moe.num_experts}) must be "
                f"divisible by number of GPUs ({self.hardware.num_gpus})"
            )
        
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
        total_mem_target = self.memory.vram_target * 3
        if total_mem_target > self.hardware.total_vram:
            # Scale down memory targets
            scale = self.hardware.total_vram / total_mem_target
            self.memory.vram_target = int(self.memory.vram_target * scale)
            
            # Ensure minimum requirements
            if self.memory.vram_target < self.memory.vram_minimum:
                raise ValueError(
                    "Cannot meet minimum memory requirements "
                    f"({self.memory.vram_minimum / 1e9:.2f}GB per component)"
                )
        
        # Enable memory optimization features
        self.memory.use_checkpointing = True
        self.memory.use_flash_attention = True
        
        # Adjust expert configuration
        self.moe.expert_capacity = min(
            self.moe.expert_capacity,
            self.model.max_seq_len // self.moe.num_experts
        )


def create_config(**kwargs) -> DeepSeekTitanConfig:
    """
    Create a configuration instance with optional overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        DeepSeekTitanConfig: Initialized configuration
    """
    # Create default configs
    hardware_config = HardwareConfig()
    memory_config = MemoryConfig()
    moe_config = MoEConfig()
    model_config = ModelConfig()
    
    # Update configs with provided values
    if 'hardware' in kwargs:
        for key, value in kwargs['hardware'].items():
            if hasattr(hardware_config, key):
                setattr(hardware_config, key, value)
    
    if 'memory' in kwargs:
        for key, value in kwargs['memory'].items():
            if hasattr(memory_config, key):
                setattr(memory_config, key, value)
    
    if 'moe' in kwargs:
        for key, value in kwargs['moe'].items():
            if hasattr(moe_config, key):
                setattr(moe_config, key, value)
    
    if 'model' in kwargs:
        for key, value in kwargs['model'].items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
    
    # Create main config
    config = DeepSeekTitanConfig(
        hardware=hardware_config,
        memory=memory_config,
        moe=moe_config,
        model=model_config
    )
    
    # Validate and optimize
    config.validate()
    config.optimize_for_hardware()
    
    return config
