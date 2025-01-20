"""
Memory modules for Titans architecture.

This module implements the specialized memory components:
1. Core Module: Modified attention mechanism
2. Long-term Memory: Neural memory for historical context
3. Persistent Memory: Task-specific knowledge storage

These modules are designed to work with DeepSeek's MoE architecture
while respecting VRAM constraints and supporting 2M+ context windows.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_config import MemoryConfig
from .memory_utils import MemoryOptimizer


class CoreMemory(nn.Module):
    """
    Core memory module with modified attention mechanism.
    
    This module implements:
    1. Efficient attention for long sequences
    2. Integration with MoE routing
    3. Memory-optimized operations
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)
        self.output = nn.Linear(config.dim, config.dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.dim)
        
        # Optional flash attention
        self.use_flash_attention = config.use_flash_attention
        
        # Initialize memory tracking
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_sequence_length).expand((1, -1))
        )
    
    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention scores and apply to values."""
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.config.dim)
        
        # Apply mask if provided
        if mask is not None:
            # Convert and reshape mask for attention scores
            mask = mask.bool() if not mask.dtype == torch.bool else mask
            # Create causal attention mask [batch_size, 1, 1, seq_len]
            attention_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)
        
        
        # Apply attention to values
        return torch.matmul(weights, value)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through core memory.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask
            past_key_values: Optional cached key/value states
            
        Returns:
            Tuple of:
            - Updated tensor with attention applied
            - Dictionary containing memory updates and stats
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Compute QKV
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Reshape for attention heads
        head_dim = self.config.dim // self.config.num_attention_heads
        query = query.view(batch_size, seq_len, self.config.num_attention_heads, head_dim)
        key = key.view(batch_size, seq_len, self.config.num_attention_heads, head_dim)
        value = value.view(batch_size, seq_len, self.config.num_attention_heads, head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch_size, n_heads, seq_len, head_dim]
        value = value.transpose(1, 2)   # [batch_size, n_heads, seq_len, head_dim]
        
        # Compute attention
        context = self._compute_attention(query, key, value, mask)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_heads, head_dim]
        context = context.view(batch_size, seq_len, self.config.dim)
        output = self.output(context)
        
        return output, {
            "attention_pattern": context,
            "memory_usage": context.element_size() * context.nelement()
        }


class LongTermMemory(nn.Module):
    """
    Neural memory module for maintaining historical context.
    
    This module implements:
    1. Efficient storage mechanism for long sequences
    2. Attention-based retrieval system
    3. Automatic context maintenance
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Initialize memory bank
        self.register_parameter(
            'memory_bank',
            nn.Parameter(torch.zeros(
                config.max_memory_length,
                config.memory_dim
            ))
        )
        
        # Access mechanisms
        self.query_proj = nn.Linear(config.dim, config.memory_dim)
        self.key_proj = nn.Linear(config.memory_dim, config.memory_dim)
        self.value_proj = nn.Linear(config.memory_dim, config.dim)
        
        # Output processing
        self.output_proj = nn.Linear(config.dim, config.dim)
        self.layer_norm = nn.LayerNorm(config.dim)
        
        # Update mechanism
        self.update_gate = nn.Linear(
            config.dim + config.memory_dim,
            config.memory_dim
        )
        
        # Initialize memory bank with small random values
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through long-term memory.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of:
            - Updated tensor with historical context
            - Dictionary containing memory updates and stats
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries
        queries = self.query_proj(x)
        
        # Get keys and values from memory bank
        keys = self.key_proj(self.memory_bank)
        values = self.value_proj(self.memory_bank)
        
        # Compute attention scores [batch_size, seq_len, memory_len]
        scores = torch.matmul(queries, keys.transpose(0, 1))
        scores = scores / math.sqrt(self.config.memory_dim)
        
        if mask is not None:
            # Convert mask to boolean and expand for memory attention
            mask = mask.bool() if not mask.dtype == torch.bool else mask
            # Expand mask for memory length dimension [batch_size, seq_len, 1]
            attention_mask = mask.unsqueeze(-1) if mask is not None else None
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Get attention weights [batch_size, seq_len, memory_len]
        weights = F.softmax(scores, dim=-1)
        memories = torch.matmul(weights, values)
        
        # Update memory bank
        with torch.no_grad():
            # Update with moving average
            update_mask = torch.rand(
                self.memory_bank.shape[0],
                device=x.device
            ) < 0.1
            
            if update_mask.any():
                new_memories = x.mean(dim=(0, 1))  # [hidden_dim]
                self.memory_bank.data[update_mask] = (
                    0.9 * self.memory_bank.data[update_mask] +
                    0.1 * new_memories.unsqueeze(0)
                )
        
        # Process and combine
        output = self.output_proj(memories)
        output = self.layer_norm(output + x)
        
        return output, {
            "updates": update_mask.sum().item(),
            "memory_usage": self.memory_bank.element_size() * self.memory_bank.nelement()
        }


class PersistentMemory(nn.Module):
    """
    Specialized storage system for task-specific knowledge.
    
    This module implements:
    1. Permanent knowledge storage
    2. Efficient retrieval mechanism
    3. Task-specific optimization
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Initialize knowledge base
        self.register_parameter(
            'knowledge_bank',
            nn.Parameter(torch.zeros(
                config.persistent_memory["max_memory_length"],
                config.memory_dim
            ))
        )
        
        # Access mechanisms
        self.query_proj = nn.Linear(config.dim, config.memory_dim)
        self.key_proj = nn.Linear(config.memory_dim, config.memory_dim)
        self.value_proj = nn.Linear(config.memory_dim, config.dim)
        
        # Integration components
        self.fusion = nn.Linear(
            config.dim + config.memory_dim,
            config.dim
        )
        self.layer_norm = nn.LayerNorm(config.dim)
        
        # Expert routing
        self.num_experts = config.persistent_memory["num_experts"]
        self.expert_gate = nn.Linear(config.dim, self.num_experts)
        
        # Initialize knowledge bank
        nn.init.normal_(self.knowledge_bank, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through persistent memory.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            task_id: Optional task identifier for specialized knowledge
            
        Returns:
            torch.Tensor: Updated tensor with integrated knowledge
        """
        batch_size, seq_len, _ = x.shape
        
        # Route to experts
        gate_logits = self.expert_gate(x)
        weights = F.softmax(gate_logits, dim=-1)
        
        # Query knowledge base
        queries = self.query_proj(x)
        keys = self.key_proj(self.knowledge_bank)
        values = self.value_proj(self.knowledge_bank)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(0, 1))
        scores = scores / math.sqrt(self.config.memory_dim)
        
        # Apply task-specific routing if provided
        if task_id is not None:
            task_mask = torch.zeros(
                (batch_size, self.num_experts),
                device=x.device
            )
            task_mask.scatter_(1, task_id.unsqueeze(1), 1)
            weights = weights * task_mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Retrieve and integrate knowledge
        retrieved = torch.matmul(
            F.softmax(scores, dim=-1),
            values
        )
        
        # Combine with input
        combined = torch.cat([x, retrieved], dim=-1)
        output = self.fusion(combined)
        
        return self.layer_norm(output + x)


class MemoryManager(nn.Module):
    """
    Coordinator for memory modules.
    
    This class handles:
    1. Memory distribution across GPUs
    2. Synchronization between components
    3. VRAM optimization
    """
    def __init__(
        self,
        config: MemoryConfig,
        optimizer: Optional[MemoryOptimizer] = None
    ):
        super().__init__()
        self.config = config
        self.optimizer = optimizer or MemoryOptimizer(config)
        
        # Initialize memory modules
        self.core = CoreMemory(config)
        self.long_term = LongTermMemory(config)
        self.persistent = PersistentMemory(config)
        
        # Memory states
        self.register_buffer('core_state', None)
        self.register_buffer('long_term_state', None)
        self.register_buffer('persistent_state', None)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through memory system.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask
            task_id: Optional task identifier
            
        Returns:
            torch.Tensor: Processed tensor with memory context
        """
        # Process through core memory
        core_out, core_stats = self.core(x, mask)
        self.core_state = core_out.detach()
        
        # Process through long-term memory
        long_term_out, long_term_stats = self.long_term(core_out, mask)
        self.long_term_state = long_term_out.detach()
        
        # Process through persistent memory
        persistent_out = self.persistent(long_term_out, task_id)
        self.persistent_state = persistent_out.detach()
        
        # Track expert routing statistics
        if hasattr(self.persistent, "expert_gate"):
            # Track expert usage from gate assignments
            gate_logits = self.persistent.expert_gate(long_term_out)
            expert_assignments = gate_logits.argmax(dim=-1)
            expert_counts = torch.bincount(
                expert_assignments.flatten(),
                minlength=self.config.num_memory_experts
            )
            self.expert_counts = expert_counts
            self.routing_stats = {
                "expert_counts": expert_counts,
                "total_tokens": long_term_out.size(0) * long_term_out.size(1),
                "active_experts": (expert_counts > 0).sum().item()
            }
        
        return persistent_out
