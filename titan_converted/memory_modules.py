"""
Memory modules for Titans architecture.

This module implements the specialized memory components of the Titans architecture:
1. Long-term Memory: For maintaining historical context
2. Persistent Memory: For task-specific knowledge storage

These modules are designed to work with the distributed computing setup across
multiple GPUs while respecting VRAM constraints.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .memory_utils import optimize_memory_distribution

@dataclass
class MemoryConfig:
    """Configuration for memory modules."""
    hidden_dim: int = 4096
    max_history_len: int = 1000000
    knowledge_dim: int = 4096
    num_memory_heads: int = 8
    dropout: float = 0.1
    update_interval: int = 100


class LongTermMemory(nn.Module):
    """
    Neural memory module for maintaining and retrieving historical context.
    
    This module implements:
    1. Efficient storage mechanism for long sequences
    2. Attention-based retrieval system
    3. Automatic context maintenance
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Initialize memory banks
        self.register_parameter(
            'memory_bank',
            nn.Parameter(torch.zeros(config.max_history_len, config.hidden_dim))
        )
        
        # Access mechanisms
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Output processing
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # Update mechanism
        self.update_gate = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        
        # Initialize memory bank with small random values
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through long-term memory.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of:
            - Updated tensor with historical context
            - Dictionary containing memory updates and stats
        """
        # Compute attention scores
        queries = self.query_proj(x)  # [batch_size, seq_len, hidden_dim]
        keys = self.key_proj(self.memory_bank)  # [max_history_len, hidden_dim]
        values = self.value_proj(self.memory_bank)  # [max_history_len, hidden_dim]
        
        # Retrieve relevant history
        scores = torch.matmul(queries, keys.transpose(0, 1))  # [batch_size, seq_len, max_history_len]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        scores = F.softmax(scores / math.sqrt(self.config.hidden_dim), dim=-1)
        
        retrieved = torch.matmul(scores, values)  # [batch_size, seq_len, hidden_dim]
        
        # Update memory contents
        with torch.no_grad():
            # Update memory bank with moving average
            update_mask = torch.rand(self.memory_bank.shape[0], device=x.device) < 0.1
            if update_mask.any():
                new_memories = x.mean(dim=(0, 1))  # [hidden_dim]
                self.memory_bank.data[update_mask] = (
                    0.9 * self.memory_bank.data[update_mask] +
                    0.1 * new_memories.unsqueeze(0).expand(update_mask.sum(), -1)
                )
        
        # Process and combine
        output = self.output_proj(retrieved)
        output = self.layer_norm(output + x)
        
        return output, {
            "updates": update_mask.sum().item(),
            "memory_usage": self.memory_bank.abs().mean().item()
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
        
        # TODO: Initialize components
        # 1. Knowledge base
        self.knowledge_bank = nn.Parameter(
            torch.zeros(config.knowledge_dim, config.hidden_dim)
        )
        
        # 2. Access mechanisms
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 3. Integration components
        self.fusion_layer = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through persistent memory.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            task_id: Optional task identifier for specialized knowledge
            
        Returns:
            torch.Tensor: Updated tensor with integrated knowledge
        """
        # Query knowledge base
        queries = self.query_proj(x)  # [batch_size, seq_len, hidden_dim]
        keys = self.key_proj(self.knowledge_bank)  # [knowledge_dim, hidden_dim]
        values = self.value_proj(self.knowledge_bank)  # [knowledge_dim, hidden_dim]
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(0, 1))  # [batch_size, seq_len, knowledge_dim]
        scores = F.softmax(scores / math.sqrt(self.config.hidden_dim), dim=-1)
        
        # Retrieve and integrate knowledge
        retrieved = torch.matmul(scores, values)  # [batch_size, seq_len, hidden_dim]
        
        # Combine with input
        combined = torch.cat([x, retrieved], dim=-1)  # [batch_size, seq_len, 2*hidden_dim]
        output = self.fusion_layer(combined)  # [batch_size, seq_len, hidden_dim]
        
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
        vram_budget: int,
        n_gpus: int
    ):
        super().__init__()
        self.config = config
        self.vram_budget = vram_budget
        self.n_gpus = n_gpus
        
        # Detect CPU mode
        self.use_gpu = (
            torch.cuda.is_available() and
            n_gpus > 0 and
            hasattr(torch.distributed, 'is_initialized') and
            torch.distributed.is_initialized()
        )
        
        if self.use_gpu:
            # Full memory system for GPU mode
            self.long_term = LongTermMemory(config)
            self.persistent = PersistentMemory(config)
            self.register_buffer('long_term_state', None)
            self.register_buffer('persistent_state', None)
            self.optimize_distribution()
        else:
            # Simplified memory for CPU mode with reduced dimensions
            reduced_dim = min(config.hidden_dim, 128)  # Smaller dimension for CPU
            # Create minimal versions of memory modules for CPU
            self.long_term = nn.ModuleDict({
                'query': nn.Linear(reduced_dim, reduced_dim),
                'output': nn.Linear(reduced_dim, reduced_dim),
                'norm': nn.LayerNorm(reduced_dim)
            })
            self.persistent = nn.ModuleDict({
                'query': nn.Linear(reduced_dim, reduced_dim),
                'output': nn.Linear(reduced_dim, reduced_dim),
                'norm': nn.LayerNorm(reduced_dim)
            })
            # Dimension reduction layers
            self.cpu_dim_reduce = nn.Linear(config.hidden_dim, reduced_dim)
            self.cpu_dim_restore = nn.Linear(reduced_dim, config.hidden_dim)
            # Memory bank
            self.register_buffer(
                'cpu_memory_bank',
                torch.zeros(min(100, config.max_history_len), reduced_dim)
            )

    def optimize_distribution(self):
        """Optimize memory distribution across GPUs."""
        if torch.cuda.is_available():
            # Distribute components across available GPUs
            if self.n_gpus >= 3:
                self.long_term.to(f'cuda:{1}')
                self.persistent.to(f'cuda:{2}')
            else:
                self.long_term.to('cuda:0')
                self.persistent.to('cuda:0')

    def get_long_term_context(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get context from long-term memory."""
        if self.long_term_state is None:
            self.long_term_state = torch.zeros_like(x)
        context, _ = self.long_term(x)
        self.long_term_state = context.detach()
        return context

    def get_persistent_context(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get context from persistent memory."""
        if self.persistent_state is None:
            self.persistent_state = torch.zeros_like(x)
        context = self.persistent(x)
        self.persistent_state = context.detach()
        return context

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through memory system with detailed logging.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            task_id: Optional task identifier
            
        Returns:
            torch.Tensor: Processed tensor with memory context
        """
        try:
            print(f"\nMemoryManager forward pass:")
            print(f"Input shape: {x.shape}")
            print(f"Input device: {x.device}")
            print(f"Mode: {'GPU' if self.use_gpu else 'CPU'}")
            if mask is not None:
                print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            
            if self.use_gpu:
                # Full memory processing for GPU mode
                long_term_out, _ = self.long_term(x, mask)
                self.long_term_state = long_term_out.detach()
                persistent_out = self.persistent(long_term_out, task_id)
                self.persistent_state = persistent_out.detach()
                return persistent_out
            else:
                print("Starting CPU memory processing...")
                # Basic attention mechanism with small memory bank
                print("Computing query projection...")
                # Reduce dimension for CPU processing
                # Reshape input for linear layer
                batch_size, seq_len, hidden_dim = x.shape
                x_flat = x.view(-1, hidden_dim)  # Flatten batch and sequence dimensions
                x_reduced_flat = self.cpu_dim_reduce(x_flat)
                x_reduced = x_reduced_flat.view(batch_size, seq_len, -1)
                
                # Process through simplified memory modules
                query = self.long_term['query'](x_reduced)
                print(f"Query shape: {query.shape}")
                
                print("Computing attention scores...")
                # Handle attention mask
                scores = torch.matmul(query, self.cpu_memory_bank.t())  # [batch_size, seq_len, memory_len]
                print(f"Initial scores shape: {scores.shape}")
                
                # Create and apply memory recency mask
            memory_len = self.cpu_memory_bank.size(0)
            memory_mask = torch.zeros(
                (1, 1, memory_len),  # Will be broadcast automatically
                device=scores.device
            )
            memory_mask[..., memory_len//2:] = float('-inf')  # Only attend to first half
            scores = scores + memory_mask
            
            # Handle attention mask if provided
            if mask is not None and isinstance(mask, torch.Tensor):
                # Convert mask to float type if needed
                if mask.dtype == torch.bool:
                    attention_mask = torch.zeros_like(mask, dtype=torch.float)
                    attention_mask.masked_fill_(mask, float('-inf'))
                else:
                    attention_mask = mask
                
                # Ensure mask has batch dimension
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0)
                
                # Create memory attention mask
                # Expand attention mask to cover memory dimension
                expanded_mask = attention_mask.unsqueeze(-1)  # [batch, seq, seq, 1]
                memory_attention_mask = expanded_mask.expand(
                    -1, -1, -1, memory_len
                ).reshape(attention_mask.size(0), attention_mask.size(1), -1)
                
                # Apply attention mask
                scores = scores + memory_attention_mask
            
            # Apply attention with temperature scaling
            attn = F.softmax(scores / math.sqrt(self.config.hidden_dim), dim=-1)
            context = torch.matmul(attn, self.cpu_memory_bank)  # [batch_size, seq_len, hidden_dim]
            
            # Update memory bank (simple moving average)
            with torch.no_grad():
                # Update last 10% of memory bank with exponential moving average
                update_size = max(1, self.cpu_memory_bank.size(0) // 10)
                # Use max pooling to select most salient features
                new_memories = F.adaptive_max_pool2d(
                    x.transpose(1, 2),  # [batch_size, hidden_dim, seq_len]
                    (x.size(-1), 1)
                ).squeeze(-1).mean(0)  # [hidden_dim]
                self.cpu_memory_bank[-update_size:] = (
                    0.9 * self.cpu_memory_bank[-update_size:] +
                    0.1 * new_memories.unsqueeze(0)
                )
            
            # Process through memory modules
            context = self.long_term['output'](context)
            context = self.long_term['norm'](x_reduced + context)
            
            # Process through persistent memory
            context = self.persistent['query'](context)
            context = self.persistent['output'](context)
            context = self.persistent['norm'](context)
            
            # Restore original dimension
            batch_size, seq_len, reduced_dim = context.shape
            context_flat = context.view(-1, reduced_dim)
            final_output_flat = self.cpu_dim_restore(context_flat)
            final_output = final_output_flat.view(batch_size, seq_len, -1)
            
            print("CPU memory processing completed successfully")
            print(f"Output shape: {final_output.shape}")
            print(f"Output stats - min: {final_output.min().item():.3f}, max: {final_output.max().item():.3f}")
            return final_output
        except Exception as e:
            print(f"Error in memory manager: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error location: {'GPU' if self.use_gpu else 'CPU'} forward pass")
            print(f"Input shape: {x.shape}")
            if mask is not None:
                print(f"Mask shape: {mask.shape}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise
