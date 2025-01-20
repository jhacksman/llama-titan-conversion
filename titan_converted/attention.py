"""
Multi-Headed Attention Layer (MLA) implementation based on DeepSeek-R1.

This module implements:
1. Low-rank query/key projections
2. Efficient key-value caching
3. Rotary embeddings support
4. Memory-optimized attention patterns
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .memory.memory_config import MemoryConfig


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Precompute frequency-based complex exponential values for rotary embeddings.
    
    Args:
        dim: Dimension of the embeddings
        seq_len: Maximum sequence length
        theta: Base for exponential computation
        scaling_factor: Scaling factor for extended sequences
    
    Returns:
        torch.Tensor: Precomputed complex exponential values
    """
    # Compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Scale frequencies for extended sequences if needed
    if scaling_factor != 1.0:
        freqs = freqs / scaling_factor
    
    # Create position-dependent frequencies
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    
    # Convert to complex exponentials
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.
    
    Args:
        x: Input tensor [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed complex exponentials
        
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
    # Convert to complex representation
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Apply rotary embeddings
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real
    return torch.view_as_real(x_rotated).flatten(3).to(x.dtype)


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer with low-rank projections and rotary embeddings.
    
    This implementation follows DeepSeek-R1's architecture with:
    1. Separate head dimensions for different projection types
    2. Low-rank approximations for efficiency
    3. Rotary embeddings for position-aware attention
    """
    def __init__(
        self,
        config: MemoryConfig,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.num_attention_heads
        
        # Head dimensions
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # Total head dimension for QK
        
        # Low-rank projections
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        
        # Calculate total dimensions
        total_qk_dim = self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        total_v_dim = self.n_heads * self.v_head_dim
        
        # Query projections
        if q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, total_qk_dim)
        else:
            self.wq_a = nn.Linear(self.dim, q_lora_rank)
            self.q_norm = nn.LayerNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, total_qk_dim)
        
        # Key-value projections with low rank
        # Adjust dimensions to match query size
        kv_a_out_dim = kv_lora_rank + (self.n_heads * self.qk_rope_head_dim)  # Scale rotary for all heads
        self.wkv_a = nn.Linear(self.dim, kv_a_out_dim)
        
        # Only normalize the non-rotary part (kv_lora_rank portion)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)
        
        # Separate projections for key and value
        # Key output should match query dimension (qk_nope_head_dim per head)
        self.wk = nn.Linear(
            kv_lora_rank,
            self.n_heads * self.qk_nope_head_dim  # Only non-rotary part
        )
        # Value projection
        self.wv = nn.Linear(
            kv_lora_rank,
            self.n_heads * self.v_head_dim
        )
        
        # Calculate and store head dimensions
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # Total head dimension for QK
        self.rotary_ndims = self.qk_rope_head_dim  # Per head
        
        # Store expected output dimensions
        self.total_k_dim = self.n_heads * self.qk_head_dim  # Total dimension across all heads
        self.total_v_dim = self.n_heads * self.v_head_dim  # Total dimension across all heads
        
        # Output projection using total dimensions (across all heads)
        self.wo = nn.Linear(self.total_v_dim, self.dim)
        
        # Attention scaling using per-head dimensions
        self.scale = self.qk_head_dim ** -0.5
        
        # Initialize key-value cache with correct dimensions
        self.k_cache = None
        self.v_cache = None
        
        # Verify dimensions
        assert self.total_k_dim == total_qk_dim, "Key dimension mismatch"
        assert self.total_v_dim == total_v_dim, "Value dimension mismatch"
        
        # Calculate total dimensions
        self.total_qk_dim = self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        self.total_v_dim = self.n_heads * self.v_head_dim
    
    def _compute_query(self, x: torch.Tensor) -> torch.Tensor:
        """Compute query vectors with optional low-rank projection."""
        if self.q_lora_rank == 0:
            return self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.q_norm(q)
            return self.wq_b(q)
    
    def _compute_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute key and value vectors with low-rank projection."""
        # Get input dimensions
        batch_size, seq_len, _ = x.shape
        
        # Initial projection with correct dimensions
        kv = self.wkv_a(x)  # [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]
        
        # Split rotary and non-rotary parts (rotary part is already scaled for n_heads)
        rotary_dim = self.n_heads * self.qk_rope_head_dim
        k_rope = kv[..., :rotary_dim]  # [batch_size, seq_len, n_heads * qk_rope_head_dim]
        kv_rest = kv[..., rotary_dim:]  # [batch_size, seq_len, kv_lora_rank]
        
        # Process through low-rank projection for non-rotary part
        kv_rest = kv_rest.reshape(-1, kv_rest.size(-1))  # Reshape to [batch_size * seq_len, kv_lora_rank]
        kv_rest = self.kv_norm(kv_rest)
        kv_rest = kv_rest.reshape(batch_size, seq_len, -1)  # Restore shape
        
        # Project to key and value separately using new projections
        k_nope = self.wk(kv_rest)  # [batch_size, seq_len, n_heads * qk_nope_head_dim]
        v = self.wv(kv_rest)  # [batch_size, seq_len, n_heads * v_head_dim]
        
        # No need to expand k_rope as it's already scaled for n_heads from wkv_a
        
        # Combine rotary and non-rotary key parts
        k = torch.cat([k_nope, k_rope], dim=-1)  # [batch_size, seq_len, n_heads * (qk_nope_head_dim + qk_rope_head_dim)]
        
        return k, v
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing multi-headed attention with memory optimization.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            freqs_cis: Optional precomputed rotary embeddings
            mask: Optional attention mask
            
        Returns:
            torch.Tensor: Output tensor after attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute query vectors
        q = self._compute_query(x)
        
        # Compute key and value vectors
        k, v = self._compute_kv(x)
        
        # Reshape for attention heads
        q = q.view(batch_size, seq_len, self.n_heads, -1)  # [batch_size, seq_len, n_heads, qk_head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, -1)  # [batch_size, seq_len, n_heads, qk_head_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.v_head_dim)  # [batch_size, seq_len, n_heads, v_head_dim]
        
        # Verify per-head dimensions
        assert q.size(-1) == self.qk_head_dim, f"Query head dim mismatch: {q.size(-1)} vs {self.qk_head_dim}"
        assert k.size(-1) == self.qk_head_dim, f"Key head dim mismatch: {k.size(-1)} vs {self.qk_head_dim}"
        assert v.size(-1) == self.v_head_dim, f"Value head dim mismatch: {v.size(-1)} vs {self.v_head_dim}"
        
        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            # Extract rotary portions
            q_rope = q[..., -self.qk_rope_head_dim:]  # Take last qk_rope_head_dim dimensions
            k_rope = k[..., -self.qk_rope_head_dim:]
            
            # Apply rotary embeddings
            q_rope = apply_rotary_emb(q_rope, freqs_cis)
            k_rope = apply_rotary_emb(k_rope, freqs_cis)
            
            # Recombine with non-rotary portions
            q = torch.cat([q[..., :-self.qk_rope_head_dim], q_rope], dim=-1)
            k = torch.cat([k[..., :-self.qk_rope_head_dim], k_rope], dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        
        # Verify dimensions after reshape
        assert q.size(-1) == self.qk_head_dim, f"Query head dim mismatch: {q.size(-1)} vs {self.qk_head_dim}"
        assert k.size(-1) == self.qk_head_dim, f"Key head dim mismatch: {k.size(-1)} vs {self.qk_head_dim}"
        assert v.size(-1) == self.v_head_dim, f"Value head dim mismatch: {v.size(-1)} vs {self.v_head_dim}"
        
        # Compute attention scores with verified dimensions
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out)
