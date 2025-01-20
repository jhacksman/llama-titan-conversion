"""
Integration tests for DeepSeek-Titans architecture.

This test suite verifies:
1. Context window handling (128K -> 2M+)
2. Memory usage across GPUs (64GB total)
3. Basic model behaviors
4. Memory module integration
"""

import unittest
import math
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..config import (
    create_config,
    DeepSeekTitanConfig,
    ModelConfig,
    HardwareConfig,
    MemoryConfig,
    MoEConfig
)
from ..titan_deepseek import create_titan_model, TitanTransformer
from ..memory import (
    CoreMemory,
    LongTermMemory,
    PersistentMemory,
    MemoryManager
)


class TestTitansIntegration(unittest.TestCase):
    """Test suite for DeepSeek-Titans integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.config = create_config(
            model=ModelConfig(
                dim=4096,
                n_layers=32,
                n_heads=32,
                vocab_size=32000,
                max_seq_len=2097152  # 2M context
            ),
            hardware=HardwareConfig(
                total_vram=64 * (1024 ** 3),  # 64GB
                num_gpus=3
            )
        )
        
        # Create model
        cls.model = create_titan_model(
            base_config=cls.config.model.__dict__,
            memory_config=cls.config.memory,
            vram_budget=cls.config.hardware.total_vram,
            num_gpus=cls.config.hardware.num_gpus
        )
        
        # Move to CPU for testing
        cls.model = cls.model.cpu()
        cls.device = torch.device("cpu")
    
    def setUp(self):
        """Reset CUDA memory stats before each test."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        self.assertTrue(self.config.validate())
        
        # Test invalid VRAM budget
        with self.assertRaises(ValueError):
            invalid_config = create_config(
                hardware={"total_vram": 10 * (1024 ** 3)}  # Only 10GB
            )
    
    def test_memory_allocation(self):
        """Test memory allocation across components."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Move model to CUDA
        model = self.model.cuda()
        
        # Generate test input
        batch_size = 4
        seq_len = 1024
        input_ids = torch.randint(
            0,
            self.config.model.vocab_size,
            (batch_size, seq_len),
            device="cuda"
        )
        
        # Forward pass
        _ = model(input_ids)
        
        # Check memory usage
        max_memory = torch.cuda.max_memory_allocated()
        self.assertLess(
            max_memory,
            self.config.hardware.total_vram,
            "Memory usage exceeds budget"
        )
        
        # Check per-component allocation
        memory_stats = {
            f"gpu_{i}": torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        }
        
        for gpu_id, allocated in memory_stats.items():
            self.assertLess(
                allocated,
                self.config.hardware.vram_per_gpu,
                f"Memory on {gpu_id} exceeds per-GPU budget"
            )
    
    def test_context_window(self):
        """Test handling of large context windows."""
        # Test increasing context lengths
        test_lengths = [
            128 * 1024,  # 128K
            512 * 1024,  # 512K
            2 * 1024 * 1024  # 2M
        ]
        
        for length in test_lengths:
            with self.subTest(context_length=length):
                # Generate test input
                batch_size = 1
                input_ids = torch.randint(
                    0,
                    self.config.model.vocab_size,
                    (batch_size, length),
                    device=self.device
                )
                
                # Verify model can process input
                try:
                    _ = self.model(input_ids)
                except RuntimeError as e:
                    self.fail(
                        f"Failed to process context length {length}: {str(e)}"
                    )
    
    def test_memory_modules(self):
        """Test individual memory modules."""
        batch_size = 4
        seq_len = 1024
        hidden_dim = self.config.model.dim
        
        # Test input
        x = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.ones(batch_size, seq_len)
        
        # Test CoreMemory
        core = CoreMemory(self.config.memory)
        core_out, core_stats = core(x, mask)
        self.assertEqual(
            core_out.shape,
            (batch_size, seq_len, hidden_dim),
            "CoreMemory output shape mismatch"
        )
        
        # Test LongTermMemory
        ltm = LongTermMemory(self.config.memory)
        ltm_out, ltm_stats = ltm(x, mask)
        self.assertEqual(
            ltm_out.shape,
            (batch_size, seq_len, hidden_dim),
            "LongTermMemory output shape mismatch"
        )
        
        # Test PersistentMemory
        pm = PersistentMemory(self.config.memory)
        pm_out = pm(x)
        self.assertEqual(
            pm_out.shape,
            (batch_size, seq_len, hidden_dim),
            "PersistentMemory output shape mismatch"
        )
    
    def test_basic_behavior(self):
        """Test basic model behaviors."""
        # Test input
        prompt = torch.randint(
            0,
            self.config.model.vocab_size,
            (1, 128),
            device=self.device
        )
        
        # Test autoregressive generation
        with torch.no_grad():
            output = self.model(prompt)
        
        self.assertEqual(
            output.shape,
            (1, 128, self.config.model.vocab_size),
            "Output shape mismatch"
        )
        
        # Test attention patterns
        attention_pattern = None
        for module in self.model.modules():
            if isinstance(module, CoreMemory):
                _, stats = module(
                    torch.randn(1, 128, self.config.model.dim),
                    torch.ones(1, 128)
                )
                attention_pattern = stats["attention_pattern"]
                break
        
        self.assertIsNotNone(
            attention_pattern,
            "No attention pattern found"
        )
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Move model to CUDA
        model = self.model.cuda()
        
        # Test gradient checkpointing
        found_checkpointing = False
        for module in model.modules():
            if hasattr(module, "checkpoint_core_attention"):
                found_checkpointing = True
                break
        self.assertTrue(
            found_checkpointing,
            "Gradient checkpointing not enabled"
        )
        
        # Test flash attention
        found_flash_attention = False
        for module in model.modules():
            if hasattr(module, "use_flash_attention"):
                found_flash_attention = True
                break
        self.assertTrue(
            found_flash_attention,
            "Flash attention not enabled"
        )
    
    def test_expert_routing(self):
        """Test MoE expert routing with memory integration."""
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(
            0,
            self.config.model.vocab_size,
            (batch_size, seq_len),
            device=self.device
        )
        
        # Get expert assignment stats
        expert_counts = {}
        for name, module in self.model.named_modules():
            if "moe" in name.lower():
                # Forward pass to trigger expert routing
                _ = module(torch.randn(batch_size, seq_len, self.config.model.dim))
                
                if hasattr(module, "expert_counts"):
                    expert_counts[name] = module.expert_counts
        
        # Verify expert utilization
        self.assertGreater(
            len(expert_counts),
            0,
            "No expert routing statistics found"
        )
        
        for name, counts in expert_counts.items():
            # Check expert utilization
            total_tokens = batch_size * seq_len
            active_experts = (counts > 0).sum().item()
            self.assertGreater(
                active_experts,
                0,
                f"No active experts found in {name}"
            )


if __name__ == "__main__":
    unittest.main()
