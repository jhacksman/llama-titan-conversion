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
    MoEConfig
)
from ..memory.memory_config import MemoryConfig
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
        # Use smaller dimensions for CPU testing
        cls.test_seq_len = 128  # Small context for CPU tests
        cls.target_seq_len = 2097152  # 2M context target (for validation only)
        
        # Create configuration with reduced dimensions for testing
        cls.config = create_config(
            model=ModelConfig(
                dim=256,  # Reduced dimension for CPU testing
                n_layers=2,  # Fewer layers for testing
                n_heads=4,  # Fewer heads
                vocab_size=1000,  # Smaller vocab
                max_seq_len=cls.test_seq_len
            ),
            hardware=HardwareConfig(
                total_vram=64 * (1024 ** 3),  # 64GB
                num_gpus=3
            ),
            memory=MemoryConfig(
                dim=256,  # Match model dimension
                intermediate_dim=1024,  # 4x model dim
                num_attention_heads=4,  # Match model heads
                num_memory_heads=2,  # Reduced for testing
                max_sequence_length=2097152,  # 2M+ context
                max_memory_length=1000,  # Reduced for testing
                memory_update_interval=100,
                vram_target_per_component=10 * (1024 ** 3),  # 10GB target
                vram_minimum_per_component=5 * (1024 ** 3),  # 5GB minimum
                total_vram_budget=64 * (1024 ** 3),  # 64GB total
                num_gpus=3,
                num_memory_experts=4  # Reduced for testing
            )
        )
        
        # Create model with minimal initialization
        cls.model = create_titan_model(
            base_config=cls.config.model.__dict__,
            memory_config=cls.config.memory,
            vram_budget=cls.config.hardware.total_vram,
            num_gpus=cls.config.hardware.num_gpus,
            initialize_weights=False  # Skip full initialization for testing
        )
        
        # Initialize memory manager
        cls.memory_manager = MemoryManager(cls.config.memory)
        
        # Move to CPU for testing
        cls.model = cls.model.cpu()
        cls.memory_manager = cls.memory_manager.cpu()
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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test with increasing sequence lengths
        batch_sizes = [1, 2, 4]
        seq_lengths = [1024, 4096, 8192]  # Test up to 8K for CPU testing
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                with self.subTest(batch_size=batch_size, seq_len=seq_len):
                    # Generate test input
                    input_ids = torch.randint(
                        0,
                        self.config.model.vocab_size,
                        (batch_size, seq_len),
                        device="cuda"
                    )
                    
                    # Forward pass
                    _ = model(input_ids)
                    
                    # Check total memory usage
                    max_memory = torch.cuda.max_memory_allocated()
                    self.assertLess(
                        max_memory,
                        self.config.hardware.total_vram,
                        f"Memory usage ({max_memory/1e9:.2f}GB) exceeds "
                        f"budget ({self.config.hardware.total_vram/1e9:.2f}GB) "
                        f"for batch_size={batch_size}, seq_len={seq_len}"
                    )
                    
                    # Check per-component allocation
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i)
                        self.assertLess(
                            allocated,
                            self.config.memory.vram_target_per_component,
                            f"Memory on GPU {i} ({allocated/1e9:.2f}GB) exceeds "
                            f"target ({self.config.memory.vram_target_per_component/1e9:.2f}GB)"
                        )
                        self.assertGreater(
                            allocated,
                            self.config.memory.vram_minimum_per_component,
                            f"Memory on GPU {i} ({allocated/1e9:.2f}GB) below "
                            f"minimum ({self.config.memory.vram_minimum_per_component/1e9:.2f}GB)"
                        )
                    
                    # Reset stats for next test
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
    
    def test_context_window(self):
        """Test handling of large context windows."""
        # Verify model configuration supports target context length
        self.assertEqual(
            self.model.max_context_length,
            self.target_seq_len,
            "Model configuration doesn't support target context length"
        )
        
        # Test with small context lengths for basic functionality
        test_lengths = [32, 64, 128]  # Small lengths for CPU testing
        
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
        
        # Verify memory scaling calculations
        mem_reqs = self.model.calculate_memory_requirements(
            batch_size=1,
            seq_length=self.target_seq_len
        )
        
        # Check total memory requirements
        self.assertLess(
            mem_reqs['total'],
            self.config.hardware.total_vram,
            "Memory requirements exceed hardware constraints"
        )
        
        # Check per-component requirements
        for component, req in mem_reqs.items():
            if component != 'total':
                self.assertGreaterEqual(
                    req,
                    self.config.memory.vram_minimum_per_component,
                    f"{component} requires less than minimum VRAM"
                )
                self.assertLessEqual(
                    req,
                    self.config.memory.vram_target_per_component,
                    f"{component} requires more than target VRAM"
                )
    
    def test_memory_modules(self):
        """Test individual memory modules."""
        batch_size = 4
        seq_len = 1024
        hidden_dim = self.config.model.dim
        
        # Test input
        x = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Create test memory config
        test_config = MemoryConfig(
            dim=hidden_dim,
            intermediate_dim=hidden_dim * 4,
            num_attention_heads=max(1, hidden_dim // 64),
            num_memory_heads=max(1, hidden_dim // 64),
            max_sequence_length=1024,
            max_memory_length=1000,
            num_memory_experts=max(1, hidden_dim // 128)
        )
        
        # Test CoreMemory
        core = CoreMemory(test_config)
        core_out, _ = core(x, mask)  # Unpack output and stats
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
        
        # For testing mode, output dimension matches input config
        self.assertEqual(
            output.shape,
            (1, 128, self.model.config['vocab_size']),
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
        hidden_dim = self.config.model.dim
        
        # Create test input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Run forward pass through memory manager
        _ = self.memory_manager(x, mask)
        
        # Verify expert routing statistics
        self.assertTrue(hasattr(self.memory_manager, "expert_counts"),
                       "Expert counts not tracked in MemoryManager")
        self.assertTrue(hasattr(self.memory_manager, "routing_stats"),
                       "Routing statistics not tracked in MemoryManager")
        
        # Check expert utilization
        expert_counts = self.memory_manager.expert_counts
        routing_stats = self.memory_manager.routing_stats
        
        self.assertEqual(
            len(expert_counts),
            self.config.memory.num_memory_experts,
            "Incorrect number of experts tracked"
        )
        
        # Verify active experts
        active_experts = (expert_counts > 0).sum().item()
        self.assertGreater(
            active_experts,
            0,
            "No active experts found"
        )
        self.assertLessEqual(
            active_experts,
            self.config.memory.num_memory_experts,
            "More active experts than configured"
        )
        
        # Verify total tokens processed
        self.assertEqual(
            routing_stats["total_tokens"],
            batch_size * seq_len,
            "Incorrect token count in routing statistics"
        )


if __name__ == "__main__":
    unittest.main()
