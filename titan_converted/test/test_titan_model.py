"""
Test suite for the Titans architecture implementation.

This module contains tests for verifying the functionality and performance
of the Titans architecture components, including memory modules and
integration tests.
"""

import unittest
import torch
import time
from typing import Optional, Tuple, Dict
from ..titan_model import TitanTransformer, TitanModelArgs
from ..memory_modules import (
    MemoryConfig,
    MemoryManager,
    LongTermMemory,
    PersistentMemory
)
from ..memory_utils import optimize_memory_distribution


class TestTitanModel(unittest.TestCase):
    """Test cases for the Titan model implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.args = TitanModelArgs(
            dim=4096,
            n_layers=32,
            n_heads=32,
            vocab_size=32000,
            max_batch_size=32,
            max_seq_len=2048
        )
        self.model = TitanTransformer(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_model_initialization(self):
        """Test model initialization."""
        # Verify component creation
        self.assertIsNotNone(self.model.memory_manager)
        self.assertEqual(len(self.model.layers), self.args.n_layers)
        
        # Check memory module setup
        self.assertTrue(hasattr(self.model.memory_manager, "long_term"))
        self.assertTrue(hasattr(self.model.memory_manager, "persistent"))
        
        # Validate configurations
        self.assertEqual(
            self.model.memory_manager.config.hidden_dim,
            self.args.dim
        )

    def test_memory_distribution(self):
        """Test memory distribution across GPUs."""
        batch_size = 4
        seq_len = 512
        
        # Create sample input
        input_ids = torch.randint(
            0, self.args.vocab_size,
            (batch_size, seq_len),
            device=self.device
        )
        
        # Track initial memory
        initial_memory = {
            i: torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        }
        
        # Run forward pass
        with torch.no_grad():
            _ = self.model(input_ids, start_pos=0)
        
        # Check memory allocation
        final_memory = {
            i: torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        }
        
        total_memory = sum(
            final_memory[i] - initial_memory[i]
            for i in range(torch.cuda.device_count())
        )
        
        # Verify within budget (64GB)
        self.assertLess(
            total_memory,
            64 * (1024 ** 3),
            "Memory usage exceeds 64GB budget"
        )

    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size = 2
        seq_len = 128
        
        # Create sample input
        input_ids = torch.randint(
            0, self.args.vocab_size,
            (batch_size, seq_len),
            device=self.device
        )
        
        # Test basic forward pass
        with torch.no_grad():
            output = self.model(input_ids, start_pos=0)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.args.vocab_size)
        self.assertEqual(
            tuple(output.shape),
            expected_shape,
            f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
        )
        
        # Test extended sequence
        long_seq_len = 4096
        long_input = torch.randint(
            0, self.args.vocab_size,
            (1, long_seq_len),
            device=self.device
        )
        
        with torch.no_grad():
            long_output = self.model(long_input, start_pos=0)
        
        self.assertEqual(
            long_output.shape[1],
            long_seq_len,
            "Failed to process extended sequence"
        )

    def test_memory_efficiency(self):
        """Test memory optimization effectiveness."""
        batch_size = 8
        seq_len = 256
        
        # Create sample input
        input_ids = torch.randint(
            0, self.args.vocab_size,
            (batch_size, seq_len),
            device=self.device
        )
        
        # Measure throughput
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):  # Multiple runs for better measurement
                _ = self.model(input_ids, start_pos=0)
        end_time = time.time()
        
        # Calculate tokens per second
        total_tokens = batch_size * seq_len * 5
        tokens_per_second = total_tokens / (end_time - start_time)
        
        # Log performance metrics
        print(f"\nPerformance Metrics:")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {seq_len}")
        
        # Basic throughput check
        self.assertGreater(
            tokens_per_second,
            100,  # Minimum acceptable tokens/second
            "Performance below minimum threshold"
        )


class TestMemoryModules(unittest.TestCase):
    """Test cases for memory module implementations."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MemoryConfig(
            hidden_dim=1024,
            max_history_len=1000,
            knowledge_dim=1024,
            num_memory_heads=4,
            dropout=0.1
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 128

    def test_long_term_memory(self):
        """Test long-term memory module."""
        memory = LongTermMemory(self.config).to(self.device)
        
        # Create sample input
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.config.hidden_dim,
            device=self.device
        )
        
        # Test forward pass
        output, stats = memory(x)
        self.assertEqual(
            output.shape,
            x.shape,
            "Output shape mismatch"
        )
        
        # Test memory retention
        first_output, _ = memory(x)
        second_output, _ = memory(x)
        self.assertFalse(
            torch.allclose(first_output, second_output),
            "Memory not updating between calls"
        )
        
        # Test memory capacity
        large_batch = torch.randn(
            4,
            self.config.max_history_len + 100,
            self.config.hidden_dim,
            device=self.device
        )
        output, _ = memory(large_batch)
        self.assertEqual(
            output.shape,
            large_batch.shape,
            "Failed to handle large input"
        )

    def test_persistent_memory(self):
        """Test persistent memory module."""
        memory = PersistentMemory(self.config).to(self.device)
        
        # Create sample input
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.config.hidden_dim,
            device=self.device
        )
        
        # Test basic functionality
        output = memory(x)
        self.assertEqual(
            output.shape,
            x.shape,
            "Output shape mismatch"
        )
        
        # Test with task ID
        task_id = torch.tensor([0, 1], device=self.device)
        task_output = memory(x, task_id)
        self.assertEqual(
            task_output.shape,
            x.shape,
            "Task-specific output shape mismatch"
        )
        
        # Test knowledge retention
        knowledge_query = torch.randn(
            1,
            1,
            self.config.hidden_dim,
            device=self.device
        )
        first_response = memory(knowledge_query)
        second_response = memory(knowledge_query)
        self.assertTrue(
            torch.allclose(first_response, second_response, atol=1e-6),
            "Persistent memory not stable for same query"
        )

    def test_memory_integration(self):
        """Test integration of both memory modules."""
        manager = MemoryManager(
            self.config,
            vram_budget=64 * (1024 ** 3),
            n_gpus=3
        )
        manager.to(self.device)
        
        # Create sample input
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.config.hidden_dim,
            device=self.device
        )
        
        # Test combined memory processing
        output = manager.forward(x)
        self.assertEqual(
            output.shape,
            x.shape,
            "Memory manager output shape mismatch"
        )
        
        # Test memory interaction
        task_id = torch.tensor([0] * self.batch_size, device=self.device)
        output_with_task = manager.forward(x, task_id=task_id)
        self.assertEqual(
            output_with_task.shape,
            x.shape,
            "Task-specific memory integration failed"
        )


if __name__ == '__main__':
    unittest.main()
