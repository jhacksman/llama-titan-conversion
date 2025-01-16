"""
Test suite for the Titans architecture implementation.

This module contains tests for verifying the functionality and performance
of the Titans architecture components, including memory modules and
integration tests.
"""

import unittest
import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
import os
import time
from typing import Optional, Tuple, Dict
from ..titan_transformer import TitanTransformer, TitanConfig
from ..memory_modules import (
    MemoryConfig,
    MemoryManager,
    LongTermMemory,
    PersistentMemory
)
from ..memory_utils import optimize_memory_distribution
from ..llama.model import ModelArgs, precompute_freqs_cis


class TestTitanModelGPU(unittest.TestCase):
    """GPU-specific test cases for the Titan model implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment once for all tests."""
        # Skip all tests if CUDA is not available
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
            
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:29501",
                world_size=1,
                rank=0
            )
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)  # Single GPU for testing
        
    def setUp(self):
        """Set up test environment."""
        # Ensure CUDA is available or skip test
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.config = TitanConfig(
            # LLaMA model parameters
            dim=4096,
            n_layers=32,
            n_heads=32,
            vocab_size=32000,
            max_batch_size=32,
            max_seq_len=2048,
            
            # Memory configuration
            memory_config=MemoryConfig(
                hidden_dim=4096,
                max_history_len=1000000,
                knowledge_dim=4096,
                num_memory_heads=32,
                dropout=0.1,
                update_interval=100
            ),
            vram_budget=64 * (1024 ** 3),
            n_gpus=torch.cuda.device_count() or 1
        )
        self.model = TitanTransformer(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_model_initialization(self):
        """Test model initialization."""
        # Verify component creation
        self.assertIsNotNone(self.model.memory_manager)
        self.assertEqual(len(self.model.layers), self.config.n_layers)
        
        # Check memory module setup
        self.assertTrue(hasattr(self.model.memory_manager, "long_term"))
        self.assertTrue(hasattr(self.model.memory_manager, "persistent"))
        
        # Validate configurations
        self.assertEqual(
            self.model.memory_manager.config.hidden_dim,
            self.config.memory_config.hidden_dim
        )

    def test_memory_distribution(self):
        """Test memory distribution across GPUs."""
        batch_size = 4
        seq_len = 512
        
        # Create sample input
        input_ids = torch.randint(
            0, 32000,  # Default vocab size
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
            0, 32000,  # Default vocab size
            (batch_size, seq_len),
            device=self.device
        )
        
        # Test basic forward pass
        with torch.no_grad():
            output = self.model(input_ids, start_pos=0)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(
            tuple(output.shape),
            expected_shape,
            f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
        )
        
        # Test extended sequence
        long_seq_len = 4096
        long_input = torch.randint(
            0, 32000,  # Default vocab size
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
            0, 32000,  # Default vocab size
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


class TestMemoryModulesGPU(unittest.TestCase):
    """GPU-specific test cases for memory module implementations."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment once for all tests."""
        # Skip all tests if CUDA is not available
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
            
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:29501",
                world_size=1,
                rank=0
            )
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)  # Single GPU for testing
        
    def setUp(self):
        """Set up test environment."""
        # Ensure CUDA is available or skip test
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
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
        print("\nTesting long-term memory module...")
        try:
            memory = LongTermMemory(self.config).to(self.device)
            print("Memory module initialized")
            
            # Create sample input
            x = torch.randn(
                self.batch_size,
                self.seq_len,
                self.config.hidden_dim,
                device=self.device
            )
            print(f"Created input tensor with shape {x.shape}")
            
            # Test forward pass
            print("Running forward pass...")
            output, stats = memory(x)
            print(f"Forward pass complete. Output shape: {output.shape}")
            
            self.assertEqual(
                output.shape,
                x.shape,
                "Output shape mismatch"
            )
            print("Shape check passed")
        except Exception as e:
            print(f"Error in long-term memory test: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise
        
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
        print("\nTesting persistent memory module...")
        try:
            memory = PersistentMemory(self.config).to(self.device)
            print("Memory module initialized")
            
            # Create sample input
            x = torch.randn(
                self.batch_size,
                self.seq_len,
                self.config.hidden_dim,
                device=self.device
            )
            print(f"Created input tensor with shape {x.shape}")
            
            # Test basic functionality
            print("Running forward pass...")
            output = memory(x)
            print(f"Forward pass complete. Output shape: {output.shape}")
            
            self.assertEqual(
                output.shape,
                x.shape,
                "Output shape mismatch"
            )
            print("Shape check passed")
        except Exception as e:
            print(f"Error in persistent memory test: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise
        
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


    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if fs_init.model_parallel_is_initialized():
            # Reset model parallel state
            fs_init._MODEL_PARALLEL_GROUP = None
            fs_init._TENSOR_MODEL_PARALLEL_GROUP = None

class TestTitanModelCPU(unittest.TestCase):
    """CPU-specific test cases for the Titan model implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create model args for CPU testing
        self.model_args = ModelArgs(
            dim=128,  # Smaller model for CPU tests
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            multiple_of=64,  # Smaller multiple for CPU tests
            norm_eps=1e-5,
            max_batch_size=2,
            max_seq_len=128
        )
        
        # Create Titan config
        self.config = TitanConfig(
            dim=self.model_args.dim,
            n_layers=self.model_args.n_layers,
            n_heads=self.model_args.n_heads,
            vocab_size=self.model_args.vocab_size,
            max_batch_size=self.model_args.max_batch_size,
            max_seq_len=self.model_args.max_seq_len,
            memory_config=MemoryConfig(
                hidden_dim=self.model_args.dim,
                max_history_len=100,
                knowledge_dim=self.model_args.dim,
                num_memory_heads=self.model_args.n_heads,
                dropout=0.1,
                update_interval=10
            )
        )
        
        # Initialize model with ModelArgs
        self.model = TitanTransformer(self.model_args)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_model_initialization(self):
        """Test model initialization on CPU."""
        self.assertIsNotNone(self.model.memory_manager)
        self.assertEqual(len(self.model.layers), self.config.n_layers)
        self.assertTrue(hasattr(self.model.memory_manager, "long_term"))
        self.assertTrue(hasattr(self.model.memory_manager, "persistent"))

    def test_forward_pass(self):
        """Test model forward pass on CPU with detailed logging."""
        print("\nStarting CPU forward pass test...")
        
        try:
            print("Initializing test parameters...")
            batch_size = 2
            seq_len = 16  # Small sequence length for CPU tests
            print(f"Test config: batch_size={batch_size}, seq_len={seq_len}")
            
            print("Creating input tokens...")
            input_ids = torch.randint(
                0, self.model_args.vocab_size,
                (batch_size, seq_len),
                device=self.device
            )
            print("Input tokens created successfully")
            
            print("Testing without mask first (shorter sequence)...")
            with torch.no_grad():
                try:
                    output_no_mask = self.model(
                        tokens=input_ids[:, :seq_len//2],
                        start_pos=0
                    )
                    print(f"Short sequence output shape: {output_no_mask.shape}")
                except Exception as e:
                    print(f"Error in short sequence test: {str(e)}")
                    raise
        
            # Check output shape
            print("Checking output shapes...")
            expected_shape_short = (batch_size, seq_len//2, self.model_args.vocab_size)
            self.assertEqual(
                tuple(output_no_mask.shape),
                expected_shape_short,
                f"Expected output shape {expected_shape_short}, got {tuple(output_no_mask.shape)}"
            )
            
            # Define expected shape for full sequence
            expected_shape = (batch_size, seq_len, self.model_args.vocab_size)
            
            print("Testing with different mask types...")
            # 1. Boolean causal mask
            bool_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            print("Created boolean mask")
            
            # 2. Float causal mask
            float_mask = torch.full(
                (seq_len, seq_len),
                float('-inf'),
                device=self.device
            )
            float_mask = torch.triu(float_mask, diagonal=1)
            print("Created float mask")
            
            # Test both mask types
            for mask_type, test_mask in [("boolean", bool_mask), ("float", float_mask)]:
                print(f"\nTesting with {mask_type} mask...")
                try:
                    with torch.no_grad():
                        output_masked = self.model(
                            tokens=input_ids,
                            start_pos=0,
                            mask=test_mask
                        )
                        print(f"Forward pass successful with {mask_type} mask")
                
                    # Check output shape
                    self.assertEqual(
                        tuple(output_masked.shape),
                        expected_shape,
                        f"Output shape mismatch with mask type {test_mask.dtype}"
                    )
                    print(f"Shape check passed for {mask_type} mask")
                    
                    # Basic sanity checks
                    self.assertFalse(
                        torch.isnan(output_masked).any(),
                        f"Output contains NaN values with mask type {test_mask.dtype}"
                    )
                    self.assertFalse(
                        torch.isinf(output_masked).any(),
                        f"Output contains infinite values with mask type {test_mask.dtype}"
                    )
                    print(f"Sanity checks passed for {mask_type} mask")
                except Exception as e:
                    print(f"Error testing {mask_type} mask: {str(e)}")
                    raise
        except Exception as e:
            print(f"Error in forward pass test: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise


class TestMemoryModulesCPU(unittest.TestCase):
    """CPU-specific test cases for memory module implementations."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MemoryConfig(
            hidden_dim=64,
            max_history_len=50,
            knowledge_dim=64,
            num_memory_heads=2,
            dropout=0.1
        )
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.seq_len = 16

    def test_long_term_memory(self):
        """Test long-term memory module on CPU."""
        memory = LongTermMemory(self.config).to(self.device)
        
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.config.hidden_dim,
            device=self.device
        )
        
        output, stats = memory(x)
        self.assertEqual(
            output.shape,
            x.shape,
            "Output shape mismatch"
        )

    def test_persistent_memory(self):
        """Test persistent memory module on CPU."""
        memory = PersistentMemory(self.config).to(self.device)
        
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.config.hidden_dim,
            device=self.device
        )
        
        output = memory(x)
        self.assertEqual(
            output.shape,
            x.shape,
            "Output shape mismatch"
        )


if __name__ == '__main__':
    unittest.main()
