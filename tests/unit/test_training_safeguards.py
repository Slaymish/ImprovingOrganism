import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.training_safeguards import TrainingSafeguards, KnowledgeRetentionTester, DiversityMonitor, AdapterManager

class TestTrainingSafeguards(unittest.TestCase):
    """Test Training Safeguards functionality"""
    
    def setUp(self):
        """Set up test safeguards"""
        try:
            self.safeguards = TrainingSafeguards()
            self.knowledge_tester = KnowledgeRetentionTester()
            self.diversity_monitor = DiversityMonitor()
            self.adapter_manager = AdapterManager()
        except ImportError as e:
            self.skipTest(f"Training safeguards not available: {e}")
    
    def test_knowledge_retention_tester(self):
        """Test knowledge retention testing components"""
        self.assertGreater(len(self.knowledge_tester.general_knowledge_tests), 0)
        self.assertGreater(len(self.knowledge_tester.reasoning_tests), 0)
        
        # Test that knowledge tests have required structure
        for test in self.knowledge_tester.general_knowledge_tests:
            self.assertIn("prompt", test)
            self.assertIn("expected_contains", test)
            self.assertIsInstance(test["expected_contains"], list)
    
    def test_diversity_monitor(self):
        """Test diversity monitoring components"""
        self.assertGreater(len(self.diversity_monitor.diversity_prompts), 0)
        
        # Test lexical diversity calculation
        test_responses = [
            "This is a test response with unique words.",
            "Another different response with varied vocabulary.",
            "A third response using distinct terminology."
        ]
        diversity_score = self.diversity_monitor._calculate_lexical_diversity(test_responses)
        self.assertGreater(diversity_score, 0)
        self.assertLessEqual(diversity_score, 1.0)
    
    def test_adapter_manager(self):
        """Test adapter management functionality"""
        self.assertEqual(self.adapter_manager.max_adapters, 5)
        
        # Test performance score calculation
        test_metadata = {
            "training_loss": 0.5,
            "training_examples": 100,
            "training_time_minutes": 30
        }
        score = self.adapter_manager._calculate_performance_score(test_metadata)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_conservative_training_args(self):
        """Test conservative training argument generation"""
        # Test different dataset sizes
        test_sizes = [25, 75, 150, 500]
        
        for size in test_sizes:
            args = self.safeguards.get_conservative_training_args(size)
            
            # Check that all required arguments are present
            required_args = [
                "num_train_epochs", "learning_rate", "per_device_train_batch_size",
                "gradient_accumulation_steps", "warmup_steps", "weight_decay"
            ]
            for arg in required_args:
                self.assertIn(arg, args)
            
            # Check that parameters are reasonable
            self.assertGreater(args["num_train_epochs"], 0)
            self.assertLessEqual(args["num_train_epochs"], 5)
            self.assertGreater(args["learning_rate"], 0)
            self.assertLessEqual(args["learning_rate"], 1e-3)
    
    def test_data_validation(self):
        """Test training data validation"""
        # Test with insufficient data
        small_dataset = [
            {"score": 4.0, "prompt": "test", "output": "response"}
        ]
        validation = self.safeguards.validate_before_training(small_dataset)
        self.assertFalse(validation["can_proceed"])
        self.assertGreater(len(validation["errors"]), 0)
        
        # Test with sufficient good data
        good_dataset = [
            {"score": 4.0 + i*0.1, "prompt": f"test prompt {i}", "output": f"response {i}"}
            for i in range(15)
        ]
        validation = self.safeguards.validate_before_training(good_dataset)
        self.assertTrue(validation["can_proceed"])
        
        # Test with low quality data
        poor_dataset = [
            {"score": 2.0, "prompt": f"test {i}", "output": f"response {i}"}
            for i in range(15)
        ]
        validation = self.safeguards.validate_before_training(poor_dataset)
        self.assertGreater(len(validation["warnings"]), 0)
    
    def test_early_stopping_detection(self):
        """Test early stopping logic"""
        # Test loss explosion
        explosion_logs = [
            {"train_loss": 1.0},
            {"train_loss": 2.0},
            {"train_loss": 15.0},  # Loss explosion
            {"train_loss": 20.0},
            {"train_loss": 25.0}
        ]
        should_stop, reason = self.safeguards.should_stop_training_early(explosion_logs)
        self.assertTrue(should_stop)
        self.assertIn("exploded", reason.lower())
        
        # Test loss stagnation
        stagnation_logs = [
            {"train_loss": 1.000},
            {"train_loss": 1.000},
            {"train_loss": 1.000},
            {"train_loss": 1.000},
            {"train_loss": 1.000}
        ]
        should_stop, reason = self.safeguards.should_stop_training_early(stagnation_logs)
        self.assertTrue(should_stop)
        self.assertIn("stagnated", reason.lower())
        
        # Test normal training (should not stop)
        normal_logs = [
            {"train_loss": 2.0},
            {"train_loss": 1.8},
            {"train_loss": 1.6},
            {"train_loss": 1.4},
            {"train_loss": 1.2}
        ]
        should_stop, reason = self.safeguards.should_stop_training_early(normal_logs)
        self.assertFalse(should_stop)

if __name__ == '__main__':
    unittest.main()
