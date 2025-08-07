import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.training_safeguards import TrainingSafeguards, KnowledgeRetentionTester, DiversityMonitor, AdapterManager

@pytest.fixture
def safeguards():
    """Fixture for TrainingSafeguards"""
    try:
        return TrainingSafeguards()
    except ImportError as e:
        pytest.skip(f"Training safeguards not available: {e}")

@pytest.fixture
def knowledge_tester():
    """Fixture for KnowledgeRetentionTester"""
    try:
        return KnowledgeRetentionTester()
    except ImportError as e:
        pytest.skip(f"Training safeguards not available: {e}")

@pytest.fixture
def diversity_monitor():
    """Fixture for DiversityMonitor"""
    try:
        return DiversityMonitor()
    except ImportError as e:
        pytest.skip(f"Training safeguards not available: {e}")

@pytest.fixture
def adapter_manager():
    """Fixture for AdapterManager"""
    try:
        return AdapterManager()
    except ImportError as e:
        pytest.skip(f"Training safeguards not available: {e}")

def test_knowledge_retention_tester(knowledge_tester):
    """Test knowledge retention testing components"""
    assert len(knowledge_tester.general_knowledge_tests) > 0
    assert len(knowledge_tester.reasoning_tests) > 0
    
    # Test that knowledge tests have required structure
    for test in knowledge_tester.general_knowledge_tests:
        assert "prompt" in test
        assert "expected_contains" in test
        assert isinstance(test["expected_contains"], list)

def test_diversity_monitor(diversity_monitor):
    """Test diversity monitoring components"""
    assert len(diversity_monitor.diversity_prompts) > 0
    
    # Test lexical diversity calculation
    test_responses = [
        "This is a test response with unique words.",
        "Another different response with varied vocabulary.",
        "A third response using distinct terminology."
    ]
    diversity_score = diversity_monitor._calculate_lexical_diversity(test_responses)
    assert 0 < diversity_score <= 1.0

def test_adapter_manager(adapter_manager):
    """Test adapter management functionality"""
    assert adapter_manager.max_adapters == 5
    
    # Test performance score calculation
    test_metadata = {
        "training_loss": 0.5,
        "training_examples": 100,
        "training_time_minutes": 30
    }
    score = adapter_manager._calculate_performance_score(test_metadata)
    assert 0 < score <= 1.0

def test_conservative_training_args(safeguards):
    """Test conservative training argument generation"""
    # Test different dataset sizes
    test_sizes = [25, 75, 150, 500]
    
    for size in test_sizes:
        args = safeguards.get_conservative_training_args(size)
        
        # Check that all required arguments are present
        required_args = [
            "num_train_epochs", "learning_rate", "per_device_train_batch_size",
            "gradient_accumulation_steps", "warmup_steps", "weight_decay"
        ]
        for arg in required_args:
            assert arg in args
        
        # Check that parameters are reasonable
        assert 0 < args["num_train_epochs"] <= 5
        assert 0 < args["learning_rate"] <= 1e-3

def test_data_validation(safeguards):
    """Test training data validation"""
    # Test with insufficient data
    small_dataset = [
        {"score": 4.0, "prompt": "test", "output": "response"}
    ]
    validation = safeguards.validate_before_training(small_dataset)
    assert not validation["can_proceed"]
    assert len(validation["errors"]) > 0
    
    # Test with sufficient good data
    good_dataset = [
        {"score": 4.0 + i*0.1, "prompt": f"test prompt {i}", "output": f"response {i}"}
        for i in range(15)
    ]
    validation = safeguards.validate_before_training(good_dataset)
    assert validation["can_proceed"]
    
    # Test with low quality data
    poor_dataset = [
        {"score": 2.0, "prompt": f"test {i}", "output": f"response {i}"}
        for i in range(15)
    ]
    validation = safeguards.validate_before_training(poor_dataset)
    assert len(validation["warnings"]) > 0

def test_early_stopping_detection(safeguards):
    """Test early stopping logic"""
    # Test loss explosion
    explosion_logs = [
        {"train_loss": 1.0},
        {"train_loss": 2.0},
        {"train_loss": 15.0},  # Loss explosion
        {"train_loss": 20.0},
        {"train_loss": 25.0}
    ]
    should_stop, reason = safeguards.should_stop_training_early(explosion_logs)
    assert should_stop
    assert "exploded" in reason.lower()
    
    # Test loss stagnation
    stagnation_logs = [
        {"train_loss": 1.000},
        {"train_loss": 1.001},
        {"train_loss": 1.000},
        {"train_loss": 1.001},
        {"train_loss": 1.000}
    ]
    should_stop, reason = safeguards.should_stop_training_early(stagnation_logs)
    assert should_stop
    assert "stagnated" in reason.lower()
    
    # Test normal training
    normal_logs = [
        {"train_loss": 2.0},
        {"train_loss": 1.5},
        {"train_loss": 1.2},
        {"train_loss": 1.0},
        {"train_loss": 0.8}
    ]
    should_stop, reason = safeguards.should_stop_training_early(normal_logs)
    assert not should_stop
