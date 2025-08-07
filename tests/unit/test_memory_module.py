import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.memory_module import MemoryModule

@pytest.fixture
def memory_module():
    """Fixture for MemoryModule"""
    return MemoryModule()

def test_basic_write_read(memory_module):
    """Test basic write and read operations"""
    # Test writing different entry types
    id1 = memory_module.write("test prompt", entry_type="prompt")
    id2 = memory_module.write("test output", entry_type="output", session_id="test_session")
    id3 = memory_module.write("test feedback", entry_type="feedback", score=4.5)
    
    # Test that we can read entries
    all_entries = memory_module.read_all()
    assert len(all_entries) >= 3
    
    # Test reading by type
    prompts = memory_module.read_by_type("prompt")
    assert len(prompts) >= 1
    
    outputs = memory_module.read_by_type("output")
    assert len(outputs) >= 1
    
    feedback = memory_module.read_by_type("feedback")
    assert len(feedback) >= 1

def test_training_data_extraction(memory_module):
    """Test training data preparation"""
    # Add some conversational data
    memory_module.write("What is AI?", entry_type="prompt", session_id="conv1")
    memory_module.write("AI is artificial intelligence...", entry_type="output", session_id="conv1")
    memory_module.write("Good explanation", entry_type="feedback", session_id="conv1", score=4.0)
    
    # Test training data extraction
    training_data = memory_module.get_training_data()
    assert len(training_data) >= 1
