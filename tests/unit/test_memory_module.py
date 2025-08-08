import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.memory_module import MemoryModule

@pytest.fixture
def memory_module():
    """Fixture for MemoryModule with a mocked VectorMemory.

    Avoids using @patch decorator directly (which interferes with generator fixtures)
    by performing patching inside the fixture context.
    """
    with patch('src.memory_module.VectorMemory') as MockVectorMemory:
        with patch.object(MemoryModule, '__init__', lambda self: None):
            module = MemoryModule()
            module.session = MagicMock()
            module.vector_memory = MockVectorMemory.return_value
            yield module

def test_write_calls_vector_memory(memory_module):
    """Test that write method also calls vector_memory.add_entry"""
    # Mock the timestamp attribute on the entry object
    mock_entry = MagicMock()
    mock_entry.timestamp.isoformat.return_value = "2023-01-01T12:00:00"
    
    # When a new MemoryEntry is created, return our mock
    with patch('src.memory_module.MemoryEntry', return_value=mock_entry):
        memory_module.write("test content", "test_type", session_id="s1", score=4.5)
    
    # Verify that the database session was used
    memory_module.session.add.assert_called_once()
    memory_module.session.commit.assert_called_once()
    
    # Verify that vector_memory was called
    memory_module.vector_memory.add_entry.assert_called_once()
    call_args = memory_module.vector_memory.add_entry.call_args[1]
    assert call_args['content'] == "test content"
    assert call_args['entry_type'] == "test_type"
    assert call_args['timestamp'] == "2023-01-01T12:00:00"

def test_search_semantic_calls_vector_memory(memory_module):
    """Test that search_semantic method calls vector_memory.search"""
    memory_module.search_semantic("test query", limit=10, entry_types=["prompt"])
    
    memory_module.vector_memory.search.assert_called_once_with("test query", 10, ["prompt"])

def test_get_training_data(memory_module):
    """Test training data preparation"""
    # Mock the feedback entries from the database
    mock_feedback = MagicMock()
    mock_feedback.content = "FEEDBACK: What is AI? -> AI is artificial intelligence... | score=4.0"
    mock_feedback.score = 4.0
    
    memory_module.get_feedback_entries = MagicMock(return_value=[mock_feedback])
    
    training_data = memory_module.get_training_data()
    
    assert len(training_data) == 1
    prompt, output, score = training_data[0]
    assert prompt == "What is AI?"
    assert output == "AI is artificial intelligence..."
    assert score == 4.0
