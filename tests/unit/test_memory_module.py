import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.memory_module import MemoryModule

class TestMemoryModule(unittest.TestCase):
    """Test MemoryModule functionality"""
    
    def setUp(self):
        """Set up test memory module"""
        self.memory = MemoryModule()
    
    def test_basic_write_read(self):
        """Test basic write and read operations"""
        # Test writing different entry types
        id1 = self.memory.write("test prompt", entry_type="prompt")
        id2 = self.memory.write("test output", entry_type="output", session_id="test_session")
        id3 = self.memory.write("test feedback", entry_type="feedback", score=4.5)
        
        # Test that we can read entries
        all_entries = self.memory.read_all()
        self.assertGreaterEqual(len(all_entries), 3)
        
        # Test reading by type
        prompts = self.memory.read_by_type("prompt")
        self.assertGreaterEqual(len(prompts), 1)
        
        outputs = self.memory.read_by_type("output")
        self.assertGreaterEqual(len(outputs), 1)
        
        feedback = self.memory.read_by_type("feedback")
        self.assertGreaterEqual(len(feedback), 1)
    
    def test_training_data_extraction(self):
        """Test training data preparation"""
        # Add some conversational data
        self.memory.write("What is AI?", entry_type="prompt", session_id="conv1")
        self.memory.write("AI is artificial intelligence...", entry_type="output", session_id="conv1")
        self.memory.write("Good explanation", entry_type="feedback", session_id="conv1", score=4.0)
        
        # Test training data extraction
        training_data = self.memory.get_training_data()
        self.assertGreaterEqual(len(training_data), 1)

if __name__ == '__main__':
    unittest.main()
