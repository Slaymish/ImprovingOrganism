import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_wrapper import LLMWrapper

class TestLLMWrapper(unittest.TestCase):
    """Test LLMWrapper functionality"""
    
    def setUp(self):
        self.llm = LLMWrapper()
    
    def test_mock_generation(self):
        """Test mock text generation"""
        prompt = "Explain artificial intelligence"
        response = self.llm.generate(prompt)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)  # Should generate substantial text
    
    def test_generation_parameters(self):
        """Test generation with different parameters"""
        prompt = "Write a short story"
        
        short_response = self.llm.generate(prompt, max_tokens=50)
        long_response = self.llm.generate(prompt, max_tokens=200)
        
        # Both should be strings
        self.assertIsInstance(short_response, str)
        self.assertIsInstance(long_response, str)

if __name__ == '__main__':
    unittest.main()
