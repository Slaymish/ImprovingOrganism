import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.critic_module import CriticModule

class TestCriticModule(unittest.TestCase):
    """Test CriticModule scoring functionality"""
    
    def setUp(self):
        self.critic = CriticModule()
    
    def test_comprehensive_scoring(self):
        """Test multi-dimensional scoring system"""
        prompt = "Explain quantum computing"
        response = "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot."
        context = ["Previous discussion about classical computing", "User interested in advanced topics"]
        
        score = self.critic.score(prompt, response, context)
        
        # Should be a float between 0 and 5
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 5.0)
    
    def test_score_components(self):
        """Test individual scoring components"""
        prompt = "What is 2+2?"
        good_response = "2+2 equals 4. This is basic arithmetic addition where we combine two quantities of 2 to get a sum of 4."
        bad_response = "flying purple monkeys"
        
        good_score = self.critic.score(prompt, good_response, [])
        bad_score = self.critic.score(prompt, bad_response, [])
        
        # Good response should score higher
        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, 3.0)  # Should be reasonable 
        self.assertLess(bad_score, 2.5)      # Should be poor due to irrelevance

if __name__ == '__main__':
    unittest.main()
