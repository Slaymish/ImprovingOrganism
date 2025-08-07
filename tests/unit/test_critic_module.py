import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.critic_module import CriticModule

@pytest.fixture
def critic():
    """Fixture for CriticModule"""
    return CriticModule()

def test_comprehensive_scoring(critic):
    """Test multi-dimensional scoring system"""
    prompt = "Explain quantum computing"
    response = "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot."
    context = ["Previous discussion about classical computing", "User interested in advanced topics"]
    
    score = critic.score(prompt, response, context)
    
    # Should be a float between 0 and 5
    assert isinstance(score, float)
    assert 0.0 <= score <= 5.0

def test_score_components(critic):
    """Test individual scoring components"""
    prompt = "What is 2+2?"
    good_response = "2+2 equals 4. This is basic arithmetic addition where we combine two quantities of 2 to get a sum of 4."
    bad_response = "flying purple monkeys"
    
    good_score = critic.score(prompt, good_response, [])
    bad_score = critic.score(prompt, bad_response, [])
    
    # Good response should score higher
    assert good_score > bad_score
    assert good_score > 3.0  # Should be reasonable 
    assert bad_score < 2.5      # Should be poor due to irrelevance
