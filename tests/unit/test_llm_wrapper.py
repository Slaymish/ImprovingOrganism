import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_wrapper import LLMWrapper

@pytest.fixture
def llm_wrapper():
    """Fixture for LLMWrapper"""
    return LLMWrapper()

def test_mock_generation(llm_wrapper):
    """Test mock text generation"""
    prompt = "Explain artificial intelligence"
    response = llm_wrapper.generate(prompt)
    
    assert isinstance(response, str)
    assert len(response) > 10  # Should generate substantial text

def test_generation_parameters(llm_wrapper):
    """Test generation with different parameters"""
    prompt = "Write a short story"
    
    short_response = llm_wrapper.generate(prompt, max_tokens=50)
    long_response = llm_wrapper.generate(prompt, max_tokens=200)
    
    # Both should be strings
    assert isinstance(short_response, str)
    assert isinstance(long_response, str)
