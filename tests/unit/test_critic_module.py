import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from critic_module import CriticModule

@pytest.fixture
def critic():
    """Fixture for CriticModule"""
    return CriticModule()

def test_semantic_novelty_scoring(critic):
    """Test the semantic novelty scoring"""
    output = "This is a new and unique sentence."
    
    # Mock the memory module and its semantic search
    mock_memory = MagicMock()
    
    # Case 1: Very similar content exists
    mock_memory.search_semantic.return_value = [
        {'_additional': {'distance': 0.05}}, # very similar
        {'_additional': {'distance': 0.1}}
    ]
    low_novelty_score = critic._score_novelty_semantic(output, mock_memory)
    
    # Case 2: Very different content exists
    mock_memory.search_semantic.return_value = [
        {'_additional': {'distance': 0.8}}, # very different
        {'_additional': {'distance': 0.9}}
    ]
    high_novelty_score = critic._score_novelty_semantic(output, mock_memory)
    
    # Case 3: No similar content
    mock_memory.search_semantic.return_value = []
    no_content_score = critic._score_novelty_semantic(output, mock_memory)

    assert high_novelty_score > low_novelty_score
    assert no_content_score == 5.0

def test_semantic_memory_alignment(critic):
    """Test memory alignment with semantic search"""
    prompt = "Tell me about dogs"
    output = "Dogs are loyal companions."
    
    mock_memory = MagicMock()
    mock_memory.search_semantic.return_value = [
        {'content': 'Canine pets are great.'},
        {'content': 'Wolves are ancestors of dogs.'}
    ]
    
    alignment_score = critic._score_memory_alignment(prompt, output, mock_memory)
    
    assert alignment_score > 3.0 # Should be a good score

def test_score_relevance(critic):
    """Test the relevance scoring"""
    prompt = "What is the capital of France?"
    good_response = "The capital of France is Paris."
    bad_response = "I like turtles."
    
    good_score = critic._score_relevance(prompt, good_response)
    bad_score = critic._score_relevance(prompt, bad_response)
    
    assert good_score > bad_score
    assert good_score > 4.0
    assert bad_score < 2.0

def test_overall_score(critic):
    """Test the overall weighted score"""
    prompt = "Explain photosynthesis"
    output = "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy."
    
    mock_memory = MagicMock()
    mock_memory.search_semantic.return_value = [] # Assume high novelty
    
    # Mock the individual score components
    with patch.object(critic, '_score_coherence', return_value=4.5), \
         patch.object(critic, '_score_novelty_semantic', return_value=5.0), \
         patch.object(critic, '_score_memory_alignment', return_value=4.0), \
         patch.object(critic, '_score_relevance', return_value=4.8), \
         patch.object(critic, '_score_semantic_relevance', return_value=4.5):

        score = critic.score(prompt, output, mock_memory)

        # Updated expected score with new weights including semantic relevance:
        # 0.08*4.5 + 0.22*5.0 + 0.25*4.0 + 0.30*4.8 + 0.15*4.5
        # = 0.36 + 1.10 + 1.00 + 1.44 + 0.675 = 4.575
        assert abs(score - 4.575) < 0.01
