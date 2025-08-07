import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.memory_module import MemoryModule
from src.critic_module import CriticModule
from src.updater import Updater
from src.llm_wrapper import LLMWrapper
from src.latent_workspace import LatentWorkspace

@pytest.fixture
def integrated_system():
    """Fixture for integrated system"""
    return {
        "memory": MemoryModule(),
        "critic": CriticModule(),
        "updater": Updater(),
        "llm": LLMWrapper(),
        "workspace": LatentWorkspace(dim=256)
    }

def test_end_to_end_conversation(integrated_system):
    """Test complete conversation flow"""
    llm = integrated_system["llm"]
    memory = integrated_system["memory"]
    critic = integrated_system["critic"]
    
    # Simulate a conversation
    prompt = "What is machine learning?"
    
    # Generate response
    response = llm.generate(prompt)
    
    # Store in memory
    session_id = "test_session"
    memory.write(prompt, entry_type="prompt", session_id=session_id)
    memory.write(response, entry_type="output", session_id=session_id)
    
    # Score the response
    context = [prompt]
    score = critic.score(prompt, response, context)
    
    # Store feedback
    memory.write("feedback", entry_type="feedback", session_id=session_id, score=score)
    
    # Check that everything worked
    assert isinstance(response, str)
    assert len(response) > 10
    assert isinstance(score, float)
    assert 0.0 <= score <= 5.0
    
    # Check memory storage
    entries = memory.read_all()
    assert len(entries) >= 3  # prompt, output, feedback

def test_reasoning_with_memory(integrated_system):
    """Test latent workspace reasoning with memory context"""
    memory = integrated_system["memory"]
    workspace = integrated_system["workspace"]
    
    # Add some data to memory
    memory.write("The sky is blue", entry_type="fact")
    memory.write("The grass is green", entry_type="fact")
    
    # Get memory context
    memory_context = memory.get_training_data()
    
    # Update workspace with memory
    for item in memory_context:
        embedding = workspace.llm.get_embedding(item['prompt'])
        workspace.update(embedding, context=item['prompt'])
    
    # Reason about the context
    result = workspace.reason("What color is the sky?")
    
    assert "blue" in result['response'].lower()

def test_retraining_loop(integrated_system):
    """Test the full feedback and retraining loop"""
    memory = integrated_system["memory"]
    updater = integrated_system["updater"]
    
    # Add some high-quality data
    for i in range(15):
        memory.write(f"good prompt {i}", entry_type="prompt", session_id=f"sess_{i}")
        memory.write(f"good output {i}", entry_type="output", session_id=f"sess_{i}")
        memory.write("feedback", entry_type="feedback", session_id=f"sess_{i}", score=4.5)
    
    # Get training data
    training_data = memory.get_training_data()
    
    # Fine-tune the model
    result = updater.fine_tune_lora(training_data)
    
    # Should indicate success
    assert result
