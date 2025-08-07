import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.latent_workspace import LatentWorkspace

@pytest.fixture
def workspace():
    """Fixture for LatentWorkspace"""
    return LatentWorkspace(dim=128)  # Smaller for faster tests

def test_workspace_initialization(workspace):
    """Test workspace initialization"""
    assert workspace.dim == 128
    assert workspace.latent_dim == 128
    assert workspace.workspace is not None

def test_update_operations(workspace):
    """Test workspace update with embeddings"""
    # Test with numpy array
    embedding = np.random.randn(128)
    workspace.update(embedding, context="test update")
    
    # Workspace should be updated
    workspace_norm = np.linalg.norm(workspace.workspace)
    assert float(workspace_norm) > 0.0

def test_reasoning_process(workspace):
    """Test latent space reasoning"""
    # Update workspace with some context
    for i in range(3):
        embedding = np.random.randn(128)
        workspace.update(embedding, context=f"context {i}")
    
    # Test reasoning with string query
    result = workspace.reason("What can you tell me about this?", reasoning_steps=2)
    
    assert isinstance(result, dict)
    assert 'response' in result
    assert 'confidence' in result
    assert 'reasoning_steps' in result
    assert result['reasoning_steps'] == 2

def test_goal_setting(workspace):
    """Test goal-directed reasoning"""
    goal_embedding = np.random.randn(128)
    workspace.set_goal(goal_embedding)
    
    # Goal should be set
    assert workspace.goal_state is not None

def test_goal_directed_reasoning(workspace):
    """Test reasoning towards a set goal"""
    # Set a goal
    goal_embedding = np.random.randn(128)
    workspace.set_goal(goal_embedding)
    
    # Update with some context
    for i in range(3):
        embedding = np.random.randn(128)
        workspace.update(embedding, context=f"context {i}")
    
    # Reason towards the goal
    result = workspace.reason("How do we achieve the goal?", reasoning_steps=3, use_goal=True)
    
    assert 'response' in result
    assert result['confidence'] > 0
