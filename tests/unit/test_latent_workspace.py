import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.latent_workspace import LatentWorkspace

class TestLatentWorkspace(unittest.TestCase):
    """Test LatentWorkspace reasoning functionality"""
    
    def setUp(self):
        self.workspace = LatentWorkspace(dim=128)  # Smaller for faster tests
    
    def test_workspace_initialization(self):
        """Test workspace initialization"""
        self.assertEqual(self.workspace.dim, 128)
        self.assertEqual(self.workspace.latent_dim, 128)
        self.assertIsNotNone(self.workspace.workspace)
    
    def test_update_operations(self):
        """Test workspace update with embeddings"""
        # Test with numpy array
        embedding = np.random.randn(128)
        self.workspace.update(embedding, context="test update")
        
        # Workspace should be updated
        workspace_norm = np.linalg.norm(self.workspace.workspace)
        self.assertGreater(float(workspace_norm), 0.0)
    
    def test_reasoning_process(self):
        """Test latent space reasoning"""
        # Update workspace with some context
        for i in range(3):
            embedding = np.random.randn(128)
            self.workspace.update(embedding, context=f"context {i}")
        
        # Test reasoning with string query
        result = self.workspace.reason("What can you tell me about this?", reasoning_steps=2)
        
        self.assertIsInstance(result, dict)
        self.assertIn('response', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning_steps', result)
        self.assertEqual(result['reasoning_steps'], 2)
    
    def test_goal_setting(self):
        """Test goal-directed reasoning"""
        goal_embedding = np.random.randn(128)
        self.workspace.set_goal(goal_embedding)
        
        # Goal should be set
        self.assertIsNotNone(self.workspace.goal_state)
    
    def test_introspection(self):
        """Test workspace introspection"""
        # Add some data
        for i in range(3):
            embedding = np.random.randn(128)
            self.workspace.update(embedding, context=f"entry {i}")
        
        state = self.workspace.introspect()
        
        self.assertIsInstance(state, dict)
        self.assertIn('workspace_norm', state)
        self.assertIn('episodic_memory_size', state)
        self.assertIn('confidence', state)

if __name__ == '__main__':
    unittest.main()
