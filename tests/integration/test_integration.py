import sys
import os
import unittest
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

class TestIntegration(unittest.TestCase):
    """Test integration between modules"""
    
    def setUp(self):
        """Set up integrated system"""
        self.memory = MemoryModule()
        self.critic = CriticModule()
        self.updater = Updater()
        self.llm = LLMWrapper()
        self.workspace = LatentWorkspace(dim=256)
    
    def test_end_to_end_conversation(self):
        """Test complete conversation flow"""
        # Simulate a conversation
        prompt = "What is machine learning?"
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Store in memory
        session_id = "test_session"
        self.memory.write(prompt, entry_type="prompt", session_id=session_id)
        self.memory.write(response, entry_type="output", session_id=session_id)
        
        # Score the response
        context = [prompt]
        score = self.critic.score(prompt, response, context)
        
        # Store feedback
        self.memory.write("feedback", entry_type="feedback", session_id=session_id, score=score)
        
        # Check that everything worked
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 5.0)
        
        # Check memory storage
        entries = self.memory.read_all()
        self.assertGreaterEqual(len(entries), 3)  # prompt, output, feedback
    
    def test_reasoning_with_memory(self):
        """Test latent workspace reasoning with memory context"""
        # Add conversation history
        conversations = [
            ("What is AI?", "AI is artificial intelligence..."),
            ("How does ML work?", "Machine learning uses algorithms..."),
            ("What about neural networks?", "Neural networks are inspired by...")
        ]
        
        for i, (prompt, response) in enumerate(conversations):
            session_id = f"conv_{i}"
            self.memory.write(prompt, entry_type="prompt", session_id=session_id)
            self.memory.write(response, entry_type="output", session_id=session_id)
            
            # Update workspace with mock embedding
            embedding = np.random.randn(256)
            self.workspace.update(embedding, context=f"Conversation {i}")
        
        # Now reason about a related question
        question = "Tell me about AI and machine learning"
        
        reasoning_result = self.workspace.reason(question, reasoning_steps=3)
        
        # Should get a coherent response
        self.assertIsInstance(reasoning_result, dict)
        self.assertIn('response', reasoning_result)
        self.assertIn('confidence', reasoning_result)
        
        # Update workspace with result
        if 'embedding' in reasoning_result:
            self.workspace.update(reasoning_result['embedding'], 
                                context="Reasoning result", importance=0.8)

if __name__ == '__main__':
    unittest.main()
