#!/usr/bin/env python3
"""
Comprehensive Test Suite for ImprovingOrganism
Tests all modules: memory, critic, updater, latent_workspace, llm_wrapper, and main API
"""

import sys
import os
import unittest
import tempfile
import json
import time
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestMemoryModule(unittest.TestCase):
    """Test MemoryModule functionality"""
    
    def setUp(self):
        """Set up test memory module"""
        from src.memory_module import MemoryModule
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

class TestCriticModule(unittest.TestCase):
    """Test CriticModule scoring functionality"""
    
    def setUp(self):
        from src.critic_module import CriticModule
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

class TestUpdater(unittest.TestCase):
    """Test Updater module for LoRA fine-tuning"""
    
    def setUp(self):
        from src.updater import Updater
        self.updater = Updater()
    
    def test_retraining_decision(self):
        """Test when system decides to retrain"""
        # Should have basic functionality
        self.assertIsNotNone(self.updater)
    
    def test_mock_fine_tuning(self):
        """Test mock fine-tuning when ML dependencies unavailable"""
        # Create training pairs with scores (as expected by API)
        training_pairs = [("input1", "output1", 4.0), ("input2", "output2", 3.5)]
        result = self.updater.fine_tune_lora(training_pairs)
        self.assertIsInstance(result, (bool, dict))  # May return bool or dict depending on implementation

class TestLLMWrapper(unittest.TestCase):
    """Test LLMWrapper functionality"""
    
    def setUp(self):
        from src.llm_wrapper import LLMWrapper
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

class TestLatentWorkspace(unittest.TestCase):
    """Test LatentWorkspace reasoning functionality"""
    
    def setUp(self):
        from src.latent_workspace import LatentWorkspace
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

class TestIntegration(unittest.TestCase):
    """Test integration between modules"""
    
    def setUp(self):
        """Set up integrated system"""
        from src.memory_module import MemoryModule
        from src.critic_module import CriticModule
        from src.updater import Updater
        from src.llm_wrapper import LLMWrapper
        from src.latent_workspace import LatentWorkspace
        
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

class TestAPIEndpoints(unittest.TestCase):
    """Test main API endpoints"""
    
    def test_api_imports(self):
        """Test that main API module can be imported"""
        try:
            from src.main import app
            self.assertTrue(True)  # Import successful
        except Exception as e:
            self.fail(f"Failed to import main API: {e}")

def run_test_suite():
    """Run the complete test suite with detailed reporting"""
    print("🚀 ImprovingOrganism Comprehensive Test Suite")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMemoryModule,
        TestCriticModule, 
        TestUpdater,
        TestLLMWrapper,
        TestLatentWorkspace,
        TestIntegration,
        TestAPIEndpoints
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Suite Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"   Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 Errors ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! System is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Please review the issues above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
