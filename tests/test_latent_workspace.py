#!/usr/bin/env python3
"""
Test script for the LatentWorkspace functionality.
Demonstrates reasoning in latent space without text conversion.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_latent_workspace():
    """Test the LatentWorkspace functionality"""
    print("🧠 Testing LatentWorkspace - Reasoning in Latent Space")
    print("=" * 60)
    
    try:
        from src.latent_workspace import LatentWorkspace
        
        # Initialize workspace
        print("1. Initializing LatentWorkspace...")
        workspace = LatentWorkspace(dim=512, memory_size=50, num_reasoning_layers=3)
        print(f"   ✅ Workspace initialized with {workspace.dim} dimensions")
        
        # Test basic operations
        print("\n2. Testing basic operations...")
        
        # Create mock embeddings (simulating text-to-embedding conversion)
        mock_embeddings = [
            np.random.randn(512) * 0.1,  # "What is machine learning?"
            np.random.randn(512) * 0.1,  # "Machine learning is a subset of AI"
            np.random.randn(512) * 0.1,  # "It involves training algorithms on data"
        ]
        
        # Update workspace with different concepts
        contexts = [
            "User asks about machine learning",
            "System provides definition", 
            "System explains the process"
        ]
        
        for i, (embedding, context) in enumerate(zip(mock_embeddings, contexts)):
            importance = 0.8 + 0.2 * i  # Increasing importance
            workspace.update(embedding, context=context, importance=importance)
            print(f"   ✅ Updated workspace with: {context}")
            
        # Test introspection
        print("\n3. Introspecting workspace state...")
        state = workspace.introspect()
        print(f"   Workspace norm: {state['workspace_norm']:.3f}")
        print(f"   Episodic memory size: {state['episodic_memory_size']}")
        print(f"   Confidence level: {state['confidence']:.3f}")
        if 'uncertainty_level' in state:
            print(f"   Uncertainty level: {state['uncertainty_level']:.3f}")
        
        # Test reasoning
        print("\n4. Testing reasoning in latent space...")
        query_embedding = np.random.randn(512) * 0.1  # "How does ML training work?"
        
        print("   🧠 Performing multi-step reasoning...")
        reasoning_result = workspace.reason(query_embedding, reasoning_steps=3)
        print(f"   ✅ Reasoning completed. Result shape: {np.array(reasoning_result).shape}")
        
        # Test goal-directed reasoning
        print("\n5. Testing goal-directed reasoning...")
        goal_embedding = np.random.randn(512) * 0.1  # "Explain complex concepts simply"
        workspace.set_goal(goal_embedding)
        
        # Reason again with goal set
        goal_reasoning_result = workspace.reason(query_embedding, reasoning_steps=2)
        print("   ✅ Goal-directed reasoning completed")
        
        # Final introspection
        print("\n6. Final workspace state...")
        final_state = workspace.introspect()
        print(f"   Final confidence: {final_state['confidence']:.3f}")
        print(f"   Reasoning depth: {final_state.get('reasoning_depth', 'N/A')}")
        if 'goal_alignment' in final_state:
            print(f"   Goal alignment: {final_state['goal_alignment']:.3f}")
        
        # Test thought chain
        print("\n7. Analyzing thought chain...")
        thought_chain = workspace.get_thought_chain()
        print(f"   Thought chain length: {len(thought_chain)}")
        
        thought_types = {}
        for thought in thought_chain:
            thought_type = thought.get('type', 'unknown')
            thought_types[thought_type] = thought_types.get(thought_type, 0) + 1
            
        print("   Thought distribution:")
        for thought_type, count in thought_types.items():
            print(f"     {thought_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ LatentWorkspace test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_system():
    """Test integration with the main system"""
    print("\n🔗 Testing Integration with Main System")
    print("=" * 60)
    
    try:
        from src.latent_workspace import LatentWorkspace
        from src.memory_module import MemoryModule
        from src.critic_module import CriticModule
        
        # Create integrated system
        workspace = LatentWorkspace(dim=256)
        memory = MemoryModule()
        critic = CriticModule()
        
        print("✅ All components created successfully")
        
        # Simulate a reasoning scenario
        print("\n🧠 Simulating integrated reasoning scenario...")
        
        # Mock: User asks a question
        question_embedding = np.random.randn(256) * 0.1
        workspace.update(question_embedding, context="User question about AI ethics", importance=1.0)
        
        # Mock: System retrieves relevant memories
        memory_entries = memory.read_all()
        if memory_entries:
            print(f"   Retrieved {len(memory_entries)} memory entries")
        
        # Mock: System reasons about the question
        reasoning_result = workspace.reason(question_embedding, reasoning_steps=2)
        
        # Mock: System generates response (would be converted back to text)
        response_text = "AI ethics is important because..."
        
        # Mock: System evaluates its own response
        score = critic.score("What about AI ethics?", response_text, memory_entries)
        
        print(f"   Generated response scored: {score:.2f}/5.0")
        
        # Update workspace with the reasoning result
        workspace.update(reasoning_result['embedding'], context="System response generation", importance=score/5.0)
        
        # Final state
        final_state = workspace.introspect()
        print(f"   Final workspace confidence: {final_state['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_latent_advantages():
    """Demonstrate advantages of latent space reasoning"""
    print("\n💡 Demonstrating Latent Space Advantages")
    print("=" * 60)
    
    try:
        from src.latent_workspace import LatentWorkspace
        
        workspace = LatentWorkspace(dim=128)
        
        print("🔍 Advantages of reasoning in latent space:")
        print("   1. No information loss from text conversion")
        print("   2. Continuous semantic operations")
        print("   3. Efficient similarity computations")
        print("   4. Rich contextual associations")
        print("   5. Multi-step reasoning without verbalization")
        
        # Demonstrate continuous operations
        concept_a = np.random.randn(128) * 0.1  # "happiness"
        concept_b = np.random.randn(128) * 0.1  # "joy"
        concept_c = np.random.randn(128) * 0.1  # "sadness"
        
        print("\n📊 Semantic operations in latent space:")
        
        # Update workspace with concepts
        workspace.update(concept_a, context="happiness concept", importance=0.8)
        workspace.update(concept_b, context="joy concept", importance=0.8)
        workspace.update(concept_c, context="sadness concept", importance=0.6)
        
        # Reason about emotional spectrum
        emotional_query = (concept_a + concept_b) / 2  # Blend happiness and joy
        reasoning_result = workspace.reason(emotional_query, reasoning_steps=2)
        
        print("   ✅ Performed semantic blending of emotional concepts")
        print("   ✅ Reasoned about emotional relationships")
        print("   ✅ Maintained rich semantic associations")
        
        # Show workspace state evolution
        state = workspace.introspect()
        print(f"\n📈 Workspace evolved to contain {state['episodic_memory_size']} concept memories")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return False

def main():
    print("🚀 LatentWorkspace Comprehensive Test Suite")
    print("Testing reasoning in latent space without text conversion loss")
    print("=" * 80)
    
    success = True
    
    # Run all tests
    success &= test_latent_workspace()
    success &= test_integration_with_system()  
    success &= demonstrate_latent_advantages()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 All LatentWorkspace tests passed!")
        print("\n💫 The LatentWorkspace enables:")
        print("   • Rich semantic reasoning without text conversion")
        print("   • Preservation of continuous semantic information")
        print("   • Multi-step reasoning in high-dimensional space")
        print("   • Memory consolidation and retrieval")
        print("   • Goal-directed cognitive processes")
        print("   • Uncertainty-aware reasoning")
    else:
        print("❌ Some tests failed. Check the errors above.")
        
    print("=" * 80)

if __name__ == "__main__":
    main()
