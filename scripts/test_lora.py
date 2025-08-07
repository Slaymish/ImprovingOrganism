#!/usr/bin/env python3
"""
Test script for LoRA training functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lora_trainer import LoRATrainer
from src.memory_module import MemoryModule
import json

def test_lora_training():
    """Test LoRA training with sample data"""
    print("🧪 Testing LoRA Training System")
    print("=" * 50)
    
    # Initialize components
    trainer = LoRATrainer()
    memory = MemoryModule()
    
    # Check training status
    print("\n📊 Checking training status...")
    status = trainer.get_training_status()
    print(json.dumps(status, indent=2))
    
    if not status.get('ml_available', False):
        print("❌ ML dependencies not available. Please install:")
        print("   pip install torch transformers peft datasets accelerate")
        return
    
    # Check if we have enough data
    if not status.get('ready_for_training', False):
        print(f"\n⚠️  Insufficient training data:")
        print(f"   - Total feedback: {status.get('total_feedback_entries', 0)}")
        print(f"   - Good feedback: {status.get('good_feedback_entries', 0)}")
        print(f"   - Need at least 10 good feedback entries")
        
        # Add some sample training data for testing
        print("\n🔧 Adding sample training data...")
        add_sample_data(memory)
        
        # Re-check status
        status = trainer.get_training_status()
        print(f"   - Updated good feedback: {status.get('good_feedback_entries', 0)}")
    
    if status.get('ready_for_training', False):
        print("\n🚀 Starting LoRA training...")
        
        # Start training with reduced parameters for testing
        result = trainer.train_lora(
            min_feedback_score=3.0,
            max_samples=50  # Small sample for testing
        )
        
        print("\n📋 Training Results:")
        print(json.dumps(result, indent=2))
        
        if result.get('status') == 'completed':
            print(f"\n✅ Training completed successfully!")
            print(f"   - Adapter path: {result.get('adapter_path')}")
            print(f"   - Training examples: {result.get('training_examples')}")
            print(f"   - Training time: {result.get('training_time_minutes', 0):.2f} minutes")
            
            # Test evaluation
            print("\n🔍 Testing adapter evaluation...")
            eval_result = trainer.evaluate_adapter(
                result.get('adapter_path'),
                ["What is machine learning?", "Explain AI briefly."]
            )
            
            if eval_result.get('status') == 'completed':
                print("✅ Evaluation completed!")
                for i, test in enumerate(eval_result.get('test_results', [])):
                    print(f"\nTest {i+1}:")
                    print(f"  Prompt: {test['prompt']}")
                    print(f"  Response: {test['response'][:100]}...")
            else:
                print(f"❌ Evaluation failed: {eval_result.get('error')}")
        else:
            print(f"\n❌ Training failed: {result.get('error')}")
    else:
        print("\n⚠️  Still not ready for training. More feedback data needed.")

def add_sample_data(memory):
    """Add sample training data for testing"""
    import uuid
    from datetime import datetime
    
    sample_conversations = [
        {
            "prompt": "What is artificial intelligence?",
            "response": "Artificial intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, and perception.",
            "score": 4.5
        },
        {
            "prompt": "Explain machine learning in simple terms.",
            "response": "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed for each task. It's like teaching a computer to recognize patterns and make predictions.",
            "score": 4.0
        },
        {
            "prompt": "What are neural networks?",
            "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information and can learn complex patterns in data.",
            "score": 3.8
        },
        {
            "prompt": "How does deep learning work?",
            "response": "Deep learning uses neural networks with multiple layers to automatically discover patterns in data. Each layer learns increasingly complex features, enabling tasks like image recognition and natural language processing.",
            "score": 4.2
        },
        {
            "prompt": "What is the difference between AI and ML?",
            "response": "AI is the broader concept of machines being able to carry out tasks in a smart way, while ML is a subset of AI that focuses on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.",
            "score": 4.1
        }
    ]
    
    for conv in sample_conversations:
        session_id = str(uuid.uuid4())
        
        # Add prompt
        memory.write(
            content=conv["prompt"],
            entry_type="prompt",
            session_id=session_id
        )
        
        # Add response
        memory.write(
            content=conv["response"],
            entry_type="output",
            session_id=session_id
        )
        
        # Add feedback
        memory.write(
            content=f"FEEDBACK: {conv['prompt']} -> {conv['response']} | score={conv['score']}",
            entry_type="feedback",
            session_id=session_id,
            score=conv["score"]
        )
    
    print(f"✅ Added {len(sample_conversations)} sample conversations")

if __name__ == "__main__":
    test_lora_training()
