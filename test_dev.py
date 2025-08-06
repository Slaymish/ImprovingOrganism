#!/usr/bin/env python3
"""
Quick test script to verify the system works in development mode
without heavy ML dependencies.
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from src.config import settings
        print("✅ config.py imported successfully")
        
        from src.memory_module import MemoryModule
        print("✅ memory_module.py imported successfully")
        
        from src.critic_module import CriticModule
        print("✅ critic_module.py imported successfully")
        
        from src.llm_wrapper import LLMWrapper
        print("✅ llm_wrapper.py imported successfully")
        
        from src.updater import Updater
        print("✅ updater.py imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic system functionality without ML dependencies"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from src.memory_module import MemoryModule
        from src.critic_module import CriticModule
        from src.llm_wrapper import LLMWrapper
        
        # Test memory module
        memory = MemoryModule()
        memory.write("Test entry", "test")
        entries = memory.read_all()
        print(f"✅ Memory module working - {len(entries)} entries")
        
        # Test critic module
        critic = CriticModule()
        score = critic.score("Test prompt", "Test output", [])
        print(f"✅ Critic module working - Score: {score:.2f}")
        
        # Test LLM wrapper (should use mock mode)
        llm = LLMWrapper()
        output = llm.generate("Test prompt")
        print(f"✅ LLM wrapper working - Output: {output[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_api_import():
    """Test that the API can be imported"""
    print("\n🧪 Testing API module...")
    
    try:
        from src.main import app
        print("✅ FastAPI app imported successfully")
        print(f"✅ App title: {app.title}")
        return True
        
    except Exception as e:
        print(f"❌ API import failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("ImprovingOrganism Development Mode Test")
    print("=" * 60)
    
    print("This test verifies the system works without heavy ML dependencies.")
    print("ML components should fall back to mock implementations.\n")
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test functionality
    success &= test_basic_functionality()
    
    # Test API
    success &= test_api_import()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! System is ready for development.")
        print("\nNext steps:")
        print("1. Install full ML dependencies: pip install -r requirements.txt")
        print("2. Start the API: uvicorn src.main:app --reload")
        print("3. Run the demo: python demo.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
