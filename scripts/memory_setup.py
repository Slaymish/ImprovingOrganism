#!/usr/bin/env python3
"""
Memory Management and System Check Script for ImprovingOrganism
Helps configure optimal settings based on available hardware
"""

import os
import sys
import logging
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_system_resources():
    """Check available system resources"""
    print("🔍 System Resource Check")
    print("=" * 50)
    
    # Check CPU
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💻 CPU Cores: {cpu_count}")
        print(f"🧠 RAM: {memory_gb:.2f} GB")
    except ImportError:
        print("❌ psutil not available, cannot check CPU/RAM details")
    
    # Check GPU
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"🎮 CUDA Available: Yes")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name}")
            print(f"   Memory: {memory_gb:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Memory usage check
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"   Currently Allocated: {allocated:.2f} GB")
                print(f"   Currently Reserved: {reserved:.2f} GB")
    else:
        print(f"🎮 CUDA Available: No")
        if TORCH_AVAILABLE:
            print("   Reason: CUDA not available")
        else:
            print("   Reason: PyTorch not installed")
    
    print()

def recommend_settings():
    """Recommend optimal settings based on available resources"""
    print("💡 Recommended Configuration")
    print("=" * 50)
    
    recommendations = {}
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory >= 8.0:
            print("✅ High-end GPU detected")
            recommendations = {
                "FORCE_CPU": "false",
                "MAX_GPU_MEMORY_GB": "4.0",
                "ENABLE_MEMORY_OPTIMIZATION": "true",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
            }
            print("   Recommended: Full GPU mode with optimization")
            
        elif gpu_memory >= 4.0:
            print("⚠️ Mid-range GPU detected")
            recommendations = {
                "FORCE_CPU": "false", 
                "MAX_GPU_MEMORY_GB": "3.0",
                "ENABLE_MEMORY_OPTIMIZATION": "true",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
            }
            print("   Recommended: GPU mode with aggressive memory management")
            
        else:
            print("🔻 Low-memory GPU detected")
            recommendations = {
                "FORCE_CPU": "true",
                "MAX_GPU_MEMORY_GB": "2.0", 
                "ENABLE_MEMORY_OPTIMIZATION": "true"
            }
            print("   Recommended: CPU fallback mode")
    else:
        print("💻 CPU-only mode")
        recommendations = {
            "FORCE_CPU": "true",
            "ENABLE_MEMORY_OPTIMIZATION": "true"
        }
        print("   Recommended: CPU mode with optimization")
    
    print("\nEnvironment Variables to Set:")
    for key, value in recommendations.items():
        print(f"export {key}=\"{value}\"")
    
    return recommendations

def create_memory_config(output_file="memory_config.sh", recommendations=None):
    """Create a shell script with memory configuration"""
    if recommendations is None:
        recommendations = recommend_settings()
    
    config_content = f"""#!/bin/bash
# Memory Configuration for ImprovingOrganism
# Generated automatically based on system capabilities

# Core memory settings
{chr(10).join([f'export {k}="{v}"' for k, v in recommendations.items()])}

# Additional PyTorch memory settings for GPU systems
if [ "$FORCE_CPU" != "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

# Optional: Reduce model precision for memory savings
# export TRANSFORMERS_OFFLINE=1  # Use offline mode to save bandwidth
# export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism warnings

echo "🧠 Memory configuration loaded"
echo "   FORCE_CPU: $FORCE_CPU"
echo "   MAX_GPU_MEMORY_GB: $MAX_GPU_MEMORY_GB" 
echo "   ENABLE_MEMORY_OPTIMIZATION: $ENABLE_MEMORY_OPTIMIZATION"

# Source this file before running the application:
# source {output_file}
# ./run_local.sh api
"""
    
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    os.chmod(output_file, 0o755)  # Make executable
    print(f"\n📝 Configuration saved to: {output_file}")
    print(f"   Usage: source {output_file} && ./run_local.sh api")

def clear_gpu_memory():
    """Clear GPU memory if available"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        
        # Show memory after clearing
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"   GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("❌ No GPU available for memory clearing")

def test_model_loading():
    """Test if the model can be loaded with current settings"""
    print("🧪 Testing Model Loading")
    print("=" * 50)
    
    try:
        # Import after setting environment
        from src.llm_wrapper import LLMWrapper
        
        print("Loading LLM wrapper...")
        llm = LLMWrapper()
        
        print("✅ Model loaded successfully")
        
        # Test generation
        test_prompt = "What is machine learning?"
        print(f"Testing generation with: '{test_prompt}'")
        
        response = llm.generate(test_prompt, max_tokens=50)
        print(f"✅ Generation successful: {response[:100]}...")
        
        # Check memory status
        memory_status = llm.get_memory_status()
        print(f"📊 Memory Status: {memory_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("💡 Try using CPU mode: export FORCE_CPU=true")
        return False

def main():
    parser = argparse.ArgumentParser(description="Memory Management for ImprovingOrganism")
    parser.add_argument("--check", action="store_true", help="Check system resources")
    parser.add_argument("--recommend", action="store_true", help="Recommend optimal settings")
    parser.add_argument("--configure", action="store_true", help="Create configuration file")
    parser.add_argument("--clear-gpu", action="store_true", help="Clear GPU memory")
    parser.add_argument("--test", action="store_true", help="Test model loading")
    parser.add_argument("--output", default="memory_config.sh", help="Output file for configuration")
    
    args = parser.parse_args()
    
    if not any([args.check, args.recommend, args.configure, args.clear_gpu, args.test]):
        # Default: run all checks and create config
        args.check = True
        args.recommend = True
        args.configure = True
    
    if args.check:
        check_system_resources()
    
    recommendations = None
    if args.recommend:
        recommendations = recommend_settings()
    
    if args.configure:
        create_memory_config(args.output, recommendations)
    
    if args.clear_gpu:
        clear_gpu_memory()
    
    if args.test:
        success = test_model_loading()
        return 0 if success else 1
    
    return 0

if __name__ == "__main__":
    exit(main())
