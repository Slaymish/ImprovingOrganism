#!/usr/bin/env python3
"""
Debug script to test ML library imports
"""

print("Testing ML library imports...")

try:
    import torch
    print(f"✅ torch imported successfully, version: {torch.__version__}")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ transformers imported successfully")
except ImportError as e:
    print(f"❌ transformers import failed: {e}")

try:
    from peft import PeftModel
    print("✅ peft imported successfully")
except ImportError as e:
    print(f"❌ peft import failed: {e}")

print("Import test complete.")
