import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_api_imports():
    """Test that main API module can be imported"""
    try:
        from src.main import app
        assert True  # Import successful
    except Exception as e:
        pytest.fail(f"Failed to import main API: {e}")
