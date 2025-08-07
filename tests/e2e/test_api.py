import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class TestAPIEndpoints(unittest.TestCase):
    """Test main API endpoints"""
    
    def test_api_imports(self):
        """Test that main API module can be imported"""
        try:
            from src.main import app
            self.assertTrue(True)  # Import successful
        except Exception as e:
            self.fail(f"Failed to import main API: {e}")

if __name__ == '__main__':
    unittest.main()
