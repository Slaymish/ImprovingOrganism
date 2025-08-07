#!/bin/bash
# Test runner script for ImprovingOrganism

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Running ImprovingOrganism Test Suite"
echo "========================================"
TOP_LEVEL_DIR="."

# Run tests with coverage
echo "🔍 Running Tests with Coverage..."
#!/bin/bash
# Run all tests using pytest, ensuring dependencies are installed

echo "🚀 Running ImprovingOrganism Test Suite"
echo "========================================"

# Add the project root to the python path
export PYTHONPATH=$(pwd)

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Run pytest with coverage
echo "🔍 Running Tests with Coverage..."
python3 -m pytest --cov=src tests/


echo "========================================"
echo "✅ All tests passed successfully!"
