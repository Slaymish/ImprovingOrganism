#!/bin/bash
# Test runner script for ImprovingOrganism

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Running ImprovingOrganism Test Suite"
echo "========================================"
TOP_LEVEL_DIR="."

# Run unit tests
echo "🔬 Running Unit Tests..."
python -m unittest discover -s tests/unit -p "test_*.py" -t "$TOP_LEVEL_DIR"

# Run integration tests
echo "🔗 Running Integration Tests..."
python -m unittest discover -s tests/integration -p "test_*.py" -t "$TOP_LEVEL_DIR"

# Run end-to-end tests
echo "🌐 Running End-to-End Tests..."
python -m unittest discover -s tests/e2e -p "test_*.py" -t "$TOP_LEVEL_DIR"

echo "========================================"
echo "✅ All tests passed successfully!"
