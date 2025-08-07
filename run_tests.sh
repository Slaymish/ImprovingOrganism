#!/bin/bash
# Test runner script for ImprovingOrganism

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Running ImprovingOrganism Test Suite"
echo "========================================"
TOP_LEVEL_DIR="."

# Run tests with coverage
echo "🔍 Running Tests with Coverage..."
pytest --cov=src tests/

echo "========================================"
echo "✅ All tests passed successfully!"
