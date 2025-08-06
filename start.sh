#!/bin/bash

echo "🚀 Starting ImprovingOrganism API..."

# Check if we're in development mode (lighter dependencies)
if [ "$DEV_MODE" = "true" ]; then
    echo "📝 Running in development mode (mock ML components)"
    export MODEL_NAME="mock-model"
    export LORA_PATH="./mock-adapters"
fi

# Create necessary directories
mkdir -p logs
mkdir -p adapters

# Start the API server
echo "🌟 Starting FastAPI server on port 8000..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
