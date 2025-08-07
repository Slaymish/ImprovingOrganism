#!/usr/bin/env bash
# run_local.sh: Run the API or training service locally in a Python venv
# Usage: ./run_local.sh [api|training|all|help]

set -euo pipefail

SERVICE=${1:-help}

# Ensure we have a venv
if [[ ! -f "venv/bin/activate" ]]; then
  echo "🔧 Python venv not found. Creating one..."
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install CPU-only PyTorch if not installed
if ! python -c "import torch" &> /dev/null; then
  echo "🧠 Installing CPU PyTorch..."
  pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Python dependencies
echo "📥 Installing Python requirements..."
pip install --no-cache-dir -r requirements.txt

# Set environment variables
export PYTHONPATH="$(pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Run the selected service
case "$SERVICE" in
  api)
    echo "🚀 Starting API (uvicorn src.main:app)..."
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    ;;
  training)
    echo "🏋️‍♂️ Starting training (python -m src.updater)..."
    python -m src.updater
    ;;
  all)
    echo "⚡ Running both API and training in parallel..."
    # Using & to background processes; logs printed to stdout
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 &
    python -m src.updater
    ;;
  help)
    echo "Usage: $0 [api|training|all]"
    echo "  api      Run the API service"
    echo "  training Run the training script"
    echo "  all      Run both (API in background)"
    exit 1
    ;;
  *)
    echo "❓ Unknown service: $SERVICE"
    echo "Run '$0 help' for usage."
    exit 1
    ;;
 esac
