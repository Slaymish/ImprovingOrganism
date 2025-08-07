FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        libssl-dev \
        libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch before other requirements
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY adapters/ ./adapters/
COPY logs/ ./logs/

# Create empty __init__.py if it doesn't exist
RUN touch /app/__init__.py

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
