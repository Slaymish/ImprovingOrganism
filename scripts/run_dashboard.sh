#!/bin/bash
# Start the dashboard
echo "🧠 Starting ImprovingOrganism Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🚀 Make sure your Docker containers are running first:"
echo "   docker-compose up"
echo ""

# Install dashboard dependencies if not already installed
pip install streamlit plotly pandas requests

# Run the dashboard
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
