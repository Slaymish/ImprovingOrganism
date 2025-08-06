# ImprovingOrganism Project

A self-improving AI system that learns from feedback to enhance its responses over time. This project implements a complete feedback loop: generate → evaluate → collect feedback → retrain → improve.

## 🚀 Features

- **FastAPI-based REST API** for text generation and feedback collection
- **Comprehensive Scoring System** with metrics for:
  - Coherence (grammatical structure, readability)
  - Novelty (uniqueness compared to past outputs)
  - Memory Alignment (consistency with stored knowledge)
  - Relevance (how well output addresses the prompt)
- **Memory System** with tagged entries for easy retrieval
- **Automatic LoRA Fine-tuning** based on feedback data
- **Session Tracking** for grouped interactions
- **End-to-end Demonstration** script

## 📁 Project Structure

```
ImprovingOrganism/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── demo.py                    # End-to-end demonstration
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   ├── llm_wrapper.py        # LLM wrapper with LoRA support
│   ├── memory_module.py      # SQLAlchemy-based memory storage
│   ├── critic_module.py      # Multi-metric scoring system
│   ├── updater.py            # LoRA fine-tuning logic
│   ├── latent_workspace.py   # Latent space management
│   └── config.py             # Configuration settings
└── logs/                     # Log files directory
```

## 🛠 Setup & Installation

### Quick Start (Development Mode)

For testing without heavy ML dependencies:

1. **Install lightweight dependencies:**
```bash
pip install -r requirements-dev.txt
```

2. **Test the system:**
```bash
python test_dev.py
```

3. **Start in development mode:**
```bash
./start.sh
# or
DEV_MODE=true uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Full Installation (Production Mode)

For complete ML functionality:

1. **Install all dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment variables (optional):**
```bash
export MODEL_NAME="microsoft/DialoGPT-medium"  # or your preferred model
export LORA_PATH="./adapters/custom_lora"
export DATABASE_URL="sqlite:///./memory.db"
```

3. **Start the API server:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Docker Setup (Fixed)

**Note:** The Docker setup has been fixed for the import and dependency issues.

1. **Build and start services:**
```bash
docker compose up --build
```

2. **Access the API at:** `http://localhost:8000`

#### Docker Services:
- **`api`**: Main FastAPI application 
- **`training`**: Dedicated retraining service

## 🔧 Troubleshooting

### Import Errors in Docker

The system now handles missing ML dependencies gracefully:
- ✅ **Fixed**: Module import paths for Docker containers
- ✅ **Fixed**: Pydantic version compatibility (`BaseSettings` moved to `pydantic-settings`)
- ✅ **Added**: Fallback to mock implementations when ML libraries unavailable

### Development vs Production Mode

- **Development Mode**: Uses mock LLM responses, lightweight dependencies
- **Production Mode**: Full ML functionality with real model inference and training

## 📚 API Endpoints

### 🔄 Core Operations

**`POST /generate`** - Generate text from a prompt
```json
{
  "text": "Explain machine learning",
  "session_id": "optional-session-id"
}
```

**`POST /feedback`** - Submit feedback for improvement
```json
{
  "prompt": "Explain machine learning",
  "output": "Generated response...",
  "score": 4.5,
  "comment": "Very clear explanation",
  "session_id": "optional-session-id"
}
```

### 📊 Analytics

**`GET /stats`** - System statistics
**`GET /detailed_score/{session_id}`** - Detailed scoring breakdown
**`POST /trigger_training`** - Check training readiness

## 🎯 End-to-End Demonstration

Run the complete demonstration to see the system in action:

```bash
# Start the API server first
uvicorn src.main:app --host 0.0.0.0 --port 8000

# In another terminal, run the demo
python demo.py
```

The demonstration will:
1. ✅ Generate multiple responses
2. ✅ Submit varied feedback scores
3. ✅ Show detailed scoring metrics
4. ✅ Display system statistics
5. ✅ Check training readiness
6. ✅ Demonstrate the complete feedback loop

## 🧠 How It Works

### 1. Generation Phase
- User submits a prompt via `/generate`
- System generates response using current model
- Automatic scoring using multi-metric evaluation
- All interactions stored with session tracking

### 2. Feedback Collection
- Human feedback submitted via `/feedback`
- Scores stored with detailed context
- System tracks feedback patterns and quality

### 3. Evaluation & Analysis
The **CriticModule** evaluates responses across four dimensions:

- **Coherence (30%)**: Grammar, structure, readability
- **Novelty (25%)**: Uniqueness vs. past responses  
- **Memory Alignment (25%)**: Consistency with stored knowledge
- **Relevance (20%)**: How well it addresses the prompt

### 4. Retraining Process
The **Updater** automatically:
- Monitors feedback volume and quality
- Prepares training datasets from good feedback
- Fine-tunes LoRA adapters when thresholds are met
- Updates the model for improved future responses

## 🔧 Configuration

Key settings in `src/config.py`:

```python
class Settings:
    model_name: str = "gpt-oss-20b"           # Base model
    lora_path: str = "./adapters/hamish_lora"  # LoRA adapter path
    db_url: str = "sqlite:///./memory.db"      # Database URL
    feedback_threshold: int = 50               # Min feedback for retraining
    batch_size: int = 8                       # Training batch size
```

## 📈 Memory System

The enhanced memory module supports:
- **Tagged entries**: `prompt`, `output`, `feedback`, `internal_feedback`
- **Session grouping**: Related interactions linked together
- **Training data extraction**: Automatic dataset preparation
- **Flexible querying**: By type, session, score, timestamp

## 🚀 Advanced Features

### Multi-Metric Scoring
Each response is evaluated on multiple dimensions with detailed breakdowns available via `/detailed_score/{session_id}`.

### Automatic Quality Assessment
The system provides immediate feedback on generated content, helping identify areas for improvement before human review.

### Incremental Learning
The LoRA fine-tuning approach allows the system to adapt without full model retraining, making continuous improvement feasible.

## 🐳 Docker Services

- **`api`**: Main FastAPI application
- **`training`**: Dedicated retraining service (runs `updater.py`)

## 🔍 Monitoring & Debugging

- Check logs in the `/logs` directory
- Use `/stats` endpoint for system health
- Monitor feedback patterns through the API
- Detailed scoring helps identify specific improvement areas

## 🚧 Next Steps

1. **Implement Vector Embeddings** for better semantic analysis
2. **Add Real-time Training Triggers** based on feedback patterns
3. **Integrate with External Knowledge Bases** for memory alignment
4. **Add A/B Testing** for model comparison
5. **Implement Model Versioning** for rollback capabilities

## 📝 Example Usage

```python
import requests

# Generate content
response = requests.post("http://localhost:8000/generate", 
    json={"text": "What is artificial intelligence?"})
result = response.json()

# Submit feedback
requests.post("http://localhost:8000/feedback", json={
    "prompt": "What is artificial intelligence?",
    "output": result["output"],
    "score": 4.5,
    "comment": "Clear and comprehensive",
    "session_id": result["session_id"]
})
```

## 🤝 Contributing

The system is designed for extensibility:
- Add new scoring metrics in `CriticModule`
- Extend memory storage with additional metadata
- Implement new training strategies in `Updater`
- Add new API endpoints for specific use cases

---

**Happy learning and improving! 🎉**
