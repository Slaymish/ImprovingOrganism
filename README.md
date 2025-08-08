# ImprovingOrganism Project

A self-improving AI system that learns from feedback to enhance its responses over time. This project implements a complete feedback loop: generate → evaluate → collect feedback → retrain → improve.

## 🚀 Features

  - Coherence (grammatical structure, readability)
  - Novelty (uniqueness compared to past outputs)
  - Memory Alignment (consistency with stored knowledge)
  - Relevance (how well output addresses the prompt)

## Preference Learning Scaffold (Phase 2 - Partial)
An initial preference learning layer has been introduced:
- Multiple candidate variants per prompt (configurable `preference_variants`).
- Critic-scored ranking; high-gap pairs become `PreferencePair` objects.
- Pairs accumulated via a global `preference_optimizer` for future DPO / logistic preference fine-tuning.
- Lightweight mode (`LIGHTWEIGHT_SELF_LEARNING=1`) avoids heavy model + vector memory for fast tests.
- Metrics extended: preference batches, variants, pairs, latency (exposed via `/stats`).

Next steps (not yet implemented): adaptive temperature/top-p diversity control, duplicate suppression, pair de-duplication, logistic preference loss training routine, and integration into updater fine-tune trigger.
## 🧠 Self-Learning System

The system can improve autonomously through several mechanisms:

### 1. **Autonomous Prompt Generation**
- Generates diverse educational prompts across 15+ knowledge domains
- Uses varied question types (factual, analytical, creative, problem-solving)
- Creates contextual prompts that challenge different capabilities

### 2. **Empirical Response Evaluation**
- **Mathematical Accuracy**: Verifies calculations and mathematical expressions
- **Logical Structure**: Checks for coherent flow and transitions
- **Completeness**: Evaluates response depth and coverage
- **Knowledge Consistency**: Ensures responses align with prompts and domain knowledge

### 3. **Continuous Learning Loop**
```
Generate Prompt → Create Response → Evaluate Quality → Store Feedback → Learn
```

### 4. **Adaptive Scheduling**
- Increases learning frequency when performance improves
- Reduces frequency when performance plateaus or declines
- Monitors trends and adjusts accordingly

## 📁 Project Structure

```
ImprovingOrganism/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── dashboard.py               # Streamlit dashboard with self-learning UI
├── demo.py                    # End-to-end demonstration
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI application with self-learning endpoints
│   ├── llm_wrapper.py        # LLM wrapper with LoRA support
│   ├── memory_module.py      # SQLAlchemy-based memory storage
│   ├── critic_module.py      # Multi-metric scoring system
│   ├── self_learning.py      # Autonomous learning and evaluation system
│   ├── updater.py            # LoRA fine-tuning logic
│   ├── latent_workspace.py   # Latent space management
│   └── config.py             # Configuration settings
├── scripts/
│   └── continuous_learning.py # Background continuous learning service
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

### Docker Setup 

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

### 🧠 Self-Learning

**`POST /self_learning/start_session`** - Start autonomous learning session
```bash
curl -X POST "http://localhost:8000/self_learning/start_session?iterations=5"
```

**`GET /self_learning/insights`** - Get learning progress insights
**`GET /self_learning/status`** - Check self-learning system status

## 🤖 Using Self-Learning Features

### 1. **Manual Self-Learning Session**
Start a learning session via the API:
```bash
curl -X POST "http://localhost:8000/self_learning/start_session?iterations=3"
```

### 2. **Dashboard Interface**
Use the Streamlit dashboard for interactive self-learning:
```bash
streamlit run dashboard.py
```
- Navigate to the "🧠 Self-Learning System" section
- Adjust iteration count (1-10)
- Click "🚀 Start Self-Learning Session"
- View real-time progress and results

### 3. **Continuous Background Learning**
Run autonomous learning in the background:
```bash
# Single session
python scripts/continuous_learning.py --single --iterations 5

# Continuous learning (every 2 hours)
python scripts/continuous_learning.py --iterations 3 --interval 2

# View learning insights
python scripts/continuous_learning.py --insights
```

### 4. **Self-Learning Workflow**
The system automatically:
1. **Generates diverse prompts** across knowledge domains
2. **Creates responses** using the current model
3. **Evaluates quality** with empirical metrics:
   - Mathematical accuracy verification
   - Logical structure analysis
   - Completeness assessment
   - Knowledge consistency checking
4. **Stores feedback** for continuous improvement
5. **Adapts learning schedule** based on performance trends

### 5. **Monitoring Progress**
Track learning progress through:
- **Dashboard metrics**: Real-time session statistics
- **API insights**: `/self_learning/insights` endpoint
- **Log files**: `logs/self_learning.log`
- **Database entries**: Self-learning sessions stored in memory

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

## ♻️ Retrieval-Augmented Generation & Metrics (Phase 1 Implemented)

Phase 1 enhancements introduced:

* Retrieval-Augmented Generation (RAG) pathway: `/query` and `/generate` now build contextual prompts using up to 3 semantically retrieved memory snippets (graceful no-op if vector backend absent).
* Unified retrieval helper with latency measurement & semantic usage flag.
* Lightweight in-memory metrics collector (`src/metrics.py`) tracking:
  * Retrieval call counts, hit rate, semantic usage ratio, latency distribution
  * Average component scores (coherence, novelty, memory_alignment, relevance, semantic_relevance)
* Extended critic scoring with semantic relevance component (embedding proximity + lexical overlap).
* `/stats` endpoint now returns a `metrics` section when available.
* Structured logging reports retrieval utilization per response.

Example `GET /stats` excerpt:
```json
{
  "metrics": {
    "retrieval": {"calls": 12, "hit_rate": 0.58, "semantic_ratio": 0.58, "latency": {"avg_ms": 4.12}},
    "scoring": {"avg_components": {"novelty": 3.6, "semantic_relevance": 3.1}}
  }
}
```

If `weaviate` (or a future vector backend) is absent, retrieval metrics still record zero-hit events without errors.

---

**Happy learning and improving! 🎉**

## 🧩 Optional Dependencies & Fallback Behavior

The project is designed to run in lightweight (dev) or full (prod) modes by making several heavy dependencies optional. Each module degrades gracefully when a dependency is unavailable:

| Component | Optional Dependency | Fallback Behavior |
|-----------|---------------------|-------------------|
| `latent_workspace` | `torch` | Uses NumPy arrays and disables tensor-specific uncertainty refinement |
| `memory_module` | `sqlalchemy` | In-memory (no-op persistence) mode; basic structures still usable |
| `vector_memory` | `weaviate-client` | Vector search disabled; all calls become safe no-ops returning empty results |
| `config` | `pydantic_settings` | Falls back to a minimal shim reading environment variables |
| `llm_wrapper` | `transformers`, `peft` | Mock text generation & random embeddings in dev mode |

### Development vs Production Imports
Guarded imports look like:
```python
try:
  import torch
except Exception:
  torch = None  # Fallback handled downstream
```
Code paths then check availability (e.g., `if self.torch:`) before executing heavy logic.

### Protocol-Based Abstractions
To reduce tight coupling and ease testing, protocol interfaces (PEP 544) are defined in `src/interfaces.py`:

```python
class SemanticSearcher(Protocol):
  def search(self, query: str, limit: int = 5) -> list: ...

class VectorMemoryLike(Protocol):
  def add_entry(self, content: str, entry_type: str = "generic"): ...
  def search(self, query: str, limit: int = 5) -> list: ...
```

Any object implementing these methods can be injected (e.g., mocks in tests), enabling lighter unit tests without real backends.

### Suppressing External Warnings
Some third-party packages (e.g., protobuf) emit noisy version warnings during tests. These are filtered via `pytest.ini`:
```ini
[pytest]
filterwarnings =
  ignore:Protobuf gencode version.*:UserWarning
```

### When to Install Full Dependencies
Install `requirements.txt` if you need:
* Real model inference (transformers / peft)
* LoRA fine-tuning
* Persistent SQL memory
* Vector semantic search (Weaviate)

Otherwise, `requirements-dev.txt` is sufficient for logic tests, API scaffolding, and fast iteration.

### Adding a New Optional Backend
1. Wrap the import in a `try/except` block.
2. Provide a clearly named availability flag (e.g., `BACKEND_AVAILABLE`).
3. Short-circuit public methods with safe no-ops returning defaults.
4. Log a single informative warning on first use.
5. Add a Protocol if multiple interchangeable implementations are expected.

---
