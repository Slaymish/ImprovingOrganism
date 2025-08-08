# Copilot Instructions for ImprovingOrganism

## Project Overview
- **ImprovingOrganism** is a self-improving AI system with a feedback loop: generate → evaluate → collect feedback → retrain → improve.
- Core components: FastAPI REST API (`src/main.py`), self-learning logic (`src/self_learning.py`), memory system (`src/memory_module.py`), multi-metric critic (`src/critic_module.py`), LoRA-based updater (`src/updater.py`), and a Streamlit dashboard (`dashboard.py`).
- Data flows: Prompts and responses are generated, evaluated, scored, and stored in memory. Feedback triggers LoRA fine-tuning for continuous improvement.

## Key Workflows
- **Development mode:** Use `requirements-dev.txt` and `./start.sh` for lightweight, mock-LLM testing.
- **Production mode:** Use `requirements.txt` and run with real models. Set environment variables for model, LoRA, and DB config.
- **Testing:** Run `python test_dev.py` for quick tests. Full tests are in `tests/` (unit, integration, e2e).
- **Continuous learning:** Use `python scripts/continuous_learning.py` for background self-learning.
- **Dashboard:** Launch with `streamlit run dashboard.py` for interactive monitoring and control.
- **Docker:** Use `docker compose up --build` for full-stack deployment (API + training service).

## Project-Specific Patterns
- **Memory system**: All interactions (prompts, outputs, feedback) are tagged and session-grouped in `memory.db` via `src/memory_module.py`.
- **Scoring**: `CriticModule` (src/critic_module.py) evaluates responses on coherence, novelty, memory alignment, and relevance. Weights are hardcoded.
- **LoRA fine-tuning**: `src/updater.py` triggers adapter training when feedback thresholds are met.
- **Self-learning**: `src/self_learning.py` automates prompt generation, evaluation, and scheduling.
- **API endpoints**: See `src/main.py` for `/generate`, `/feedback`, `/self_learning/*`, `/stats`, `/detailed_score/{session_id}`.
- **Configuration**: Centralized in `src/config.py` (model, LoRA, DB, thresholds).

## Integration & Conventions
- **External dependencies**: FastAPI, SQLAlchemy, Streamlit, LoRA/transformers, SQLite.
- **Logs**: All logs go to `/logs`.
- **Database**: Uses SQLite by default, path configurable.
- **Session tracking**: All user/system interactions are session-grouped for analysis and retraining.
- **Testing**: Tests are organized by type in `tests/unit/`, `tests/integration/`, `tests/e2e/`.

## Examples
- Start API: `uvicorn src.main:app --host 0.0.0.0 --port 8000`
- Run demo: `python demo.py` (after API is running)
- Trigger self-learning: `curl -X POST "http://localhost:8000/self_learning/start_session?iterations=3"`

## Quick Reference
- **Key files:** `src/main.py`, `src/self_learning.py`, `src/critic_module.py`, `src/memory_module.py`, `src/updater.py`, `src/config.py`, `dashboard.py`
- **Scripts:** `scripts/continuous_learning.py`, `scripts/demo.py`, `scripts/memory_setup.py`
- **Tests:** `tests/unit/`, `tests/integration/`, `tests/e2e/`

---
For more details, see `README.md` and code comments in each module.
