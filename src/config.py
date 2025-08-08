import os, json
from typing import Optional, Dict

# Support environments where pydantic-settings isn't installed (e.g. lightweight dev deps)
try:
    from pydantic_settings import BaseSettings  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    try:
        from pydantic import BaseSettings  # type: ignore
    except ImportError:
        # Minimal fallback shim so tests that only touch settings still work
        class BaseSettings:  # type: ignore
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

class Settings(BaseSettings):
    model_name: str = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    lora_path: str = os.getenv("LORA_PATH", "./adapters/hamish_lora")
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/memory.db")
    feedback_threshold: int = 50
    batch_size: int = 8
    fast_start: bool = os.getenv("FAST_START", "false").lower() == "true"
    
    # Memory management settings
    force_cpu: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    max_gpu_memory_gb: float = float(os.getenv("MAX_GPU_MEMORY_GB", "4.0"))
    enable_memory_optimization: bool = os.getenv("ENABLE_MEMORY_OPTIMIZATION", "true").lower() == "true"

    # Weaviate settings
    weaviate_url: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY", None)
    # Critic scoring weights (can override via SCORING_WEIGHTS JSON env)
    _default_scoring_weights: Dict[str, float] = {
        "coherence": 0.08,
        "novelty": 0.22,
        "memory_alignment": 0.25,
        "relevance": 0.30,
        "semantic_relevance": 0.15
    }
    _scoring_weights: Dict[str, float] = None  # type: ignore

    def __init__(self, **data):
        super().__init__(**data)
        raw = os.getenv("SCORING_WEIGHTS", "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    self._scoring_weights = {**self._default_scoring_weights, **{k: float(v) for k,v in parsed.items() if k in self._default_scoring_weights}}
            except Exception:
                self._scoring_weights = None
        if self._scoring_weights is None:
            self._scoring_weights = self._default_scoring_weights.copy()
        # Normalize just in case
        total = sum(self._scoring_weights.values())
        if total > 0:
            self._scoring_weights = {k: v/total for k,v in self._scoring_weights.items()}

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return self._scoring_weights
    
settings = Settings()
