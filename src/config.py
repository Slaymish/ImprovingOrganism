import os
from typing import Optional

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
    
    # Memory management settings
    force_cpu: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    max_gpu_memory_gb: float = float(os.getenv("MAX_GPU_MEMORY_GB", "4.0"))
    enable_memory_optimization: bool = os.getenv("ENABLE_MEMORY_OPTIMIZATION", "true").lower() == "true"

    # Weaviate settings
    weaviate_url: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY", None)
    
settings = Settings()
