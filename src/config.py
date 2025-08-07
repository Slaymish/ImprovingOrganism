import os
from pydantic_settings import BaseSettings

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

settings = Settings()
