import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = os.getenv("MODEL_NAME", "gpt-oss-20b")
    lora_path: str = os.getenv("LORA_PATH", "./adapters/hamish_lora")
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/memory.db")
    feedback_threshold: int = 50
    batch_size: int = 8

settings = Settings()
