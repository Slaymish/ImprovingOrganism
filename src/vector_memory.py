import weaviate
from weaviate.auth import AuthApiKey
import logging
from typing import Optional, List, Dict, Any
import logging

# Optional dependency: weaviate
try:  # pragma: no cover - import guard
    import weaviate  # type: ignore
    from weaviate.auth import AuthApiKey  # type: ignore
    WEAVIATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    weaviate = None  # type: ignore
    AuthApiKey = None  # type: ignore
    WEAVIATE_AVAILABLE = False
from .config import settings
from .llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)

class VectorMemory:
    def __init__(self):
        self.client = self._connect() if WEAVIATE_AVAILABLE else None
        self.llm_wrapper = LLMWrapper()
        self.class_name = "MemoryEntry"
        self._ensure_schema()

    def _connect(self) -> Optional[weaviate.Client]:
        if not WEAVIATE_AVAILABLE:
            logger.warning("Weaviate not installed; VectorMemory operating in no-op mode.")
            return None
        try:
            client = weaviate.Client(
                url=settings.weaviate_url,
                auth_client_secret=AuthApiKey(api_key=settings.weaviate_api_key) if settings.weaviate_api_key else None
            )
            if client.is_ready():
                logger.info("Weaviate client connected successfully.")
                return client
            logger.error("Weaviate is not ready.")
            return None
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to connect to Weaviate: {e}")
            return None

    def _ensure_schema(self):
        if not self.client:
            return

        if not self.client.schema.exists(self.class_name):
            logger.info(f"Creating schema for class: {self.class_name}")
            schema = {
                "class": self.class_name,
                "vectorizer": "none",  # We will provide our own vectors
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "entry_type", "dataType": ["string"]},
                    {"name": "session_id", "dataType": ["string"]},
                    {"name": "score", "dataType": ["number"]},
                    {"name": "timestamp", "dataType": ["date"]},
                ],
            }
            self.client.schema.create_class(schema)
            logger.info("Schema created successfully.")

    def add_entry(self, content: str, entry_type: str, session_id: Optional[str] = None, score: Optional[float] = None, timestamp: Optional[str] = None):
        if not self.client:
            logger.warning("Weaviate client not available. Skipping entry.")
            return

        try:
            embedding = self.llm_wrapper.get_embeddings(content)
            if embedding is None:
                logger.error("Failed to generate embedding. Skipping entry.")
                return

            data_object = {
                "content": content,
                "entry_type": entry_type,
                "session_id": session_id,
                "score": score,
                "timestamp": timestamp,
            }

            self.client.data_object.create(
                data_object=data_object,
                class_name=self.class_name,
                vector=embedding.tolist()
            )
            logger.info(f"Added entry of type '{entry_type}' to vector memory.")
        except Exception as e:
            logger.error(f"Failed to add entry to Weaviate: {e}")

    def search(self, query: str, limit: int = 5, entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.client:
            logger.warning("Weaviate client not available. Returning empty search results.")
            return []

        try:
            query_embedding = self.llm_wrapper.get_embeddings(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding. Returning empty search results.")
                return []

            near_vector = {"vector": query_embedding.tolist()}
            
            where_filter = None
            if entry_types:
                where_filter = {
                    "path": ["entry_type"],
                    "operator": "ContainsAny",
                    "valueString": entry_types,
                }

            result = (
                self.client.query
                .get(self.class_name, ["content", "entry_type", "session_id", "score", "timestamp", "_additional {distance}"])
                .with_near_vector(near_vector)
                .with_limit(limit)
                .with_where(where_filter)
                .do()
            )
            
            return result.get("data", {}).get("Get", {}).get(self.class_name, [])
        except Exception as e:
            logger.error(f"Failed to perform vector search: {e}")
            return []
