"""Common protocol (interface) definitions to improve static type safety.

These are intentionally light-weight so optional dependencies (like weaviate
or torch) can remain optional while downstream code can still express intent.
"""
from __future__ import annotations

from typing import Protocol, List, Dict, Any, Optional


class SemanticSearcher(Protocol):
    def search_semantic(self, query: str, limit: int = 5, entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]: ...  # pragma: no cover


class VectorMemoryLike(SemanticSearcher, Protocol):
    def add_entry(self, content: str, entry_type: str, session_id: Optional[str] = None, score: Optional[float] = None, timestamp: Optional[str] = None) -> None: ...  # pragma: no cover


__all__ = [
    "SemanticSearcher",
    "VectorMemoryLike",
]
