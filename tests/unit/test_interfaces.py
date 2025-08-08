import sys, os
from typing import List, Dict, Any
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.interfaces import SemanticSearcher

class DummySearcher:
    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results
    def search_semantic(self, query: str, limit: int = 5, entry_types=None):  # noqa: D401
        return self._results[:limit]


def test_protocol_like_usage():
    results = [{"content": "example"}]
    searcher: SemanticSearcher = DummySearcher(results)  # type: ignore
    out = searcher.search_semantic("query")
    assert out and out[0]["content"] == "example"
