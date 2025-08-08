"""Preference learning utilities for Phase 2.

Provides lightweight preference pair generation & optimization scaffolding.
Graceful degradation: if ML libs unavailable, operates in mock mode.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # Optional ML imports
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

@dataclass
class PreferencePair:
    prompt: str
    better: str
    worse: str
    better_score: float
    worse_score: float

    def to_json(self) -> str:
        return json.dumps({
            "prompt": self.prompt,
            "better": self.better,
            "worse": self.worse,
            "better_score": self.better_score,
            "worse_score": self.worse_score
        })

class PreferenceOptimizer:
    """Stores and prepares preference pairs for later optimization.

    Placeholder for DPO / logistic preference fine-tuning. For now
    it simply tracks pairs and can export a basic training signal.
    """
    def __init__(self):
        self.pairs: List[PreferencePair] = []

    def add_pair(self, pair: PreferencePair):
        self.pairs.append(pair)

    def count(self) -> int:
        return len(self.pairs)

    def export_training_examples(self) -> List[Dict[str, Any]]:
        """Return simple logistic preference tuples.
        Each item: {"prompt": str, "better": str, "worse": str, "delta": float}
        """
        examples: List[Dict[str, Any]] = []
        for p in self.pairs:
            delta = p.better_score - p.worse_score
            examples.append({
                "prompt": p.prompt,
                "better": p.better,
                "worse": p.worse,
                "delta": delta
            })
        return examples

# Global singleton (lightweight)
preference_optimizer = PreferenceOptimizer()
