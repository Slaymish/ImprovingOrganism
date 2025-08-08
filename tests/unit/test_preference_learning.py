import sys, os
os.environ['LIGHTWEIGHT_SELF_LEARNING'] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.self_learning import SelfLearningModule  # type: ignore
from src.preference_learning import preference_optimizer  # type: ignore

def test_preference_pair_generation_basic():
    sl = SelfLearningModule()
    prompt = "Explain gravity briefly"
    pairs = sl.generate_preference_pairs(prompt)
    # Should not error; pairs may be empty if model unavailable, but optimizer should remain consistent
    assert isinstance(pairs, list)
    for p in pairs:
        assert p.prompt == prompt
        assert p.better_score >= p.worse_score
    # Export format check
    exported = preference_optimizer.export_training_examples()
    for ex in exported:
        assert set(ex.keys()) == {"prompt", "better", "worse", "delta"}
