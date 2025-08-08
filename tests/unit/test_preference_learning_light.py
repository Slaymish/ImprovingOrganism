import os, sys
os.environ['LIGHTWEIGHT_SELF_LEARNING']='1'
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, src_path)
from src.self_learning import SelfLearningModule  # type: ignore
from src.preference_learning import preference_optimizer  # type: ignore

def test_preference_pairs_light():
    sl = SelfLearningModule()
    pairs = sl.generate_preference_pairs('Test prompt about physics')
    assert isinstance(pairs, list)
    # Even if empty, export should work
    exported = preference_optimizer.export_training_examples()
    assert isinstance(exported, list)
