"""Reproducibility seed utilities."""
from typing import Optional

def apply_global_seed(seed: int) -> None:
    try:
        import random, os
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch  # type: ignore
            if hasattr(torch, 'manual_seed'):
                torch.manual_seed(seed)
                if torch.cuda.is_available():  # type: ignore
                    torch.cuda.manual_seed_all(seed)  # type: ignore
        except Exception:
            pass
        # Best-effort: Python hash seed (effective on process start)
        if 'PYTHONHASHSEED' not in os.environ:
            os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception:
        pass
