"""STEP 4 — 영 대체 Transformer 진입점 (구현: adpipe.zero_replace)."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.zero_replace import MultiplicativeReplacement, LrEMReplacement  # noqa: F401

if __name__ == "__main__":
    import numpy as np
    from adpipe.config import load_raw, to_proportion
    d = load_raw()
    zr = MultiplicativeReplacement(0.65).fit(d["X"])
    out = zr.transform(d["X"])
    print(f"δ={zr.delta_:.3e}, 최소비영={zr.min_nonzero_:.3e}")
    print(f"대체 후 영 개수={int((out.values==0).sum())}, 행합 보존(max오차)="
          f"{np.abs(out.sum(axis=1)-1).max():.2e}")
