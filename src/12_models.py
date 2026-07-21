"""모델 레지스트리 진입점 (구현: adpipe.models)."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.models import (MODEL_REGISTRY, TIER1, TIER2,  # noqa: F401
                           EXCLUDED_MODELS)

if __name__ == "__main__":
    print("채택 모델:")
    for k, v in MODEL_REGISTRY.items():
        print(f"  [T{v['tier']}] {k:11s} 후보={len(v['grid']):2d}  변환={v['transforms']}")
    print("\n배제 모델(사유):")
    for k, why in EXCLUDED_MODELS.items():
        print(f"  {k}: {why}")
