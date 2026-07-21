"""STEP 5 — 변환 세트 진입점 (구현: adpipe.transforms). TRANSFORM_REGISTRY 재노출."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.transforms import TRANSFORM_REGISTRY, make_transform  # noqa: F401

if __name__ == "__main__":
    print("등록된 변환(TRANSFORM_REGISTRY):")
    for k, v in TRANSFORM_REGISTRY.items():
        print(f"  {k:10s}  영대체필요={str(v['needs_zero_replace']):5s}  scaler={v['scaler']}")
