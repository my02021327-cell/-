"""STEP 6 — 파생 특징 블록 진입점 (구현: adpipe.derived)."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.derived import (AlphaDiversity, BetaPCoA, DomainRatios,  # noqa: F401
                            DesignFeatures)

if __name__ == "__main__":
    from adpipe.config import load_raw
    d = load_raw()
    ctx = dict(taxonomy=d["taxonomy"], domain_taxa=d["domain_taxa"], meta=d["meta"])
    a = AlphaDiversity().fit_transform(d["X"])
    dm = DomainRatios(d["taxonomy"], d["domain_taxa"]).fit(d["X"])
    print("블록 A 알파다양성 shape:", a.shape, "->", list(AlphaDiversity().get_feature_names_out()))
    print("블록 C 도메인 매칭 길드:", sorted(dm.matched_))
    print("블록 C 미매칭 길드:", sorted(dm.unmatched_))
    print("블록 C 특징:", list(dm.get_feature_names_out()))
