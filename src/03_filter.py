"""STEP 3 — 필터링 Transformer 진입점.

구현 클래스는 피클 안정성을 위해 `adpipe.filter` 에 있으며 여기서 재노출한다.
직접 실행하면 데모(잔여분/통과 taxa 수)를 출력한다.
"""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.filter import PrevalenceAbundanceFilter  # noqa: F401  (deliverable class)

if __name__ == "__main__":
    from adpipe.config import load_raw
    d = load_raw()
    for mp in (0.1, 0.25, 0.5):
        f = PrevalenceAbundanceFilter(min_prevalence=mp).fit(d["X"])
        print(f"min_prevalence={mp}: 통과 {len(f.kept_)}개, "
              f"residual_fraction_={f.residual_fraction_:.3f}")
