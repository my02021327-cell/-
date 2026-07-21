"""평가 프로토콜 진입점 (구현: adpipe.cv). 확정 후 변경 금지."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.cv import (nested_cv_oof, inner_select, optimistic_best,  # noqa: F401
                       random_kfold_r2, modal_config, seed_r2_summary)

PROTOCOL = """평가 프로토콜 (변경 금지)
  외부 루프 : LeaveOneGroupOut(site) → 10 fold
  내부 루프 : GroupKFold(5) on training sites → (변환 × 필터 × 하이퍼파라미터) 선택
  반복      : seed 5개(내부분할/모델시드) → 분산 추정
  최종 성능 : 반드시 중첩 CV '외부 루프' 값. 탐색 그리드 최고점은 낙관 편향으로 별도 표기.
  누수 진단 : 무작위 K-fold(라벨=누수) 를 나란히 보고해 격차를 드러냄."""

if __name__ == "__main__":
    print(PROTOCOL)
