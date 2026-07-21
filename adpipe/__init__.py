"""
adpipe — 혐기성 소화조 16S RA 데이터용 누수-없는(leakage-free) 조성자료(CoDA)
전처리·모델링 파이프라인.

설계 원칙 (PROMPT 1/2):
  * 모든 전처리 파라미터는 fit()에서 '학습 fold'만 보고 학습한다(누수 금지).
  * 변환/필터는 sklearn 호환 Transformer 로 구현 → 단일 Pipeline 직렬화 가능.
  * 계절은 배치가 아니라 '특징(feature)'. site 는 CV 그룹키(특징 아님).
  * 하나의 정답 변환을 미리 고르지 않는다 → TRANSFORM_REGISTRY 병렬 산출.

파일명이 숫자로 시작하는 src/0X_*.py 는 실행 진입점(runner)이며,
재사용 클래스는 모두 이 패키지에 있다(피클 안정성).
"""
from .config import CONFIG, load_raw
__all__ = ["CONFIG", "load_raw"]
