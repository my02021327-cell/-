"""학습/검증/시험 분할 정의 — 6년 원자료 기준 확정안.

설계 근거
  · 메탄 자기상관 0.93 → 무작위 분할은 미래가 과거로 새어 R2 가 크게 부풀려진다.
    시간순 분할만 유효하다.
  · 단일 홀드아웃 연도를 바꾸면 R2 가 0.32~0.68 로 흔들린다(측정 확인).
    따라서 **모델 선택은 롤링 CV 로 하고, 홀드아웃은 확인용 1회**로만 쓴다.
  · 연 주기(온도·반입량 계절성)가 있어 최소 학습은 2년으로 둔다.
  · 2018-10 충격 사건은 최소학습 2년 규칙에서는 항상 학습에 들어가 절대
    검증되지 않는다. 그래서 **별도 스트레스 폴드**로 분리한다.

데이터 경계
  · 유효 마지막 날 2023-09-17 (이후 투입량 0 = 기록 종료)
  · 가스 발생량은 매일(365일/년), CH4 함량은 평일 → 메탄 라벨 1,257일
"""
from __future__ import annotations

import pandas as pd

# ── 경계 ──────────────────────────────────────────────────────────────────────
DATA_START = pd.Timestamp("2018-01-01")
DATA_END = pd.Timestamp("2023-09-17")          # 이후 투입량 0, 기록 종료

DEV_END = pd.Timestamp("2022-12-31")           # 개발 구간 끝
TEST_START = pd.Timestamp("2023-01-01")        # 봉인 홀드아웃 시작

CV_MIN_TRAIN_END = pd.Timestamp("2019-12-31")  # 최소 학습 = 2년(계절 2회)
CV_BLOCK = 120                                 # 검증창(달력일)
CV_STEP = 120

# 충격 스트레스 폴드 : 2018-10 고농도 유입(유기산화조 VS 최대 5.4배, 가스 200%)
SHOCK_TRAIN_END = pd.Timestamp("2018-09-15")
SHOCK_TEST = (pd.Timestamp("2018-09-16"), pd.Timestamp("2018-11-30"))


def cv_folds() -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """(학습끝, 검증시작, 검증끝) — 확장창 롤링 오리진."""
    out = []
    tr_end = CV_MIN_TRAIN_END
    while True:
        te_start = tr_end + pd.Timedelta(days=1)
        te_end = te_start + pd.Timedelta(days=CV_BLOCK - 1)
        if te_end > DEV_END:
            break
        out.append((tr_end, te_start, te_end))
        tr_end = tr_end + pd.Timedelta(days=CV_STEP)
    return out


def describe(labels: pd.Series) -> None:
    """labels : 메탄 라벨이 존재하는 날만 True 인 불리언 시리즈(달력일 인덱스)."""
    def n(a, b):
        return int(labels.loc[a:b].sum())

    print("=" * 72)
    print("최종 분할")
    print("=" * 72)
    print(f"  개발 구간   {DATA_START.date()} ~ {DEV_END.date()}   라벨 {n(DATA_START, DEV_END):5d}일")
    print(f"  봉인 홀드아웃 {TEST_START.date()} ~ {DATA_END.date()}   라벨 {n(TEST_START, DATA_END):5d}일")
    print(f"\n  롤링 CV (확장창, 최소학습 2년, 검증창 {CV_BLOCK}일, 간격 {CV_STEP}일)")
    for i, (tr, a, b) in enumerate(cv_folds(), 1):
        print(f"    폴드{i}: 학습 ~{tr.date()} ({n(DATA_START, tr):4d}일) → "
              f"검증 {a.date()}~{b.date()} ({n(a, b):3d}일)")
    a, b = SHOCK_TEST
    print(f"\n  충격 스트레스(별도 보고): 학습 ~{SHOCK_TRAIN_END.date()} "
          f"({n(DATA_START, SHOCK_TRAIN_END):3d}일) → 검증 {a.date()}~{b.date()} ({n(a, b):3d}일)")


if __name__ == "__main__":
    import numpy as np

    import src.raw_extract as rx

    d = rx.load_all()

    def c(*t):
        x = rx.pick(d, *t)
        return d[x[0]].replace(0, np.nan) if x else None

    gas = (c("혐기성소화조", "#A", "바이오가스발생량").fillna(0)
           + c("혐기성소화조", "#B", "바이오가스발생량").fillna(0)).replace(0, np.nan)
    a4, b4 = c("혐기성소화조", "#A", "메탄함량"), c("혐기성소화조", "#B", "메탄함량")
    ch4 = pd.concat([a4.where(a4.between(45, 80)), b4.where(b4.between(45, 80))],
                    axis=1).mean(axis=1)
    describe((gas * ch4).notna())
