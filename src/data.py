"""
데이터 로딩 · 전처리 · 생물학 기반 Feature Engineering
BioGuard-AI : 혐기성 소화조 메탄 생성량 예측 및 건강상태 관제

영천 통합바이오가스화시설 운전 데이터(2018.01~2023.12, 일별).
 - master.xlsx  : 운전 변수(독립변수 후보)
 - targets.xlsx : methane(메탄발생량), VS_in(투입 VS부하), MY(=methane/VS_in)

────────────────────────────────────────────────────────────────────────
생물학적 모델링 원칙 (혐기성소화 메커니즘)
────────────────────────────────────────────────────────────────────────
혐기성소화는 가수분해→산생성(VFA)→아세트산생성→메탄생성의 다단계 미생물 반응으로,
투입 유기물(VS)이 즉시 메탄으로 전환되지 않고 체류시간(HRT)만큼 지연된다. 따라서
'오늘의 메탄생성량'은 (1) 과거 수일~수십일 누적 투입 기질부하 + (2) 현재 소화조 내부
미생물 상태(VFA·알칼리도·pH·온도·VS)의 함수이다. 교차상관 분석에서 투입 VS부하의
7~10일 누적(이동평균)이 메탄과 가장 강하게 상관(r≈0.64 > 순간값 0.54)하여 지연을
정량 확인하였고, 이를 반영해 체류창 부하(load5/10) 피처를 별도로 설계한다.

■ 수율(MY=methane/VS_in)은 '예측 대상'에서 제외
  분모가 금일 투입 VS이므로 어제까지 데이터로 오늘 수율을 예측하는 것은 순환적이며
  생물학적으로 성립하지 않는다. 예측 목표는 메탄생성량 단일, 수율은 진단용 참조값.

■ 건강상태(관제)는 VFA/알칼리도 비 기반
  산성화(완충능 소진) 지표. AD 문헌 밴드 <0.3 안정 / 0.3~0.4 주의 / 0.4~0.8 불안정 /
  ≥0.8 위험. 신호등은 수율이 아닌 이 건강지표로 정의한다(src/signal.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 결측 과다(75~87%)로 제외하는 열 : 투입 TN, 소화조 TN, 소화조 NH4N
HIGH_MISSING_COLS = ["유입_TN", "소화조_TN", "소화조_NH4N"]

# 예측 목표 : 메탄생성량 단일 (수율 MY 는 예측하지 않고 진단·건강 참조용)
TARGET_COL = "methane"
DIAG_COLS = ["MY"]

# 리치 시계열 통계를 생성할 운전변수(부하·내부상태·투입특성)
TS_BASE_VARS = [
    "VS_in", "유입_VS", "소화조_VS", "소화조_VFA", "소화조_TAlk",
    "소화조_pH", "소화조_온도", "투입량합계", "반입량",
]
ROLL_WINDOWS = [7, 14, 30]
LAGS = [1, 7, 14]

# 생물학적 체류창 누적 부하(이동평균)를 추가로 부여할 기질부하 변수
LOADING_VARS = ["VS_in", "투입량합계", "유입_VS"]
RETENTION_WINDOWS = [5, 10]  # 7·14 는 위 ROLL_WINDOWS 로 이미 생성됨


def load_raw(master_path: str, targets_path: str) -> pd.DataFrame:
    """master + targets 를 date 기준 병합한 일별 원본 프레임."""
    m = pd.read_excel(master_path)
    t = pd.read_excel(targets_path)
    df = m.merge(t[["date", "methane", "MY"]], on="date", how="left")
    return df.sort_values("date").reset_index(drop=True)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """건강·부하 파생변수."""
    df = df.copy()
    if {"소화조_VFA", "소화조_TAlk"}.issubset(df.columns):
        # 산성화·건강 지표
        df["VFA_ALK"] = (df["소화조_VFA"] / df["소화조_TAlk"].replace(0, np.nan))
    if {"VS_in", "투입량합계"}.issubset(df.columns):
        df["VS_load_ratio"] = (df["VS_in"] / df["투입량합계"].replace(0, np.nan))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    일별 연속 시계열에 대해 결측 보간 → 파생 → 리치 시계열 + 생물학 체류부하 피처.
    (타깃 methane 은 보간하지 않고 원 실측만 라벨로 사용)
    """
    df = df.copy()

    # 1) 결측 과다 열 제거
    df = df.drop(columns=[c for c in HIGH_MISSING_COLS if c in df.columns])

    # 2) 피처 후보 결측 채움 — 인과적(과거 방향)으로만.
    #    v1 은 여기서 `interpolate(method="linear", limit_direction="both")` 를 썼으나
    #    이는 분할 이전에 전체 계열에 적용되므로 bfill 성분이 미래값을 과거로 끌어온다
    #    (학습셋이 홀드아웃 구간 정보를 본다). ffill 만 쓰면 과거만 참조하므로 누출이 없다.
    #    ffill 후 남는 결측은 '첫 관측 이전' 선두 구간뿐이고, 이를 채우는 값은 반드시
    #    최초 관측치(2018년, 학습 구간)이므로 마지막에 bfill 로 마감해도 누출이 아니다.
    exclude = {"date", "year", TARGET_COL, *DIAG_COLS}
    base_feats = [c for c in df.columns if c not in exclude and df[c].dtype != "O"]
    df[base_feats] = df[base_feats].ffill()

    # 3) 파생(건강·부하) — 단일 피처로 사용
    df = _add_derived(df)
    for c in ["VFA_ALK", "VS_load_ratio"]:
        if c in df.columns:
            df[c] = df[c].ffill()

    new = {}
    # 4) 리치 시계열 통계 : rmean·rstd·lag
    for v in TS_BASE_VARS:
        if v not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            new[f"{v}_rmean{w}"] = df[v].rolling(w, min_periods=1).mean()
            new[f"{v}_rstd{w}"] = df[v].rolling(w, min_periods=1).std().fillna(0.0)
        for lg in LAGS:
            new[f"{v}_lag{lg}"] = df[v].shift(lg)

    # 5) 생물학적 체류창 누적 부하(이동평균)
    for v in LOADING_VARS:
        if v not in df.columns:
            continue
        for w in RETENTION_WINDOWS:
            new[f"{v}_load{w}"] = df[v].rolling(w, min_periods=1).mean()

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ffill 이후 남은 결측은 계열 선두(첫 관측 이전)뿐이다. 이 구간을 bfill 로 마감하면
    # 채워지는 값은 항상 최초 관측치 = 2018년 학습 구간의 값이므로 홀드아웃 누출이 없다.
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype != "O"]
    df[feat_cols] = df[feat_cols].bfill()
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """모델 입력 피처 열 (date/year/타깃/진단 제외)."""
    exclude = {"date", "year", TARGET_COL, *DIAG_COLS}
    return [c for c in df.columns if c not in exclude and df[c].dtype != "O"]


def train_holdout_split(df: pd.DataFrame, holdout_year: int = 2023):
    """시계열 분할 : holdout_year 이전=학습, holdout_year=홀드아웃. 실측 methane 행만."""
    labeled = df.dropna(subset=[TARGET_COL]).copy()
    train = labeled[labeled["year"] < holdout_year].reset_index(drop=True)
    test = labeled[labeled["year"] == holdout_year].reset_index(drop=True)
    return train, test
