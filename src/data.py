"""
데이터 로딩 · 전처리 · 시계열 Feature Engineering
BioGuard-AI : 혐기성 소화조 메탄 발생량·수율 예측

영천 통합바이오가스화시설 운전 데이터(2018.01~2023.12, 일별)를 사용한다.
 - master.xlsx  : 운전 변수(독립변수 후보)
 - targets.xlsx : methane(메탄발생량), VS_in(투입 VS부하), MY(메탄발생비=수율)

핵심 관계
 - MY = methane / VS_in  (결정론적, 오차 ~1e-15)
   -> VS_in 은 투입 부하로서 예측 시점에 알 수 있는 입력 변수이므로 피처로 유지한다.
     (methane 은 종속변수이므로 피처에서 제외)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 결측 과다(75~87%)로 보고서에서 제외한 열 : 투입 TN, 소화조 TN, 소화조 NH4N
HIGH_MISSING_COLS = ["유입_TN", "소화조_TN", "소화조_NH4N"]

# 두 종속변수
TARGET_COLS = ["methane", "MY"]

# 시계열 피처를 생성할 핵심 운전변수
TS_BASE_VARS = [
    "VS_in",
    "유입_VS",
    "소화조_VS",
    "소화조_VFA",
    "소화조_TAlk",
    "소화조_pH",
    "소화조_온도",
    "투입량합계",
    "반입량",
]

ROLL_WINDOWS = [7, 14, 30]   # rolling window (일)
LAGS = [1, 7, 14]            # 자기회귀(lag) (일)


def load_raw(master_path: str, targets_path: str) -> pd.DataFrame:
    """master + targets 를 date 기준으로 병합한 일별 원본 프레임 반환."""
    m = pd.read_excel(master_path)
    t = pd.read_excel(targets_path)
    df = m.merge(t[["date", "methane", "MY"]], on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """공정 의미가 있는 파생변수 추가 (누수 없는 입력변수만 사용)."""
    df = df.copy()
    # 산성화 지표 : VFA / 알칼리도 비
    if {"소화조_VFA", "소화조_TAlk"}.issubset(df.columns):
        df["VFA_TAlk_ratio"] = df["소화조_VFA"] / df["소화조_TAlk"].replace(0, np.nan)
    # 유기물부하율 대용 : 투입 VS 부하 대비 투입량
    if {"VS_in", "투입량합계"}.issubset(df.columns):
        df["VS_load_ratio"] = df["VS_in"] / df["투입량합계"].replace(0, np.nan)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    일별 연속 시계열에 대해 결측 보간 → 파생변수 → rolling / lag 피처 생성.

    시계열 피처는 '전체 일별 계열'에서 시간순으로 만들어야 rolling/lag 이 연속적으로
    계산된다(라벨 유무와 무관). 이후 train.py 에서 라벨이 있는 행만 학습에 사용한다.
    """
    df = df.copy()

    # 1) 결측 과다 열 제거
    drop_cols = [c for c in HIGH_MISSING_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 2) 파생변수
    df = _add_derived(df)

    # 3) 입력변수 후보 = date/year/타깃 제외 전체 수치열
    exclude = {"date", "year", *TARGET_COLS}
    feat_cols = [c for c in df.columns if c not in exclude]

    # 4) 소량 결측 선형 보간 (시간순) + 양끝 채움
    df[feat_cols] = (
        df[feat_cols]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    # 5) 시계열 피처 : rolling mean/std + lag
    ts_vars = [c for c in TS_BASE_VARS if c in df.columns]
    new_cols = {}
    for v in ts_vars:
        for w in ROLL_WINDOWS:
            new_cols[f"{v}_rmean{w}"] = df[v].rolling(w, min_periods=1).mean()
            new_cols[f"{v}_rstd{w}"] = df[v].rolling(w, min_periods=1).std().fillna(0.0)
        for lag in LAGS:
            new_cols[f"{v}_lag{lag}"] = df[v].shift(lag)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # lag 로 생긴 앞부분 결측은 후방 채움 (계열 시작부 소수 행)
    lag_cols = [c for c in new_cols if "_lag" in c]
    df[lag_cols] = df[lag_cols].bfill()

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """모델 입력 피처 열 목록 (date/year/타깃 제외)."""
    exclude = {"date", "year", *TARGET_COLS}
    return [c for c in df.columns if c not in exclude]


def train_holdout_split(df: pd.DataFrame, holdout_year: int = 2023):
    """
    시계열 분할 : holdout_year 이전 = 학습, holdout_year = 홀드아웃(미래).
    두 종속변수가 모두 존재하는 행만 사용한다.
    """
    labeled = df.dropna(subset=TARGET_COLS).copy()
    train = labeled[labeled["year"] < holdout_year].reset_index(drop=True)
    test = labeled[labeled["year"] == holdout_year].reset_index(drop=True)
    return train, test
