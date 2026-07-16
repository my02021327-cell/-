"""
시퀀스 데이터 빌더 (LSTM / Transformer 용)

build_features() 의 '일별 연속 프레임'에서 각 라벨 행 t 에 대해 과거 W일
[t-W+1, t] 구간의 원천 운전변수 시퀀스를 만든다. 시퀀스 모델은 시간 구조를
스스로 학습하므로 rolling/lag/load 파생열은 제외하고 원천 변수만 사용한다.
타깃은 메탄생성량(단일)이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import DIAG_COLS, TARGET_COL

_ENG_TOKENS = ("_load", "_lag", "_rmean", "_rstd")


def sequence_feature_columns(df: pd.DataFrame) -> list[str]:
    """원천(비파생) 운전변수 열 = 파생 및 date/year/타깃/진단 제외."""
    exclude = {"date", "year", TARGET_COL, *DIAG_COLS}
    cols = []
    for c in df.columns:
        if c in exclude or df[c].dtype == "O":
            continue
        if any(tok in c for tok in _ENG_TOKENS):
            continue
        cols.append(c)
    return cols


def build_sequences(daily: pd.DataFrame, window: int = 14):
    """
    반환
      X_seq : (n, window, n_feat)
      Y     : (n, 1)   메탄생성량
      years : (n,)     연도
      dates : (n,)     날짜
      feat_cols : 시퀀스 피처명
    """
    daily = daily.sort_values("date").reset_index(drop=True)
    feat_cols = sequence_feature_columns(daily)
    F = daily[feat_cols].to_numpy(dtype=float)

    labeled = daily[TARGET_COL].notna().to_numpy()
    idx = np.where(labeled)[0]

    X, Y, yrs, dts = [], [], [], []
    for t in idx:
        if t - window + 1 < 0:
            continue
        X.append(F[t - window + 1 : t + 1])
        Y.append([float(daily.loc[t, TARGET_COL])])
        yrs.append(int(daily.loc[t, "year"]))
        dts.append(daily.loc[t, "date"])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(yrs),
        np.asarray(dts),
        feat_cols,
    )
