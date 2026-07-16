"""
시퀀스 데이터 빌더 (LSTM / Transformer 용)

build_features() 로 만든 '일별 연속 프레임'(결측 보간 완료)에서
각 라벨 행 t 에 대해 과거 W일 [t-W+1, t] 구간의 원천 운전변수 시퀀스를 만든다.
rolling/lag 파생열은 시퀀스 모델이 스스로 학습하므로 제외하고 원천 변수만 사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TARGET_COLS


def sequence_feature_columns(df: pd.DataFrame) -> list[str]:
    """원천(비파생) 운전변수 열 = rolling/lag 파생 및 date/year/타깃 제외."""
    exclude = {"date", "year", *TARGET_COLS}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(tok in c for tok in ("_rmean", "_rstd", "_lag")):
            continue
        cols.append(c)
    return cols


def build_sequences(daily: pd.DataFrame, window: int = 14):
    """
    반환
      X_seq : (n_labeled, window, n_feat)  각 라벨 행의 과거 window일 시퀀스
      Y     : (n_labeled, 2)               [methane, MY]
      years : (n_labeled,)                 라벨 행의 연도 (분할용)
      dates : (n_labeled,)                 라벨 행 날짜
    daily 은 시간순 정렬된 일별 연속 프레임(결측 보간 완료)이어야 한다.
    """
    daily = daily.sort_values("date").reset_index(drop=True)
    feat_cols = sequence_feature_columns(daily)
    F = daily[feat_cols].to_numpy(dtype=float)

    labeled_mask = daily[TARGET_COLS].notna().all(axis=1).to_numpy()
    idx = np.where(labeled_mask)[0]

    X, Y, yrs, dts = [], [], [], []
    for t in idx:
        if t - window + 1 < 0:
            continue  # 시퀀스가 확보되지 않는 계열 시작부는 제외
        X.append(F[t - window + 1 : t + 1])
        Y.append(daily.loc[t, TARGET_COLS].to_numpy(dtype=float))
        yrs.append(int(daily.loc[t, "year"]))
        dts.append(daily.loc[t, "date"])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(yrs),
        np.asarray(dts),
        feat_cols,
    )
