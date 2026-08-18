"""
피처 구성 — 네 각도 × 시계열 변환, 전부 인과적(원점 t 이전만 참조)

원점 t 에서 y(t+h) 를 예측하므로, 모든 피처는 **t 시점까지의 정보만** 써야 한다.
따라서
  · 결측은 인과적 ffill 로만 메운다(선형보간·bfill 금지 — 미래를 끌어온다).
  · 이동통계는 전부 과거창(rolling)이며 중심창을 쓰지 않는다.
  · 타깃 되먹임(y 의 시차·이동평균)은 t 시점까지 관측된 값만 쓴다.

각 각도는 독립적으로 켜고 끌 수 있다(ablation). 「어느 각도가 실제로 정보를 주는가」는
성능 자체만큼 중요한 산출물이다(요구사항 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bgp import config as C
from src.bgp.data import ANGLES

# 각 원천 변수에 붙일 시계열 변환
LAGS = (1, 3, 7, 14)
ROLLS = (3, 7, 14, 30)
TREND_WINDOWS = (7, 30)


def _causal_block(df: pd.DataFrame, cols: list[str], prefix: str,
                  lags=LAGS, rolls=ROLLS) -> pd.DataFrame:
    """한 각도의 원천 변수들을 인과적 시계열 피처로 확장한다."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    base = df[cols].ffill()                       # 인과적 채움만
    out = {f"{prefix}__{c}": base[c] for c in cols}
    for c in cols:
        s = base[c]
        for w in rolls:
            out[f"{prefix}__{c}_ma{w}"] = s.rolling(w, min_periods=1).mean()
        for lg in lags:
            out[f"{prefix}__{c}_lag{lg}"] = s.shift(lg)
        for w in TREND_WINDOWS:                   # 추세 = 현재 − w일 평균
            out[f"{prefix}__{c}_tr{w}"] = s - s.rolling(w, min_periods=1).mean()
    return pd.DataFrame(out, index=df.index)


def build_features(df: pd.DataFrame, angles: tuple[str, ...] | None = None,
                   target_col: str | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    반환 (피처 프레임, 각도→열이름 목록).

    `target_col` 은 시계열 각도에서 되먹임으로 쓸 타깃 계열이다. 재구성 타깃
    (`CH4_m3d_filled`)을 넣으면 되먹임이 조밀해지고, 원 관측(`CH4_m3d`)을 넣으면
    실제 측정만 쓴다. 어느 쪽이든 **t 시점까지의 값만** 참조한다.
    """
    angles = angles or tuple(ANGLES)
    target_col = target_col or C.TARGET
    blocks, groups = [], {}

    if "substrate" in angles:
        b = _causal_block(df, ANGLES["substrate"], "sub")
        blocks.append(b); groups["substrate"] = list(b.columns)
    if "chemistry" in angles:
        b = _causal_block(df, ANGLES["chemistry"], "chem")
        blocks.append(b); groups["chemistry"] = list(b.columns)
    if "vsbalance" in angles:
        b = _causal_block(df, ANGLES["vsbalance"], "vs")
        blocks.append(b); groups["vsbalance"] = list(b.columns)

    if "temporal" in angles:
        tb = {}
        # 유량 — 결측 0 % 인 유일한 완전 계열이다. 시계열 정보의 주공급원.
        flow = df[C.FLOW].ffill()
        tb["tmp__flow"] = flow
        for w in (3, 7, 14, 30, 60):
            tb[f"tmp__flow_ma{w}"] = flow.rolling(w, min_periods=1).mean()
            tb[f"tmp__flow_sd{w}"] = flow.rolling(w, min_periods=2).std()
        for lg in (1, 2, 3, 7, 14, 30):
            tb[f"tmp__flow_lag{lg}"] = flow.shift(lg)
        for w in (7, 30):
            tb[f"tmp__flow_tr{w}"] = flow - flow.rolling(w, min_periods=1).mean()

        # 농도
        conc = df.get("CH4_pct_filled", df[C.CONC]).ffill()
        tb["tmp__conc"] = conc
        for w in (7, 30):
            tb[f"tmp__conc_ma{w}"] = conc.rolling(w, min_periods=1).mean()
        tb["tmp__conc_tr30"] = conc - conc.rolling(30, min_periods=1).mean()

        # 타깃 되먹임 — 원점까지 알려진 메탄
        y = df[target_col].ffill()
        tb["tmp__y_last"] = y
        for w in (3, 7, 14, 30, 60):
            tb[f"tmp__y_ma{w}"] = y.rolling(w, min_periods=1).mean()
        for lg in (1, 3, 7, 14, 30):
            tb[f"tmp__y_lag{lg}"] = y.shift(lg)
        tb["tmp__y_tr7"] = y - y.rolling(7, min_periods=1).mean()
        tb["tmp__y_tr30"] = y - y.rolling(30, min_periods=1).mean()
        # 관측 신선도 — 마지막 실측이 며칠 전인지(재구성 라벨과 실측을 구분하는 신호)
        obs_pos = pd.Series(np.where(df[C.TARGET].notna(), np.arange(len(df)), np.nan),
                            index=df.index).ffill()
        tb["tmp__y_age"] = np.arange(len(df)) - obs_pos

        # 달력
        tb["tmp__dow"] = df["date"].dt.dayofweek
        tb["tmp__month"] = df["date"].dt.month
        doy = df["date"].dt.dayofyear
        tb["tmp__doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        tb["tmp__doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        b = pd.DataFrame(tb, index=df.index)
        blocks.append(b); groups["temporal"] = list(b.columns)

    X = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, groups


def horizon_pairs(X: pd.DataFrame, y: pd.Series, observed: pd.Series,
                  lo: int, hi: int, min_origin: int = 0):
    """
    지평 구간 (lo, hi) 의 (원점 t, 지평 h) 쌍을 만든다.

    직접 다단계(direct multi-step) 방식 — h 를 **피처로** 넣어 구간 안의 여러 지평이
    같은 모델을 공유한다. 재귀 예측과 달리 오차가 누적되지 않고, 지평별 개별 모델보다
    표본을 효율적으로 쓴다.

    반환 origin_idx, h, y_target, y_observed_flag
    """
    n = len(X)
    origins, hs = [], []
    for h in range(lo, hi + 1):
        t = np.arange(min_origin, n - h)
        origins.append(t)
        hs.append(np.full(t.shape, h))
    o = np.concatenate(origins)
    h = np.concatenate(hs)
    order = np.lexsort((h, o))
    o, h = o[order], h[order]
    tgt = y.to_numpy(float)[o + h]
    obs = observed.to_numpy(bool)[o + h]
    keep = np.isfinite(tgt)
    return o[keep], h[keep], tgt[keep], obs[keep]
