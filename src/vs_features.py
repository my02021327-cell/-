"""HRT·체류시간 기반 부하 파생 피처.

설계 의도(제안 원안)
  · EMA30(평일)  = 달력 약 42일 → HRT 40.6일 대응. 지수감쇄라 과평탄화 회피.
  · Roll10(평일) = 달력 약 14일 → 메탄생성조 응답(t90 ≈ 10일) 대응.
  · Surge_Ratio  = Roll10 / Roll30. 최근 부하가 장기 평균 대비 높은지 낮은지.

본 저장소 데이터는 평일 격자가 아니라 **달력일 격자(2,190일)** 이므로
창 길이를 달력일로 환산해 의도를 보존한다(10→14, 30→42).
Surge_Ratio 는 비율이라 계측 드리프트·단위 변화에 둔감하다 — 이 데이터의
최대 약점이 소화조 계측 단절이므로 그 성질이 중요하다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 평일 창 → 달력일 창 환산 (주 5일 → 7/5 배)
W_SHORT, W_LONG = 14, 42          # 원안 10일 / 30일
EMA_SPAN = 42                     # 원안 span=30(평일)
EPS = 1e-6


def add_load_features(df: pd.DataFrame, col: str, prefix: str | None = None,
                      fill: bool = True) -> pd.DataFrame:
    """부하 계열 하나에 EMA·Surge 파생을 붙인다. 인덱스는 날짜 오름차순 가정."""
    d = df.sort_index().copy()
    p = prefix or col
    s = d[col].astype(float)
    if fill:
        s = s.ffill()                       # 계측 결측일은 최신값 유지(미래 정보 미사용)
    d[f"{p}_EMA"] = s.ewm(span=EMA_SPAN, adjust=False).mean()
    r_short = s.rolling(W_SHORT, min_periods=1).mean()
    r_long = s.rolling(W_LONG, min_periods=1).mean()
    d[f"{p}_SURGE"] = r_short / (r_long + EPS)
    return d


def build(df: pd.DataFrame) -> pd.DataFrame:
    """VS 부하 + COD 부하 + 유입 성상에 파생을 붙인다.

    COD 부하를 함께 만드는 이유: 유입 pH 4.5~5.1 에서 건조 중 VFA 가 휘발해
    유입_VS 가 과소 측정된다(COD/VS 2.1~3.6 이 그 지문). COD 기준 부하가
    같은 물질을 덜 편향되게 잰다.
    """
    d = df.sort_index().copy()
    d["COD_in"] = d["투입량합계"] * d["유입_CODcr"] / 1000.0       # kg COD/d
    d["OLR"] = d["VS_in"] / 8000.0                                # kg VS/m3/d
    for c in ["VS_in", "COD_in"]:
        d = add_load_features(d, c)
    return d
