"""
자료 → 보간 → 피처 까지의 준비 단계를 한 곳에 모은다.

보간은 **학습 구간 안에서만** 방식을 고르고 적합한다(`fit_end`). 전 구간으로 한 번
고르면 시험 구간의 성질이 보간 방식 선택에 새어 들어온다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bgp import config as C
from src.bgp.data import ANGLES, build_frame, quality_report
from src.bgp.features import build_features
from src.bgp.impute import HybridImputer, flag_measurement_errors


def prepare(fit_end: str | pd.Timestamp | None = None, causal_only: bool = False,
            angles: tuple[str, ...] | None = None, verbose: bool = True):
    """
    반환 (df, X, groups, info)
      df   : 파생·보간까지 끝난 일별 프레임 (CH4_pct_filled, CH4_m3d_filled 포함)
      X    : 인과적 피처 행렬
      info : 품질 요약 + 보간 선택 결과
    """
    df = build_frame()
    pred_cols = [c for k, v in ANGLES.items() if k != "temporal"
                 for c in v if c in df.columns] + [C.FLOW, "dow", "month"]

    fit_end = pd.Timestamp(fit_end) if fit_end else df["date"].quantile(0.7)
    train_slice = df[df["date"] < fit_end]

    imp = HybridImputer(causal_only=causal_only).fit(train_slice, pred_cols, n_blocks=80)
    df["CH4_pct_filled"] = imp.transform(df)
    df["CH4_m3d_filled"] = df[C.TARGET].where(
        df[C.TARGET].notna(), df[C.FLOW] * df["CH4_pct_filled"] / 100.0)

    df["flag_outlier_iso"] = flag_measurement_errors(
        df, [C.FLOW, "feed_AB_tpd", "dig_pH_A", "VFA_A_mgL", "ALK_A_mgL",
             "acid_TS_pct", "dig_VS_A_pct"]).astype(int)

    X, groups = build_features(df, angles=angles, target_col="CH4_m3d_filled")

    info = {
        "quality": quality_report(df),
        "imputation": {
            "selected_method": imp.method_,
            "candidate_MAE_pp": imp.scores_,
            "fit_end": str(pd.Timestamp(fit_end).date()),
            "causal_only": causal_only,
            "labels_observed": int(df[C.TARGET].notna().sum()),
            "labels_after_fill": int(df["CH4_m3d_filled"].notna().sum()),
        },
        "features": {"n_total": int(X.shape[1]),
                     "by_angle": {k: len(v) for k, v in groups.items()}},
        "outliers_flagged": int(df["flag_outlier_iso"].sum()),
    }
    if verbose:
        print(f"보간 방식 선택: {imp.method_}  후보 MAE(%p)={imp.scores_}")
        print(f"라벨 {info['imputation']['labels_observed']} → 재구성 후 "
              f"{info['imputation']['labels_after_fill']}")
        print(f"피처 {X.shape[1]}개 {info['features']['by_angle']}")
    return df, X, groups, info
