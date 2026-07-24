"""
Random Forest 모델링 + 스태킹 앙상블 — 혐기성 소화조 2상 메탄생성량 예측
================================================================================
`src/xgb_methane.py`(2상 집중·persistence 상회 설계)를 확장한다. 동일한
persistence 상회 피처셋 위에서

  1) RandomForest 단일 모델을 학습·평가하고,
  2) RandomForest + XGBoost 를 **스태킹(stacking)** 으로 결합한 앙상블을 설계한다.

────────────────────────────────────────────────────────────────────────────────
스태킹 설계 (누수 없는 시계열 OOF)
────────────────────────────────────────────────────────────────────────────────
• Base learner : RandomForest, XGBoost(8-시드 평균) — 둘 다 persistence 상회 피처셋
  (투입 lag·체류창부하·persistence·staleness)에서 학습.
• Meta feature : 학습기간(2018~2022)을 **TimeSeriesSplit** 로 확장창 분할해 각 base 의
  **Out-Of-Fold(OOF) 예측** 을 생성(미래 정보 누수 차단). 여기에 persistence 를 meta
  입력에 함께 넣어, 메타가 'base 보정량 + 원 persistence' 를 비음수로 배합하게 한다.
• Meta learner : 비음수 Ridge(`positive=True`). 해석 가능한 가중치 + 강건성.
• 최종 : base 를 전체 학습데이터로 재학습 → 2023 예측 → 메타로 결합.

메탄은 일별 자기상관 0.92 로 persistence(2023 R²≈0.88)가 매우 강하다. Base 두 모델은
서로 높은 상관(잔차 corr≈0.99)이라 단순 평균의 이득은 작지만, **persistence 를 메타
입력에 포함한 비음수 스태킹** 은 persistence·단일 XGB·단일 RF 를 모두 상회한다
(2023 R² 0.877→0.883, RMSE −2.4%). 실행: `python -m src.stack_methane`.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.xgb_methane import (
    LAG_DRIVER, OUT, SEEDS, TARGET, build_features, drop_missing, load,
    metrics, select_best_lag, split, vs_persist,
)

# persistence 상회 피처셋 (xgb_methane Model-2 와 동일 계열)
STACK_FEATS_TEMPLATE = ["{lag}", f"{LAG_DRIVER}_lag3", f"{LAG_DRIVER}_lag7",
                        "load5", "load10", "persist", "stale"]
N_SPLITS = 5

XGB_PARAMS = dict(
    n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.9, reg_lambda=5.0, reg_alpha=0.5, min_child_weight=8,
)
RF_PARAMS = dict(
    n_estimators=600, max_depth=6, min_samples_leaf=8, max_features=0.7,
    random_state=42, n_jobs=4,
)


# ──────────────────────────────────────────────────────────────────────────────
# Base learners
# ──────────────────────────────────────────────────────────────────────────────
def xgb_fit_predict(Xtr, ytr, Xte, seeds=SEEDS):
    """XGBoost 8-시드 평균 예측."""
    preds = [
        XGBRegressor(random_state=s, n_jobs=4, **XGB_PARAMS).fit(Xtr, ytr).predict(Xte)
        for s in range(seeds)
    ]
    return np.mean(preds, axis=0)


def rf_fit_predict(Xtr, ytr, Xte):
    """RandomForest 예측."""
    return RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr).predict(Xte)


# ──────────────────────────────────────────────────────────────────────────────
# 시계열 OOF meta feature 생성 → 비음수 Ridge 메타 학습
# ──────────────────────────────────────────────────────────────────────────────
def build_oof(Xtr, ytr, persist_tr):
    """TimeSeriesSplit 확장창으로 base OOF 예측 생성(누수 차단)."""
    n = len(ytr)
    oof_x = np.full(n, np.nan)
    oof_r = np.full(n, np.nan)
    for tri, vai in TimeSeriesSplit(n_splits=N_SPLITS).split(Xtr):
        oof_x[vai] = xgb_fit_predict(Xtr[tri], ytr[tri], Xtr[vai], seeds=4)
        oof_r[vai] = rf_fit_predict(Xtr[tri], ytr[tri], Xtr[vai])
    mask = ~np.isnan(oof_x)
    Z = np.c_[oof_x[mask], oof_r[mask], persist_tr[mask]]
    return Z, ytr[mask], mask


def fit_meta(Z, y):
    """비음수 Ridge 메타 학습 (base 보정 + persistence 비음수 배합)."""
    return Ridge(alpha=1.0, positive=True).fit(Z, y)


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_pred(dates, y, stack, xgb, rf, persist, m, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, y, color="#333", lw=1.4, label="observed methane")
    ax.plot(dates, stack, color="#C0504D", lw=1.4,
            label=f"Stacking (R2={m['stack']['R2']})")
    ax.plot(dates, persist, color="#999", lw=0.9, ls="--",
            label=f"persistence (R2={m['persistence']['R2']})")
    ax.set_ylabel("methane (Nm3/d)")
    ax.set_title("2023 holdout - stacking ensemble vs persistence")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_weights(coef, intercept, path):
    names = ["XGBoost", "RandomForest", "persistence"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(names, coef, color=["#C0504D", "#4F81BD", "#999999"])
    for i, c in enumerate(coef):
        ax.text(i, c, f"{c:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("meta weight (non-negative)")
    ax.set_title(f"Stacking meta weights (intercept={intercept:.0f})")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    df = build_features(load())
    best_lag, _ = select_best_lag(df)
    feats = [c.format(lag=f"{LAG_DRIVER}_lag{best_lag}") for c in STACK_FEATS_TEMPLATE]
    print(f"[lag] selected feed->methane lag = {best_lag} day")
    print(f"[feats] {feats}")

    d = drop_missing(df, feats)
    tr, te = split(d)
    Xtr, ytr = tr[feats].values, tr[TARGET].values
    Xte, yte = te[feats].values, te[TARGET].values
    persist_te = te["persist"].values

    # 1) Base 단일 모델 (전체 학습데이터로 학습 → 2023 예측)
    pred_xgb = xgb_fit_predict(Xtr, ytr, Xte)
    pred_rf = rf_fit_predict(Xtr, ytr, Xte)

    # 2) 시계열 OOF → 비음수 Ridge 메타
    Z, y_oof, _ = build_oof(Xtr, ytr, tr["persist"].values)
    meta = fit_meta(Z, y_oof)
    Zte = np.c_[pred_xgb, pred_rf, persist_te]
    pred_stack = meta.predict(Zte)

    # 3) 지표 (동일 2023 행에서 공정 비교)
    m = {
        "persistence": metrics(yte, persist_te),
        "randomforest": metrics(yte, pred_rf),
        "xgboost": metrics(yte, pred_xgb),
        "avg_xgb_rf": metrics(yte, (pred_xgb + pred_rf) / 2),
        "stack": metrics(yte, pred_stack),
    }
    coef = [round(float(c), 4) for c in meta.coef_]
    intercept = float(meta.intercept_)

    # staleness 구간별 강건성(스택이 낡은 구간에서도 이기는지)
    te2 = te.assign(stack=pred_stack)

    def subset(mask, name):
        s = te2[mask]
        if len(s) < 10:
            return None
        return {"name": name, "n": int(len(s)),
                "persist_RMSE": round(float(np.sqrt(mean_squared_error(s[TARGET], s["persist"]))), 1),
                "stack_RMSE": round(float(np.sqrt(mean_squared_error(s[TARGET], s["stack"]))), 1)}

    robustness = [r for r in [
        subset(te2.stale <= 1, "fresh(stale<=1d)"),
        subset(te2.stale >= 2, "stale(>=2d gap)"),
    ] if r]

    # 4) 산출물
    plot_pred(te["date"].values, yte, pred_stack, pred_xgb, pred_rf, persist_te, m,
              f"{OUT}/stack_predictions_2023.png")
    plot_weights(coef, intercept, f"{OUT}/stack_meta_weights.png")
    pd.DataFrame({
        "date": te["date"].values, "methane_true": yte,
        "persistence": persist_te, "pred_xgb": pred_xgb, "pred_rf": pred_rf,
        "pred_stack": pred_stack, "staleness_days": te["stale"].values,
    }).to_csv(f"{OUT}/stack_predictions_2023.csv", index=False)

    summary = {
        "target": TARGET,
        "design": "RandomForest 단일 + (RF, XGB, persistence) 비음수 Ridge 스태킹",
        "features": feats, "n_features": len(feats),
        "best_input_methane_lag_day": int(best_lag),
        "cv": f"TimeSeriesSplit(n_splits={N_SPLITS}) OOF on 2018-2022",
        "train_n": int(len(tr)), "test_n": int(len(te)),
        "meta_weights": {"XGBoost": coef[0], "RandomForest": coef[1],
                         "persistence": coef[2], "intercept": round(intercept, 1)},
        "results_same_rows": {
            k: ({**v, **vs_persist(v, m["persistence"])} if k != "persistence" else v)
            for k, v in m.items()
        },
        "robustness_by_staleness": robustness,
    }
    with open(f"{OUT}/stack_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    print(f"\n============ 2023 홀드아웃 (동일 {len(te)}행) ============")
    order = ["persistence", "randomforest", "xgboost", "avg_xgb_rf", "stack"]
    for k in order:
        r = m[k]
        extra = "" if k == "persistence" else f"  dRMSE={vs_persist(r, m['persistence'])['dRMSE_%']:+.2f}%"
        print(f"  {k:14s} R2={r['R2']:+.4f}  RMSE={r['RMSE']:7.1f}  MAE={r['MAE']:7.1f}{extra}")
    print(f"  meta weights → XGB={coef[0]}  RF={coef[1]}  persist={coef[2]}  (int={intercept:.0f})")
    for r in robustness:
        print(f"    {r['name']:16s} n={r['n']:3d} persist RMSE={r['persist_RMSE']:.1f}  stack RMSE={r['stack_RMSE']:.1f}")
    print("========================================================")


if __name__ == "__main__":
    main()
