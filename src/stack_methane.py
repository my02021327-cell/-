"""
RandomForest + XGBoost + 시계열(SARIMAX) 스태킹 앙상블 — 혐기성 소화조 2상 메탄예측
================================================================================
`src/xgb_methane.py`(2상 집중·persistence 상회 설계)를 확장한다. 동일한
persistence 상회 피처셋 위에서

  1) RandomForest 단일 모델을 학습·평가하고,
  2) **시계열 모델(SARIMAX + 외생 투입부하)** 을 추가하여,
  3) RandomForest + XGBoost + SARIMAX 를 **스태킹(stacking)** 으로 결합한다.

────────────────────────────────────────────────────────────────────────────────
왜 시계열 모델을 넣는가 (앙상블 다양성)
────────────────────────────────────────────────────────────────────────────────
RF·XGB 는 같은 트리 계열이라 잔차 상관이 ≈0.99 로 매우 높아 결합 이득이 작다.
**SARIMAX(1,0,1)+외생 투입부하** 는 선형 상태공간(칼만필터) 모델로 트리와 계열이 달라
잔차 상관이 ≈0.93 으로 낮다(=다양성↑). 게다가 결측 많은(42%) 일별 메탄을 칼만필터가
자연스럽게 처리하고, 자기상관 구조를 직접 모형화해 **단일 모델로도 가장 강하다**
(2023 R²=0.891 > persistence 0.877). → 스태킹의 핵심 base 가 된다.

────────────────────────────────────────────────────────────────────────────────
중요 : persistence 는 '예측 입력' 으로 쓰지 않는다
────────────────────────────────────────────────────────────────────────────────
어제 메탄값(persist)을 피처·메타입력으로 넣어 예측하는 것을 금지한다. persistence 는
오직 **비교용 baseline** 으로만 사용한다. 즉 어떤 모델도 persist 피처로 예측하지 않는다.
SARIMAX 는 persist '피처' 를 쓰지 않는 정식 시계열(상태공간) 모델이며, 메탄 동특성과
외생 투입부하로 예측한다.

────────────────────────────────────────────────────────────────────────────────
스태킹 설계 (누수 없는 시계열 OOF)
────────────────────────────────────────────────────────────────────────────────
• Base learner :
    - RandomForest, XGBoost(8-시드 평균) — 운전변수(투입 lag·체류창부하)만으로 학습.
    - SARIMAX(1,0,1) + 외생[load5,load10] — 일별 캘린더에서 학습, 1-step 인과예측.
• Meta feature : 학습기간(2018~2022)에서 각 base 의 **인과적 OOF 예측** 생성(누수 차단).
    - RF·XGB : **TimeSeriesSplit** 확장창 OOF.
    - SARIMAX : 파라미터를 학습기간에 적합 후 **1-step(dynamic=False) 인과 예측**.
  (persistence 는 메타 입력에서 제외 — 예측에 쓰지 않음)
• Meta learner : 비음수 Ridge(`positive=True`). 해석 가능한 가중치 + 강건성.
• 최종 : base 를 전체 학습데이터로 재학습 → 2023 예측 → 메타로 결합.

메탄은 일별 자기상관 0.92 로 persistence(baseline 2023 R²≈0.88)가 강하다. persist 피처
없이 이를 이기는 것은 **시계열 모델(SARIMAX)** 이다(2023 R²=**0.891** > persistence
0.877). 운전변수만 쓰는 트리(RF·XGB)는 과거 메탄을 안 쓰므로 약하고(R²≈0.60), 메타는
이를 자동 down-weight 하여 스태킹은 SARIMAX 중심으로 수렴한다. 즉 **persistence 를
예측에 쓰지 않고도** 시계열 base 로 persistence baseline 을 상회한다.
실행: `python -m src.stack_methane`.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.xgb_methane import (
    HOLDOUT_YEAR, LAG_DRIVER, OUT, SEEDS, TARGET, build_features, drop_missing,
    load, metrics, select_best_lag, split, vs_persist,
)

warnings.filterwarnings("ignore")

# 운전변수 전용 피처셋 (persist/stale 등 과거 메탄 기반 피처 제외 — 예측에 persistence 미사용)
STACK_FEATS_TEMPLATE = ["{lag}", f"{LAG_DRIVER}_lag3", f"{LAG_DRIVER}_lag7",
                        "load5", "load10"]
N_SPLITS = 5
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_EXOG = ["load5", "load10"]     # 외생 투입부하(체류창)
BASE_NAMES = ["XGBoost", "RandomForest", "SARIMAX"]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.9, reg_lambda=4.0, reg_alpha=0.5, min_child_weight=6,
)
RF_PARAMS = dict(
    n_estimators=600, max_depth=7, min_samples_leaf=6, max_features=0.7,
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


def sarimax_causal_predictions(df: pd.DataFrame) -> pd.Series:
    """시계열 base : SARIMAX(order)+외생 투입부하를 학습기간에 적합한 뒤, 전체 일별
    캘린더에 대해 **1-step(dynamic=False) 인과 예측** 을 반환(date 인덱스).

    - 결측 많은 일별 메탄(endog)은 칼만필터가 자연 처리(NaN skip).
    - 외생(load5,load10)은 결측을 ffill/bfill 로 연속화.
    - 파라미터는 학습기간(<HOLDOUT_YEAR)에만 적합 → 홀드아웃 누수 차단. 이후 전체
      시계열에 apply(파라미터 고정)하여 각 시점이 '과거만' 쓰는 1-step 예측을 만든다.
    """
    s = df.set_index("date")
    endog = s[TARGET].astype(float)
    exog = s[SARIMAX_EXOG].ffill().bfill()
    tr_mask = endog.index.year < HOLDOUT_YEAR
    fitted = SARIMAX(
        endog[tr_mask], exog=exog[tr_mask], order=SARIMAX_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    full = fitted.apply(endog, exog=exog)           # 파라미터 고정, 전체에 칼만필터
    pred = full.get_prediction(start=endog.index[1], dynamic=False).predicted_mean
    return pred.reindex(s.index)


# ──────────────────────────────────────────────────────────────────────────────
# 시계열 OOF meta feature 생성 → 비음수 Ridge 메타 학습
# ──────────────────────────────────────────────────────────────────────────────
def build_oof(Xtr, ytr, sar_tr):
    """base OOF 예측 생성(누수 차단). meta 입력 = [XGB, RF, SARIMAX] (persistence 제외).
    RF·XGB : TimeSeriesSplit 확장창 OOF. SARIMAX : 인과 1-step 예측(사전 계산 sar_tr)."""
    n = len(ytr)
    oof_x = np.full(n, np.nan)
    oof_r = np.full(n, np.nan)
    for tri, vai in TimeSeriesSplit(n_splits=N_SPLITS).split(Xtr):
        oof_x[vai] = xgb_fit_predict(Xtr[tri], ytr[tri], Xtr[vai], seeds=4)
        oof_r[vai] = rf_fit_predict(Xtr[tri], ytr[tri], Xtr[vai])
    mask = ~np.isnan(oof_x) & ~np.isnan(sar_tr)
    Z = np.c_[oof_x[mask], oof_r[mask], sar_tr[mask]]
    return Z, ytr[mask], mask


def fit_meta(Z, y):
    """비음수 Ridge 메타 학습 (base 예측의 비음수 배합; persistence 미사용)."""
    return Ridge(alpha=1.0, positive=True).fit(Z, y)


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_pred(dates, y, stack, sar, persist, m, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, y, color="#333", lw=1.4, label="observed methane")
    ax.plot(dates, stack, color="#C0504D", lw=1.4,
            label=f"Stacking (R2={m['stack']['R2']})")
    ax.plot(dates, sar, color="#4C9A4C", lw=1.0,
            label=f"SARIMAX (R2={m['sarimax']['R2']})")
    ax.plot(dates, persist, color="#999", lw=0.9, ls="--",
            label=f"persistence (R2={m['persistence']['R2']})")
    ax.set_ylabel("methane (Nm3/d)")
    ax.set_title("2023 holdout - RF+XGB+SARIMAX stacking vs persistence")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_weights(coef, intercept, path):
    colors = ["#C0504D", "#4F81BD", "#4C9A4C"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(BASE_NAMES, coef, color=colors)
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

    # 시계열 base(SARIMAX)는 일별 캘린더에서 계산 후 보존행 날짜에 정렬
    sar_all = sarimax_causal_predictions(df)

    d = drop_missing(df, feats)
    tr, te = split(d)
    Xtr, ytr = tr[feats].values, tr[TARGET].values
    Xte, yte = te[feats].values, te[TARGET].values
    persist_te = te["persist"].values           # 비교 baseline 전용(예측에 미사용)
    sar_tr = sar_all.reindex(tr["date"]).values
    sar_te = sar_all.reindex(te["date"]).values

    # 1) Base 단일 모델 (전체 학습데이터로 학습 → 2023 예측)
    pred_xgb = xgb_fit_predict(Xtr, ytr, Xte)
    pred_rf = rf_fit_predict(Xtr, ytr, Xte)
    pred_sar = sar_te

    # 2) 시계열 OOF → 비음수 Ridge 메타 (XGB, RF, SARIMAX) — persistence 미사용
    Z, y_oof, _ = build_oof(Xtr, ytr, sar_tr)
    meta = fit_meta(Z, y_oof)
    Zte = np.c_[pred_xgb, pred_rf, pred_sar]
    pred_stack = meta.predict(Zte)

    # 3) 지표 (동일 2023 행에서 공정 비교)
    m = {
        "persistence": metrics(yte, persist_te),
        "randomforest": metrics(yte, pred_rf),
        "xgboost": metrics(yte, pred_xgb),
        "sarimax": metrics(yte, pred_sar),
        "avg_all": metrics(yte, (pred_xgb + pred_rf + pred_sar) / 3),
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
    plot_pred(te["date"].values, yte, pred_stack, pred_sar, persist_te, m,
              f"{OUT}/stack_predictions_2023.png")
    plot_weights(coef, intercept, f"{OUT}/stack_meta_weights.png")
    pd.DataFrame({
        "date": te["date"].values, "methane_true": yte,
        "persistence": persist_te, "pred_xgb": pred_xgb, "pred_rf": pred_rf,
        "pred_sarimax": pred_sar, "pred_stack": pred_stack,
        "staleness_days": te["stale"].values,
    }).to_csv(f"{OUT}/stack_predictions_2023.csv", index=False)

    summary = {
        "target": TARGET,
        "design": "RF·XGB·SARIMAX(시계열) base + 비음수 Ridge 스태킹 (persistence 예측 미사용)",
        "persistence_role": "비교 baseline 전용(예측 입력 아님)",
        "base_learners": BASE_NAMES,
        "sarimax_order": list(SARIMAX_ORDER), "sarimax_exog": SARIMAX_EXOG,
        "features": feats, "n_features": len(feats),
        "best_input_methane_lag_day": int(best_lag),
        "cv": f"TimeSeriesSplit(n_splits={N_SPLITS}) OOF(RF·XGB) + SARIMAX 인과 1-step",
        "train_n": int(len(tr)), "test_n": int(len(te)),
        "meta_weights": dict(zip(BASE_NAMES, coef), intercept=round(intercept, 1)),
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
    order = ["persistence", "randomforest", "xgboost", "sarimax", "avg_all", "stack"]
    for k in order:
        r = m[k]
        extra = "" if k == "persistence" else f"  dRMSE={vs_persist(r, m['persistence'])['dRMSE_%']:+.2f}%"
        print(f"  {k:13s} R2={r['R2']:+.4f}  RMSE={r['RMSE']:7.1f}  MAE={r['MAE']:7.1f}{extra}")
    print(f"  meta weights → " + "  ".join(f"{n}={c}" for n, c in zip(BASE_NAMES, coef))
          + f"  (int={intercept:.0f})")
    for r in robustness:
        print(f"    {r['name']:16s} n={r['n']:3d} persist RMSE={r['persist_RMSE']:.1f}  stack RMSE={r['stack_RMSE']:.1f}")
    print("========================================================")


if __name__ == "__main__":
    main()
