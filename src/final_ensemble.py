"""
최종 앙상블 모델 — 혐기성 소화조 2상 메탄생성량 예측 (검증된 Lag 반영)
================================================================================
`src/lag_validation.py` 의 **재검증된 지연(lag=0일)** 을 반영해 base 를 재구성하고,
스태킹 앙상블을 확정한 뒤 **롤링-오리진 교차검증으로 신뢰도를 재산출** 한다.
실행: `python -m src.final_ensemble`.

────────────────────────────────────────────────────────────────────────────────
확정 설계
────────────────────────────────────────────────────────────────────────────────
[Lag]   투입량합계 → 메탄 지연 = **0일**
        · AR(7) 프리화이트닝 교차상관(영업일 격자) r=0.500, p=0.007(>null95=0.187)
        · 부트스트랩 peak lag [0,0], ±3일 집중도 **100%**, AR 차수·연도(2019~2023) 안정
        · 기존 XGBoost gain 선택(2일)은 상관된 후보 중 임의선택이라 기각
        · 투입량은 **당일 결정되는 제어입력** 이므로 lag0 사용은 정보누수가 아님

[Base]  ① SARIMAX(1,0,1) + 외생[투입량합계(lag0), load10]  ← **CV 로 외생조합 선택**
        ② XGBoost(8-시드 평균) — 운전변수 전용
        ③ RandomForest        — 운전변수 전용
[Meta]  비음수 Ridge(positive=True), 학습기간 **OOF**(RF·XGB: TimeSeriesSplit,
        SARIMAX: 1-step 인과예측) → 누수 없음
[제약]  **persistence(어제 메탄값)는 예측 입력에 쓰지 않는다.** 비교 baseline 전용.

────────────────────────────────────────────────────────────────────────────────
신뢰도 산출 방식
────────────────────────────────────────────────────────────────────────────────
단일 홀드아웃(2023) 성능은 우연일 수 있으므로, **앙상블 전체 파이프라인**(base 학습 →
OOF → 메타 학습 → 예측)을 **롤링-오리진 확장창 CV** 로 폴드마다 처음부터 재학습해
평가한다. 최종 신뢰도는 'CV 평균±표준편차'와 '2023 홀드아웃'을 함께 보고한다.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.xgb_methane import HOLDOUT_YEAR, OUT, TARGET, build_features, load

warnings.filterwarnings("ignore")

# ── 확정 설정 ────────────────────────────────────────────────────────────────
VALIDATED_LAG = 0                                   # lag_validation.py 재검증 결과
FEED = "투입량합계"
FEED0 = "feed0"                                     # = 투입량합계 (lag 0)
TREE_FEATS = [FEED0, f"{FEED}_lag1", f"{FEED}_lag2", "load5", "load10"]
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_EXOG = [FEED0, "load10"]                    # CV 로 선택된 외생조합
SEEDS = 8
N_SPLITS = 5          # 최종 모델 메타 OOF
N_SPLITS_CV = 3       # CV 폴드 내부 메타 OOF(속도)
MIN_TRAIN, STEP, BLOCK = 730, 180, 180              # 롤링-오리진 CV

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.9, reg_lambda=4.0, reg_alpha=0.5, min_child_weight=6)
RF_PARAMS = dict(n_estimators=600, max_depth=7, min_samples_leaf=6, max_features=0.7,
                 random_state=42, n_jobs=4)
BASE_NAMES = ["XGBoost", "RandomForest", "SARIMAX"]


# ──────────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────────
def prepare() -> pd.DataFrame:
    """일별 연속 캘린더 + 검증 lag 기반 피처."""
    df = build_features(load())
    df[FEED0] = df[FEED]                                  # lag 0 (당일 제어입력)
    df[f"{FEED}_lag1"] = df[FEED].shift(1)
    df[f"{FEED}_lag2"] = df[FEED].shift(2)
    return df.set_index("date")


def metrics(y, p) -> dict:
    return {"R2": round(float(r2_score(y, p)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 1),
            "MAE": round(float(mean_absolute_error(y, p)), 1),
            "n": int(len(y))}


# ──────────────────────────────────────────────────────────────────────────────
# Base learners
# ──────────────────────────────────────────────────────────────────────────────
def xgb_pred(Xtr, ytr, Xte, seeds=SEEDS):
    return np.mean([XGBRegressor(random_state=s, n_jobs=4, **XGB_PARAMS)
                    .fit(Xtr, ytr).predict(Xte) for s in range(seeds)], axis=0)


def rf_pred(Xtr, ytr, Xte):
    return RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr).predict(Xte)


def sarimax_causal(df: pd.DataFrame, cut: int) -> pd.Series:
    """[:cut] 로 파라미터 적합 후 전체구간 1-step 인과예측(각 시점은 과거만 사용)."""
    endog = df[TARGET].astype(float)
    exog = df[SARIMAX_EXOG].ffill().bfill()
    res = SARIMAX(endog.iloc[:cut], exog=exog.iloc[:cut], order=SARIMAX_ORDER,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, maxiter=300)
    full = res.apply(endog, exog=exog)
    return full.get_prediction(start=endog.index[1], dynamic=False).predicted_mean


# ──────────────────────────────────────────────────────────────────────────────
# 스태킹 (한 폴드 = base 학습 → OOF → 메타 학습 → 테스트 예측)
# ──────────────────────────────────────────────────────────────────────────────
def run_split(df: pd.DataFrame, cut: int, end: int, n_splits: int, seeds: int):
    """cut 이전=학습, [cut:end]=테스트. 앙상블 전 과정을 폴드 내에서 재수행."""
    sar_all = sarimax_causal(df, cut)

    lab = df.dropna(subset=[TARGET] + TREE_FEATS)
    dates_tr = df.index[:cut]
    dates_te = df.index[cut:end]
    tr = lab[lab.index.isin(dates_tr)]
    te = lab[lab.index.isin(dates_te)]
    if len(te) < 10 or len(tr) < 200:
        return None

    Xtr, ytr = tr[TREE_FEATS].values, tr[TARGET].values
    Xte, yte = te[TREE_FEATS].values, te[TARGET].values
    sar_tr = sar_all.reindex(tr.index).values
    sar_te = sar_all.reindex(te.index).values

    # base (전체 학습데이터로 학습) → 테스트 예측
    p_x, p_r = xgb_pred(Xtr, ytr, Xte, seeds), rf_pred(Xtr, ytr, Xte)

    # 메타 학습용 OOF (학습기간 내부, 누수 없음)
    ox = np.full(len(ytr), np.nan); orf = np.full(len(ytr), np.nan)
    for tri, vai in TimeSeriesSplit(n_splits=n_splits).split(Xtr):
        ox[vai] = xgb_pred(Xtr[tri], ytr[tri], Xtr[vai], max(seeds // 2, 2))
        orf[vai] = rf_pred(Xtr[tri], ytr[tri], Xtr[vai])
    m = ~np.isnan(ox) & ~np.isnan(sar_tr)
    meta = Ridge(alpha=1.0, positive=True).fit(np.c_[ox[m], orf[m], sar_tr[m]], ytr[m])

    ok = ~np.isnan(sar_te)
    pred = meta.predict(np.c_[p_x, p_r, sar_te])
    return {
        "dates": te.index[ok], "y": yte[ok], "stack": pred[ok],
        "xgb": p_x[ok], "rf": p_r[ok], "sarimax": sar_te[ok],
        "persist": te["persist"].values[ok],          # baseline 비교 전용
        "weights": [float(c) for c in meta.coef_], "intercept": float(meta.intercept_),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 신뢰도 재산출 : 앙상블 전체 파이프라인 롤링-오리진 CV
# ──────────────────────────────────────────────────────────────────────────────
def rolling_cv(df: pd.DataFrame):
    n = len(df)
    folds, pool = [], []
    for k, cut in enumerate(range(MIN_TRAIN, n - BLOCK // 2, STEP)):
        r = run_split(df, cut, min(cut + BLOCK, n), N_SPLITS_CV, seeds=4)
        if r is None:
            continue
        sc = metrics(r["y"], r["stack"])
        folds.append({"fold": k + 1, "test_start": str(r["dates"][0].date()),
                      "test_end": str(r["dates"][-1].date()), **sc,
                      "persist_R2": round(float(r2_score(r["y"], r["persist"])), 4)})
        pool.append(pd.DataFrame({"y": r["y"], "p": r["stack"]}))
    fdf = pd.DataFrame(folds)
    pooled = pd.concat(pool, ignore_index=True)
    agg = {"folds": int(len(fdf)),
           "R2_mean": round(float(fdf.R2.mean()), 4), "R2_std": round(float(fdf.R2.std()), 4),
           "RMSE_mean": round(float(fdf.RMSE.mean()), 1), "RMSE_std": round(float(fdf.RMSE.std()), 1),
           "MAE_mean": round(float(fdf.MAE.mean()), 1),
           "R2_pooled": round(float(r2_score(pooled.y, pooled.p)), 4),
           "persist_R2_mean": round(float(fdf.persist_R2.mean()), 4)}
    return fdf, agg


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_final(res, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(res["dates"], res["y"], color="#333", lw=1.4, label="observed methane")
    ax.plot(res["dates"], res["stack"], color="#C0504D", lw=1.3,
            label=f"Final ensemble (R2={metrics(res['y'], res['stack'])['R2']})")
    ax.plot(res["dates"], res["persist"], color="#999", lw=0.9, ls="--",
            label=f"persistence baseline (R2={metrics(res['y'], res['persist'])['R2']})")
    ax.set_ylabel("methane (Nm3/d)")
    ax.set_title("2023 holdout - final stacking ensemble (validated lag=0)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cv(fdf, agg, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(fdf.fold, fdf.R2, color="#4C9A4C", alpha=0.85, label="ensemble R2")
    ax.plot(fdf.fold, fdf.persist_R2, "o--", color="#999", ms=5, label="persistence baseline")
    ax.axhline(agg["R2_mean"], color="#C0504D", ls="--",
               label=f"CV mean R2={agg['R2_mean']:.3f} ± {agg['R2_std']:.3f}")
    ax.set_xlabel("rolling-origin fold"); ax.set_ylabel("R2")
    ax.set_title("Final ensemble - rolling-origin cross-validated reliability")
    ax.set_ylim(0.6, 1.0); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    df = prepare()

    # 1) 최종 홀드아웃(2023) 평가
    cut = int((df.index.year < HOLDOUT_YEAR).sum())
    res = run_split(df, cut, len(df), N_SPLITS, SEEDS)
    y = res["y"]
    m = {"persistence_baseline": metrics(y, res["persist"]),
         "xgboost": metrics(y, res["xgb"]),
         "randomforest": metrics(y, res["rf"]),
         "sarimax": metrics(y, res["sarimax"]),
         "final_ensemble": metrics(y, res["stack"])}

    # 2) 신뢰도 재산출 : 앙상블 전체 파이프라인 롤링-오리진 CV
    fdf, agg = rolling_cv(df)
    fdf.to_csv(f"{OUT}/final_ensemble_cv_folds.csv", index=False)

    plot_final(res, f"{OUT}/final_ensemble_2023.png")
    plot_cv(fdf, agg, f"{OUT}/final_ensemble_cv.png")
    pd.DataFrame({"date": res["dates"], "methane_true": y, "final_ensemble": res["stack"],
                  "sarimax": res["sarimax"], "xgboost": res["xgb"], "randomforest": res["rf"],
                  "persistence_baseline": res["persist"]}
                 ).to_csv(f"{OUT}/final_ensemble_2023.csv", index=False)

    summary = {
        "model": "Stacking(XGBoost, RandomForest, SARIMAX) with non-negative Ridge meta",
        "validated_lag_day": VALIDATED_LAG,
        "lag_evidence": "AR(7) prewhitened CCF (business-day grid): r=0.500, p=0.007, "
                        "bootstrap peak [0,0], within-3d 100%, stable across AR orders & 2019-2023",
        "previous_xgb_gain_lag": 2,
        "sarimax": {"order": list(SARIMAX_ORDER), "exog": SARIMAX_EXOG,
                    "exog_selected_by": "rolling-origin CV (not the holdout)"},
        "tree_features": TREE_FEATS,
        "persistence_policy": "예측 입력으로 사용 안 함(비교 baseline 전용)",
        "holdout_2023": m,
        "meta_weights": dict(zip(BASE_NAMES, [round(w, 4) for w in res["weights"]]),
                             intercept=round(res["intercept"], 1)),
        "reliability_rolling_cv": agg,
        "cv_folds": fdf.to_dict(orient="records"),
    }
    with open(f"{OUT}/final_ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=========== 최종 앙상블 (검증 lag=0일 반영) ===========")
    print(f"SARIMAX{SARIMAX_ORDER} exog={SARIMAX_EXOG} (CV 선택) | tree feats={TREE_FEATS}")
    print(f"\n[2023 홀드아웃]  n={m['final_ensemble']['n']}")
    for k in ["persistence_baseline", "randomforest", "xgboost", "sarimax", "final_ensemble"]:
        r = m[k]
        print(f"  {k:22s} R2={r['R2']:+.4f}  RMSE={r['RMSE']:7.1f}  MAE={r['MAE']:7.1f}")
    print(f"  meta weights → " + "  ".join(f"{n}={w:.3f}" for n, w in zip(BASE_NAMES, res["weights"])))
    print(f"\n[신뢰도 재산출 — 앙상블 전체 파이프라인 롤링-오리진 CV, {agg['folds']}폴드]")
    for _, r in fdf.iterrows():
        print(f"  fold{int(r.fold)} [{r.test_start}~{r.test_end}] R2={r.R2:+.4f} "
              f"RMSE={r.RMSE:7.1f} (persist {r.persist_R2:+.4f})")
    print(f"  --> CV R2 = {agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}  "
          f"(pooled {agg['R2_pooled']}), CV RMSE = {agg['RMSE_mean']:.1f} ± {agg['RMSE_std']:.1f}")
    print(f"  --> persistence baseline CV R2 = {agg['persist_R2_mean']:.4f}")
    print("=====================================================")


if __name__ == "__main__":
    main()
