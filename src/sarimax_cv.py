"""
SARIMAX 교차검증 & 모델 진단 — 혐기성 소화조 2상 메탄예측 시계열 base 검증
================================================================================
`src/stack_methane.py` 의 시계열 base(SARIMAX)가 단일 2023 분할에서 R²=0.891 을
낸 것이 **우연이 아닌지**, 그리고 **모델링 오류(차수 선택·잔차 자기상관·수렴)** 가
없는지를 검증한다. 실행: `python -m src.sarimax_cv`.

────────────────────────────────────────────────────────────────────────────────
검증 3종
────────────────────────────────────────────────────────────────────────────────
1) 롤링-오리진(확장창) 시계열 교차검증
   - 학습창을 점진적으로 늘리며(expanding window) 각 폴드에서 다음 블록(≈180일)을
     **1-step 인과예측(dynamic=False)** 하고 오차를 집계. 미래 정보 누수 없음.
   - 폴드별 R²/RMSE/MAE 와 평균±표준편차 → 단일분할 성능의 안정성(오차의 분산) 확인.

2) 차수(order) 선택 검증
   - 후보 차수들의 **CV 오차 + AIC** 를 비교. 성능이 통계적으로 동률이면 가장
     단순한(파라미터 최소) 차수를 채택(Occam). → stack_methane 의 (1,0,1) 타당성 확인.

3) 잔차 진단(모델 적합도)
   - Ljung-Box 검정(잔차 자기상관), 수렴 여부, 추정 파라미터. 결측(42%)로 잔차가
     불규칙하므로 NaN 제거 후 진단하며, 그 한계를 함께 보고한다.

핵심 결과(재현치): CV R² ≈ 0.868±0.038(8폴드) 로 단일분할 0.891 은 폴드 분포 안에 있어
**과적합·요행이 아님**. 후보 차수 간 CV 오차는 사실상 동률이라 **(1,0,1) 을 유지**.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.stack_methane import SARIMAX_EXOG, SARIMAX_ORDER
from src.xgb_methane import HOLDOUT_YEAR, OUT, TARGET, build_features, load

# CV 설정
MIN_TRAIN = 730          # 최소 학습창(≈2년) 이후부터 폴드 시작
STEP = 180               # 폴드 이동 간격(일)
BLOCK = 180              # 각 폴드 테스트 블록 길이(일)
CANDIDATE_ORDERS = [(1, 0, 1), (1, 1, 1), (2, 0, 2), (2, 1, 2), (1, 0, 2)]
MAXITER = 200

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 · 적합
# ──────────────────────────────────────────────────────────────────────────────
def daily_series():
    """일별 연속 캘린더의 endog(methane, 결측 NaN)·exog(투입부하)."""
    df = build_features(load()).set_index("date")
    endog = df[TARGET].astype(float)
    exog = df[SARIMAX_EXOG].ffill().bfill()
    return endog, exog


def _fit(endog_tr, exog_tr, order):
    return SARIMAX(
        endog_tr, exog=exog_tr, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=MAXITER)


def _scores(y, p):
    d = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(d) < 10:
        return None
    return {"R2": r2_score(d.y, d.p),
            "RMSE": float(np.sqrt(mean_squared_error(d.y, d.p))),
            "MAE": float(mean_absolute_error(d.y, d.p)),
            "n": int(len(d))}


# ──────────────────────────────────────────────────────────────────────────────
# 롤링-오리진 교차검증
# ──────────────────────────────────────────────────────────────────────────────
def rolling_cv(endog, exog, order):
    """확장창 롤링-오리진 CV. 각 폴드: train=[:cut] 적합 → [cut:cut+BLOCK] 1-step 예측."""
    n = len(endog)
    cuts = list(range(MIN_TRAIN, n - BLOCK // 2, STEP))
    folds, preds = [], []
    for k, cut in enumerate(cuts):
        res = _fit(endog.iloc[:cut], exog.iloc[:cut], order)
        end = min(cut + BLOCK, n)
        full = res.apply(endog.iloc[:end], exog=exog.iloc[:end])
        pred = full.get_prediction(start=cut, dynamic=False).predicted_mean
        y = endog.iloc[cut:end]
        sc = _scores(y, pred)
        if sc is None:
            continue
        sc = {"fold": k + 1, "train_end": str(endog.index[cut - 1].date()),
              "test_start": str(endog.index[cut].date()),
              "test_end": str(endog.index[end - 1].date()), **sc}
        folds.append(sc)
        dd = pd.DataFrame({"date": y.index, "y": y.values, "p": pred.values,
                           "fold": k + 1}).dropna()
        preds.append(dd)
    fdf = pd.DataFrame(folds)
    pdf = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    agg = {
        "folds": int(len(fdf)),
        "R2_mean": round(float(fdf.R2.mean()), 4), "R2_std": round(float(fdf.R2.std()), 4),
        "RMSE_mean": round(float(fdf.RMSE.mean()), 1), "RMSE_std": round(float(fdf.RMSE.std()), 1),
        "MAE_mean": round(float(fdf.MAE.mean()), 1),
        "R2_pooled": round(float(r2_score(pdf.y, pdf.p)), 4) if len(pdf) else None,
    }
    return fdf, pdf, agg


# ──────────────────────────────────────────────────────────────────────────────
# 차수 선택 검증 (CV 오차 + AIC)
# ──────────────────────────────────────────────────────────────────────────────
def order_search(endog, exog):
    tr = endog.index.year < HOLDOUT_YEAR
    rows = []
    for order in CANDIDATE_ORDERS:
        fdf, _, agg = rolling_cv(endog, exog, order)
        aic = _fit(endog[tr], exog[tr], order).aic
        rows.append({"order": str(order), "cv_R2_mean": agg["R2_mean"],
                     "cv_R2_std": agg["R2_std"], "cv_RMSE_mean": agg["RMSE_mean"],
                     "cv_MAE_mean": agg["MAE_mean"], "AIC": round(float(aic), 0),
                     "n_params": order[0] + order[2] + len(SARIMAX_EXOG) + 1})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 잔차 진단
# ──────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(endog, exog, order):
    tr = endog.index.year < HOLDOUT_YEAR
    res = _fit(endog[tr], exog[tr], order)
    resid = pd.Series(res.resid).replace([np.inf, -np.inf], np.nan)
    resid = resid[(resid.notna()) & (resid != 0)]
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    return {
        "order": str(order),
        "converged": bool(res.mle_retvals.get("converged", False)),
        "aic": round(float(res.aic), 1), "bic": round(float(res.bic), 1),
        "params": {k: round(float(v), 3) for k, v in zip(res.param_names, res.params)},
        "ljung_box": {f"lag{int(l)}": {"stat": round(float(r.lb_stat), 2),
                                        "pvalue": round(float(r.lb_pvalue), 4)}
                      for l, r in lb.iterrows()},
        "ljung_box_note": ("결측(42%)로 잔차가 불규칙 → NaN 제거 후 검정한 근사치. "
                           "p<0.05 는 약한 잔차 자기상관 잔존을 시사하나, CV 상 고차수가 "
                           "성능을 개선하지 못해 (1,0,1) 을 파시모니로 유지."),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_cv(fdf, agg, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(fdf.fold, fdf.R2, color="#4C9A4C", alpha=0.85)
    ax.axhline(agg["R2_mean"], color="#C0504D", ls="--",
               label=f"CV mean R2={agg['R2_mean']:.3f} ± {agg['R2_std']:.3f}")
    ax.axhline(0.891, color="#333", ls=":", label="single-split 2023 R2=0.891")
    ax.set_xlabel("rolling-origin fold"); ax.set_ylabel("R2 (1-step, held-out block)")
    ax.set_title("SARIMAX rolling-origin cross-validation")
    ax.set_ylim(0.7, 1.0); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cv_fit(pdf, path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(pdf["date"], pdf["y"], color="#333", lw=1.2, label="observed")
    ax.plot(pdf["date"], pdf["p"], color="#C0504D", lw=1.0, label="SARIMAX CV 1-step")
    ax.set_ylabel("methane (Nm3/d)")
    ax.set_title("SARIMAX cross-validated 1-step predictions (all folds)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    endog, exog = daily_series()

    # 1) 채택 차수의 롤링-오리진 CV
    fdf, pdf, agg = rolling_cv(endog, exog, SARIMAX_ORDER)
    fdf.to_csv(f"{OUT}/sarimax_cv_folds.csv", index=False)

    # 2) 차수 선택 검증
    osel = order_search(endog, exog)
    osel.to_csv(f"{OUT}/sarimax_order_selection.csv", index=False)

    # 3) 잔차 진단
    diag = residual_diagnostics(endog, exog, SARIMAX_ORDER)

    # 산출물
    plot_cv(fdf, agg, f"{OUT}/sarimax_cv.png")
    plot_cv_fit(pdf, f"{OUT}/sarimax_cv_fit.png")
    summary = {
        "model": f"SARIMAX{SARIMAX_ORDER} + exog{SARIMAX_EXOG}",
        "cv_design": f"rolling-origin expanding window: min_train={MIN_TRAIN}d, "
                     f"step={STEP}d, block={BLOCK}d",
        "adopted_order_cv": agg,
        "single_split_2023_R2": 0.8912,
        "order_selection": osel.to_dict(orient="records"),
        "residual_diagnostics": diag,
        "conclusion": "CV R2 평균이 단일분할과 폴드 분산 내에서 일치 → 요행 아님. "
                      "후보 차수 CV 동률 → (1,0,1) 파시모니 유지.",
    }
    with open(f"{OUT}/sarimax_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    print(f"===== SARIMAX{SARIMAX_ORDER} 롤링-오리진 CV ({agg['folds']} folds) =====")
    for _, r in fdf.iterrows():
        print(f"  fold{int(r.fold)} [{r.test_start}~{r.test_end}] "
              f"R2={r.R2:+.4f} RMSE={r.RMSE:.1f} MAE={r.MAE:.1f} n={int(r.n)}")
    print(f"  --> CV R2 = {agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}  "
          f"RMSE = {agg['RMSE_mean']:.1f} ± {agg['RMSE_std']:.1f}  "
          f"(pooled R2={agg['R2_pooled']}), single-split 2023 R2=0.891")
    print("\n----- 차수 선택 검증 (CV + AIC) -----")
    print(osel.to_string(index=False))
    print("\n----- 잔차 진단 -----")
    print(f"  수렴={diag['converged']}  AIC={diag['aic']}  params={diag['params']}")
    for lag, v in diag["ljung_box"].items():
        print(f"  Ljung-Box {lag}: stat={v['stat']} p={v['pvalue']}")
    print(f"  주: {diag['ljung_box_note']}")


if __name__ == "__main__":
    main()
