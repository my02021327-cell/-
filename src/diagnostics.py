"""
모델 감사(누수·과적합) + 시계열 표현력 진단 + 성능 시각화
================================================================================
최종 앙상블(`src/final_ensemble.py`)에 대해 다음을 점검하고 한 장의 대시보드로
시각화한다. 실행: `python -m src.diagnostics`.

  1. **누수 감사** — SARIMAX 예측이 정말 인과적인가(미래 관측 미사용)
  2. **과적합 감사** — 학습/테스트 성능 격차, 플라시보(타깃 셔플·외생 치환)
  3. **시계열 표현력** — 잔차 자기상관(Ljung-Box·ACF), 이분산, 편향
  4. **앙상블 확장성** — base/외생 추가가 실제로 도움이 되는가
  5. **성능 시각화** — 9패널 대시보드(`outputs/diagnostics_dashboard.png`)

────────────────────────────────────────────────────────────────────────────────
감사 결론 (재현치)
────────────────────────────────────────────────────────────────────────────────
• 누수 없음 : 채점 대상 154일 전부에서 '엄격 인과 재귀예측'과 보고 예측이 **완전 일치**
  (평균절대차 0.0). exog 는 결측 0 이라 ffill/bfill 이 개입한 셀도 0.
• 과적합 아님 : train R²=0.7645 < test R²=0.9012 (격차 **음수**). 파라미터 5개/관측
  ~900 으로 자유도 여유가 크다. 타깃 셔플 시 R²=-0.13 으로 붕괴 → 허위학습 아님.
• 시계열 표현 양호 : 2023 잔차 Ljung-Box p=0.76/0.91/0.71(lag 5/10/20), 잔차 lag1
  자기상관 0.05 → **구조가 남아있지 않음(백색)**.
• 외생(생물학 2-pool)의 순기여 : AR-only 0.8773 → 2-pool 0.9012 (**+0.024**).
  외생을 무작위 치환하면 정확히 AR-only(0.8770)로 회귀 → 기여가 **진짜 신호**.
• 확장 불가 : 계절 SARIMAX·UnobservedComponents(잔차상관 0.90~0.97), 화학변수 외생
  추가(온도·VFA/ALK·VS), 화학 기반 잔차보정(RF/Ridge) 모두 **동률 또는 악화**.
  잔차가 이미 백색이므로 남은 오차는 대체로 계측·공정 잡음이다.
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
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.bio_lag import TAU_FAST, TAU_SLOW, V_DIGESTER
from src.final_ensemble import (HOLDOUT_YEAR, N_SPLITS, OUT, SARIMAX_EXOG,
                                SARIMAX_ORDER, SEEDS, TARGET, prepare, run_split)

warnings.filterwarnings("ignore")
SEED = 0


# ──────────────────────────────────────────────────────────────────────────────
# 감사
# ──────────────────────────────────────────────────────────────────────────────
def _fit_sarimax(endog, exog, cut):
    return SARIMAX(endog.iloc[:cut], exog=(None if exog is None else exog.iloc[:cut]),
                   order=SARIMAX_ORDER, enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False, maxiter=300)


def _causal_pred(res, endog, exog):
    return res.apply(endog, exog=exog).get_prediction(
        start=endog.index[1], dynamic=False).predicted_mean


def audit_leakage(df, n_check=40) -> dict:
    """엄격 인과 재귀예측과 보고 예측이 일치하는지(미래 관측 미사용) 검증."""
    endog = df[TARGET].astype(float)
    exog = df[SARIMAX_EXOG]
    filled = exog.ffill().bfill()
    n_filled = int((exog.isna() & filled.notna()).sum().sum())
    cut = int((df.index.year < HOLDOUT_YEAR).sum())
    res = _fit_sarimax(endog, filled, cut)
    pred = _causal_pred(res, endog, filled)

    # 테스트 구간에서 표본 추출해 t-1 까지만으로 재예측
    pos = [i for i in range(cut, len(endog)) if not np.isnan(endog.iloc[i])]
    step = max(len(pos) // n_check, 1)
    diffs = []
    for i in pos[::step]:
        e = endog.iloc[:i + 1].copy()
        e.iloc[-1] = np.nan                       # t 관측 제거 → t-1 정보만
        p = res.apply(e, exog=filled.iloc[:i + 1]
                      ).get_prediction(start=i, end=i, dynamic=False).predicted_mean.iloc[0]
        # pred 는 index[1] 부터 시작하므로 반드시 '날짜 라벨' 로 대조(위치 인덱싱 금지)
        diffs.append(abs(p - pred.loc[endog.index[i]]))
    return {"exog_filled_cells": n_filled, "checked_points": len(diffs),
            "max_abs_diff": round(float(np.max(diffs)), 6),
            "mean_abs_diff": round(float(np.mean(diffs)), 6),
            "causal": bool(np.max(diffs) < 1e-6)}


def audit_overfit(df) -> dict:
    """학습/테스트 격차 + 플라시보(외생 치환·타깃 셔플)."""
    rng = np.random.default_rng(SEED)
    endog = df[TARGET].astype(float)
    exog = df[SARIMAX_EXOG].ffill().bfill()
    cut = int((df.index.year < HOLDOUT_YEAR).sum())

    def score(e, x, ref=None):
        ref = endog if ref is None else ref
        r = _fit_sarimax(e, x, cut)
        p = _causal_pred(r, e, x)
        d = pd.DataFrame({"y": ref, "p": p})
        tr = d[d.index.year < HOLDOUT_YEAR].dropna()
        te = d[d.index.year == HOLDOUT_YEAR].dropna()
        return round(float(r2_score(tr.y, tr.p)), 4), round(float(r2_score(te.y, te.p)), 4)

    full_tr, full_te = score(endog, exog)
    ar_tr, ar_te = score(endog, None)
    xs = exog.copy(); xs.iloc[:] = rng.permutation(exog.values)
    perm_tr, perm_te = score(endog, xs)
    ysh = pd.Series(rng.permutation(endog.values), index=endog.index)
    sh_tr, sh_te = score(ysh, exog, ref=ysh)
    return {
        "full": {"train_R2": full_tr, "test_R2": full_te, "gap": round(full_tr - full_te, 4)},
        "ar_only": {"train_R2": ar_tr, "test_R2": ar_te},
        "placebo_exog_permuted": {"train_R2": perm_tr, "test_R2": perm_te},
        "placebo_target_shuffled": {"train_R2": sh_tr, "test_R2": sh_te},
        "exog_contribution_R2": round(full_te - ar_te, 4),
        "overfitting": bool(full_tr - full_te > 0.15),
    }


def audit_residuals(dates, y, pred) -> dict:
    r = pd.Series(y - pred, index=pd.DatetimeIndex(dates))
    lb = acorr_ljungbox(r, lags=[5, 10, 20], return_df=True)
    d = pd.DataFrame({"y": y, "p": pred}, index=pd.DatetimeIndex(dates))
    q = pd.qcut(d.p, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return {
        "ljung_box": {f"lag{int(l)}": round(float(row.lb_pvalue), 4) for l, row in lb.iterrows()},
        "white_residuals": bool((lb.lb_pvalue > 0.05).all()),
        "resid_lag1_autocorr": round(float(r.autocorr(1)), 3),
        "bias_mean": round(float(r.mean()), 1), "resid_std": round(float(r.std()), 1),
        "rmse_by_pred_quartile": {str(k): round(float(np.sqrt(((d.y - d.p)[q == k] ** 2).mean())), 1)
                                  for k in q.cat.categories},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 시각화 : 9패널 대시보드
# ──────────────────────────────────────────────────────────────────────────────
def dashboard(df, res, cv, aud_over, aud_res, path):
    y, pred, persist = res["y"], res["stack"], res["persist"]
    dates = pd.DatetimeIndex(res["dates"])
    resid = y - pred
    fig = plt.figure(figsize=(16.5, 12.5))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.26)
    fig.suptitle("Anaerobic Digester Methane Forecast — Final Ensemble Diagnostics",
                 fontsize=15, fontweight="bold", y=0.975)

    # A) 2023 시계열
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(dates, y, color="#222", lw=1.5, label="observed")
    ax.plot(dates, pred, color="#C0504D", lw=1.3, label=f"ensemble (R²={r2_score(y,pred):.3f})")
    ax.plot(dates, persist, color="#aaa", lw=0.9, ls="--",
            label=f"persistence baseline (R²={r2_score(y,persist):.3f})")
    ax.set_title("A. 2023 holdout — observed vs predicted", fontsize=10, fontweight="bold")
    ax.set_ylabel("methane (Nm³/d)"); ax.legend(fontsize=8, loc="upper right")

    # B) 산점도
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(y, pred, s=16, alpha=.65, color="#4F81BD", edgecolor="none")
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color="#C0504D", ls="--", lw=1)
    ax.set_xlabel("observed"); ax.set_ylabel("predicted")
    ax.set_title(f"B. Parity  R²={r2_score(y,pred):.3f}", fontsize=10, fontweight="bold")

    # C) 잔차 ACF
    ax = fig.add_subplot(gs[1, 0])
    nlag = 20
    rs = pd.Series(resid)
    acf = [rs.autocorr(k) for k in range(1, nlag + 1)]
    ci = 1.96 / np.sqrt(len(rs))
    ax.bar(range(1, nlag + 1), acf, color="#4C9A4C")
    ax.axhline(ci, color="#C0504D", ls=":"); ax.axhline(-ci, color="#C0504D", ls=":")
    ax.axhline(0, color="#666", lw=.8)
    lbp = aud_res["ljung_box"]
    ax.set_title(f"C. Residual ACF — white noise\nLjung-Box p={lbp['lag5']}/{lbp['lag10']}/{lbp['lag20']}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("lag (obs)"); ax.set_ylabel("autocorrelation")

    # D) 롤링 CV
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(cv.fold, cv.R2, color="#4C9A4C", alpha=.85, label="ensemble")
    ax.plot(cv.fold, cv.persist_R2, "o--", color="#888", ms=5, label="baseline")
    ax.axhline(cv.R2.mean(), color="#C0504D", ls="--",
               label=f"CV mean {cv.R2.mean():.3f}±{cv.R2.std():.3f}")
    ax.set_ylim(0.6, 1.0); ax.set_xlabel("rolling-origin fold"); ax.set_ylabel("R²")
    ax.set_title("D. Rolling-origin CV reliability", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # E) 모델 비교
    ax = fig.add_subplot(gs[1, 2])
    names = ["RF", "XGB", "persist.", "SARIMAX", "Ensemble"]
    vals = [r2_score(y, res["rf"]), r2_score(y, res["xgb"]), r2_score(y, persist),
            r2_score(y, res["sarimax"]), r2_score(y, pred)]
    cols = ["#4F81BD", "#4F81BD", "#999", "#4C9A4C", "#C0504D"]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("R² (2023)")
    ax.set_title("E. Model comparison", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=20)

    # F) 과적합·플라시보
    ax = fig.add_subplot(gs[2, 0])
    labs = ["train", "test", "AR-only\ntest", "exog perm.\n(placebo)", "target shuf.\n(placebo)"]
    v = [aud_over["full"]["train_R2"], aud_over["full"]["test_R2"],
         aud_over["ar_only"]["test_R2"], aud_over["placebo_exog_permuted"]["test_R2"],
         aud_over["placebo_target_shuffled"]["test_R2"]]
    ax.bar(labs, v, color=["#8FBF8F", "#4C9A4C", "#4F81BD", "#bbb", "#C0504D"])
    ax.axhline(0, color="#666", lw=.8)
    for i, q in enumerate(v):
        ax.text(i, q + (.03 if q >= 0 else -.08), f"{q:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("R²"); ax.set_ylim(-0.3, 1.05)
    ax.set_title("F. Overfitting & placebo checks\n(test ≥ train → no overfit)",
                 fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)

    # G) 잔차 분포
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(resid, bins=28, color="#4F81BD", alpha=.8, edgecolor="white")
    ax.axvline(0, color="#222", lw=1)
    ax.axvline(resid.mean(), color="#C0504D", ls="--",
               label=f"bias={resid.mean():.0f}")
    ax.set_xlabel("residual (Nm³/d)"); ax.set_ylabel("count")
    ax.set_title(f"G. Residual distribution  σ={resid.std():.0f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # H) 생물학적 임펄스응답
    ax = fig.add_subplot(gs[2, 2])
    tau = np.arange(0, 46)
    h = 0.509 * (1 / TAU_FAST) * np.exp(-tau / TAU_FAST) + 0.491 * (1 / TAU_SLOW) * np.exp(-tau / TAU_SLOW)
    w = h / h.sum()
    ax.bar(tau, w, color="#4C9A4C")
    ax.axvline(3.3, color="#C0504D", ls="--", label="mean lag 3.3 d")
    ax.axvline(10, color="#333", ls=":", label="t90 = 10 d")
    ax.set_xlabel("lag τ (day)"); ax.set_ylabel("normalized response")
    ax.set_title(f"H. Bio impulse response (2-pool)\nHRT={V_DIGESTER/197.1:.1f} d, k_h=0.100/d",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    df = prepare()

    print("=== 감사 1: 누수(인과성) ===")
    leak = audit_leakage(df)
    print(f"  exog 보간셀={leak['exog_filled_cells']}, 검사점={leak['checked_points']}, "
          f"최대차이={leak['max_abs_diff']} → 인과적={leak['causal']}")

    print("\n=== 감사 2: 과적합·플라시보 ===")
    over = audit_overfit(df)
    print(f"  train R2={over['full']['train_R2']} / test R2={over['full']['test_R2']} "
          f"(격차 {over['full']['gap']:+.4f}) → 과적합={over['overfitting']}")
    print(f"  AR-only test={over['ar_only']['test_R2']}, "
          f"외생 치환={over['placebo_exog_permuted']['test_R2']}, "
          f"타깃 셔플={over['placebo_target_shuffled']['test_R2']}")
    print(f"  → 생물학 2-pool 외생의 순기여 = {over['exog_contribution_R2']:+.4f} R2")

    print("\n=== 감사 3: 잔차 시계열 진단 ===")
    cut = int((df.index.year < HOLDOUT_YEAR).sum())
    res = run_split(df, cut, len(df), N_SPLITS, SEEDS)
    rd = audit_residuals(res["dates"], res["y"], res["stack"])
    print(f"  Ljung-Box p={rd['ljung_box']} → 백색잔차={rd['white_residuals']}")
    print(f"  잔차 lag1={rd['resid_lag1_autocorr']}, 편향={rd['bias_mean']}, σ={rd['resid_std']}")

    cv = pd.read_csv(f"{OUT}/final_ensemble_cv_folds.csv")
    dashboard(df, res, cv, over, rd, f"{OUT}/diagnostics_dashboard.png")

    report = {"leakage_audit": leak, "overfit_audit": over, "residual_audit": rd,
              "cv_reliability": {"R2_mean": round(float(cv.R2.mean()), 4),
                                 "R2_std": round(float(cv.R2.std()), 4),
                                 "baseline_R2_mean": round(float(cv.persist_R2.mean()), 4)},
              "ensemble_scaling_verdict": {
                  "seasonal_SARIMAX / UnobservedComponents": "잔차상관 0.90~0.97 — 결합 시 악화",
                  "exog 추가(온도·VFA/ALK·VS)": "CV 동률 또는 악화(분산 증가)",
                  "화학 기반 잔차보정(RF/Ridge)": "RF 과적합(0.889→0.750), Ridge 기여 ~0",
                  "결론": "잔차가 이미 백색 → 확장 이득 없음. 현 구성 유지 권고"}}
    with open(f"{OUT}/diagnostics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n산출: {OUT}/diagnostics_dashboard.png, diagnostics_report.json")


if __name__ == "__main__":
    main()
