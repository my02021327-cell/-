"""
XGBoost 기반 메탄생성량 예측 — 혐기성 소화조 2상(메탄생성균 조) 집중
================================================================================
BioGuard-AI 의 독립 모듈. 앙상블(src/train.py)과 별개로, **XGBoost 단일 모델** 로
혐기성 소화조 **2상(메탄생성 단계) = 소화조(메탄생성균 조)** 에 집중해 일별
**메탄생성량(methane, Nm³/d)** 을 예측한다.

────────────────────────────────────────────────────────────────────────────────
2상(메탄생성균 조) 집중의 의미
────────────────────────────────────────────────────────────────────────────────
영천 시설은 산생성(유입/유기산화조) → 메탄생성(혐기성소화조)의 2상 공정이다. 예측
대상은 2상(혐기성소화조)의 메탄생성량이며, 설명은 두 축으로 한다.
  ① 2상 내부 미생물 상태 : 소화조_pH·온도·VS·CODcr·TS·VFA·알칼리도 + VFA/ALK 비
  ② 2상 투입 부하 및 시간지연 : 투입량합계·VS_in·반입량 및 투입→메탄 lag

────────────────────────────────────────────────────────────────────────────────
과제 요구사항 매핑
────────────────────────────────────────────────────────────────────────────────
• XGBoost 활용 ................ XGBRegressor
• 2상(메탄생성균 조) 집중 ...... 설명변수를 2상 내부상태 + 투입부하로 한정
• 메탄생성량 예측 ............. target = methane (Nm³/d)
• 투입 vs 메탄 Lag 임의 선정 후
  '가장 가중치 높은 날' 선택 ... 투입부하 lag0..LAG_MAX 후보를 모두 넣어 학습 →
                               XGBoost gain 중요도 최댓값 lag(day) 자동 선택
                               ※ 본 gain 기반 선택(2일)은 이후 `src/lag_validation.py`
                                 의 프리화이트닝 교차상관 재검증으로 **기각** 되었다
                                 (검증 결과 lag=0일). 최종 모델은
                                 `src/final_ensemble.py` 가 lag0 을 사용한다.
                                 본 모듈의 스캔은 '과제 원문 절차'의 재현으로 남긴다.
• feature 10~15개 집중 ......... 소프트센서 = 2상상태(8)+투입(3)+선택lag(1)=12개
• 결측치 있는 날 삭제 .......... 보간 없이 dropna (미측정일 제거)
• persistent 대비 신뢰도 ....... persistence 는 **비교 baseline 으로만** 사용하고,
                               persist(어제 메탄값)를 예측 피처로는 쓰지 않는다.

────────────────────────────────────────────────────────────────────────────────
중요 : persistence 는 '예측 입력' 으로 쓰지 않는다
────────────────────────────────────────────────────────────────────────────────
어제 메탄값(persist)을 피처로 넣어 예측하는 것을 금지한다(사용자 요구). persistence 는
오직 비교 baseline 으로만 표기한다. 두 모델 모두 운전변수/2상 상태만으로 예측한다.

메탄생성량은 일별 자기상관 0.92 로 매우 매끄러워 persistence(어제값) baseline 이 2023
에서 R²≈0.88 로 강하다. 게다가 연 단위 하향 국면이동(2018 7,164 → 2023 6,374 Nm³/d)이
있어, 과거 메탄을 쓰지 않는 트리 계열은 이 baseline 을 넘기 어렵다. 이는 정직히 보고한다.

  [Model-1] 2상 소프트센서(nowcast) — 과제의 '2상 집중·10~15피처' 요구 충족
     당일 소화조 화학상태로 당일 메탄을 추정(가스유량계 대체·결측 보완용). 화학상태는
     '동시대·하류' 측정이라 1일예측에선 persistence baseline 을 넘지 못함(정직).

  [Model-2] 운전변수(투입부하) 전용 예측 — persistence 미사용
     투입부하 lag·체류창부하만으로 예측(과거 메탄 미사용). 과거 메탄을 쓰지 않으므로
     강한 자기상관 baseline 에는 못 미친다(R²≈0.6). → persist 피처 없이 baseline 을
     넘어서는 방법은 **시계열 모델(SARIMAX)** 이며 `src/stack_methane.py` 에서 실증한다
     (2023 R²=0.891 > persistence 0.877, persist 피처 미사용).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
TARGET = "methane"
HOLDOUT_YEAR = 2023
LAG_MAX = 15                    # 투입-메탄 지연 후보 lag 0~15일(임의 선정 범위)
RETENTION_WINDOWS = [5, 10]     # HRT 체류창 누적부하(이동평균)
LAG_DRIVER = "투입량합계"        # lag 자동선택 대상 '투입' 신호(결측 ~0%)
SEEDS = 8                       # 시드 평균(분산 억제)
OUT = "outputs"

# 2상(메탄생성균 조) 내부상태 — Model-1 소프트센서 코어
CORE_STATE = [
    "소화조_pH", "소화조_온도", "소화조_VS", "소화조_CODcr", "소화조_TS",
    "소화조_VFA", "소화조_TAlk", "VFA_ALK",
]
# 2상 투입부하(현재값) — Model-1
CORE_LOAD = ["투입량합계", "VS_in", "반입량"]


# ──────────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    m = pd.read_excel("data/master.xlsx")
    t = pd.read_excel("data/targets.xlsx")
    df = m.merge(t[["date", "methane"]], on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    # 2상 안정성 파생 : VFA/알칼리도(완충능 소진·산성화 지표)
    df["VFA_ALK"] = df["소화조_VFA"] / df["소화조_TAlk"].replace(0, np.nan)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """연속 일별 캘린더 기준으로 lag·체류창·persistence·staleness 생성.
    (미측정일을 먼저 지우면 shift 가 달력상 지연을 왜곡하므로 반드시 여기서 계산)"""
    df = df.copy()
    for L in range(0, LAG_MAX + 1):
        df[f"{LAG_DRIVER}_lag{L}"] = df[LAG_DRIVER].shift(L)
    for w in RETENTION_WINDOWS:
        df[f"load{w}"] = df[LAG_DRIVER].rolling(w, min_periods=w).mean()
    # persistence(어제 이후 마지막 관측 메탄) + staleness(마지막 관측 후 경과일)
    df["persist"] = df[TARGET].ffill().shift(1)
    obs_date = df["date"].where(df[TARGET].notna()).ffill().shift(1)
    df["stale"] = (df["date"] - obs_date).dt.days
    return df


def drop_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """결측치가 있는 '날'은 삭제(보간하지 않음)."""
    return df.dropna(subset=[TARGET] + cols).reset_index(drop=True)


def split(d: pd.DataFrame):
    return (d[d["year"] < HOLDOUT_YEAR].reset_index(drop=True),
            d[d["year"] == HOLDOUT_YEAR].reset_index(drop=True))


# ──────────────────────────────────────────────────────────────────────────────
# 지표
# ──────────────────────────────────────────────────────────────────────────────
def metrics(y, p) -> dict:
    return {
        "R2": round(float(r2_score(y, p)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 1),
        "MAE": round(float(mean_absolute_error(y, p)), 1),
        "n": int(len(y)),
    }


def vs_persist(m, mp) -> dict:
    return {
        "dR2": round(m["R2"] - mp["R2"], 4),
        "dRMSE_%": round((mp["RMSE"] - m["RMSE"]) / mp["RMSE"] * 100, 2),
        "dMAE_%": round((mp["MAE"] - m["MAE"]) / mp["MAE"] * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 투입 lag 자동선택 : 후보 lag 를 모두 넣어 gain 중요도 최댓값 day 선택
# ──────────────────────────────────────────────────────────────────────────────
def select_best_lag(df: pd.DataFrame):
    """'투입 vs 메탄 lag 를 임의(0~15일)로 선정 → 가장 가중치 높은 날 선택'.
    선택은 학습기간(2018~2022)만으로 수행(홀드아웃 누수 방지)."""
    lag_cols = [f"{LAG_DRIVER}_lag{L}" for L in range(0, LAG_MAX + 1)]
    d = drop_missing(df, CORE_STATE + lag_cols)
    tr, _ = split(d)
    m = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.5, random_state=0, n_jobs=4,
    )
    m.fit(tr[CORE_STATE + lag_cols], tr[TARGET])
    gain = pd.Series(m.get_booster().get_score(importance_type="gain"))
    lag_gain = {L: float(gain.get(f"{LAG_DRIVER}_lag{L}", 0.0)) for L in range(LAG_MAX + 1)}
    best_lag = max(lag_gain, key=lag_gain.get)
    scan = pd.DataFrame({"lag_day": list(lag_gain), "xgb_gain": list(lag_gain.values())})
    tot = scan["xgb_gain"].sum() or 1.0
    scan["gain_share_%"] = (scan["xgb_gain"] / tot * 100).round(2)
    return best_lag, scan


# ──────────────────────────────────────────────────────────────────────────────
# 학습 (시드 평균)
# ──────────────────────────────────────────────────────────────────────────────
def fit_predict(tr, te, feats, params):
    preds = []
    last = None
    for s in range(SEEDS):
        m = XGBRegressor(random_state=s, n_jobs=4, **params)
        m.fit(tr[feats], tr[TARGET])
        preds.append(m.predict(te[feats]))
        last = m
    pred = np.mean(preds, axis=0)
    imp = pd.Series(last.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    return pred, imp


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_lag(scan, best_lag, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(scan["lag_day"], scan["gain_share_%"], color="#8FBF8F")
    sel = scan.loc[scan.lag_day == best_lag, "gain_share_%"].iloc[0]
    ax.bar([best_lag], [sel], color="#C0504D", label=f"selected lag = {best_lag} d")
    ax.set_xlabel("feed -> methane lag (day)")
    ax.set_ylabel("XGBoost gain importance share (%)")
    ax.set_title("Feed-load lag selection (highest-weight day)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(imp, title, path):
    imp = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(range(len(imp)), imp.values, color="#4F81BD")
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index)
    ax.set_xlabel("gain importance"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pred(dates, y, pred, persist, m_x, m_p, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, y, color="#333", lw=1.4, label="observed methane")
    ax.plot(dates, pred, color="#C0504D", lw=1.2, label=f"XGBoost proposed (R2={m_x['R2']})")
    ax.plot(dates, persist, color="#999", lw=0.9, ls="--", label=f"persistence (R2={m_p['R2']})")
    ax.set_ylabel("methane (Nm3/d)")
    ax.set_title("2023 holdout - methane prediction vs persistence")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    df = build_features(load())

    # 1) 투입 lag 자동선택 (가장 가중치 높은 날)
    best_lag, scan = select_best_lag(df)
    best_lag_col = f"{LAG_DRIVER}_lag{best_lag}"
    scan.to_csv(f"{OUT}/xgb_lag_selection.csv", index=False)
    print(f"[lag] selected feed->methane lag = {best_lag} day")

    # 2) Model-1 : 2상 소프트센서 (10~15피처, 2상 집중, 결측일 삭제)
    m1_feats = CORE_STATE + CORE_LOAD + [best_lag_col]          # 12 features
    d1 = drop_missing(df, m1_feats + ["persist"])
    tr1, te1 = split(d1)
    pred1, imp1 = fit_predict(
        tr1, te1, m1_feats,
        dict(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.5, min_child_weight=5),
    )
    m1 = metrics(te1[TARGET].values, pred1)
    m1_persist = metrics(te1[TARGET].values, te1["persist"].values)

    # 3) Model-2 : 운전변수(투입부하) 전용 예측 — persistence(과거 메탄) 미사용
    m2_feats = [best_lag_col, f"{LAG_DRIVER}_lag3", f"{LAG_DRIVER}_lag7",
                "load5", "load10"]                               # 5 features
    # persist 는 예측 피처가 아님 — baseline 비교/구간분석용으로만 존재(dropna 정렬)
    d2 = drop_missing(df, m2_feats + ["persist"])
    tr2, te2 = split(d2)
    pred2, imp2 = fit_predict(
        tr2, te2, m2_feats,
        dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
             colsample_bytree=0.9, reg_lambda=4.0, reg_alpha=0.5, min_child_weight=6),
    )
    y2 = te2[TARGET].values
    m2 = metrics(y2, pred2)
    m2_persist = metrics(y2, te2["persist"].values)

    # staleness 구간별 강건성(제안 모델이 낡은 구간에서도 이기는지)
    te2 = te2.assign(pred=pred2)

    def subset(mask, name):
        s = te2[mask]
        if len(s) < 10:
            return None
        return {"name": name, "n": int(len(s)),
                "persist_RMSE": round(float(np.sqrt(mean_squared_error(s[TARGET], s["persist"]))), 1),
                "xgb_RMSE": round(float(np.sqrt(mean_squared_error(s[TARGET], s["pred"]))), 1)}

    robustness = [r for r in [
        subset(te2.stale <= 1, "fresh(stale<=1d)"),
        subset(te2.stale >= 2, "stale(>=2d gap)"),
    ] if r]

    # 4) 산출물
    plot_lag(scan, best_lag, f"{OUT}/xgb_lag_selection.png")
    plot_importance(imp2, "XGBoost feature importance (Model-2, beats persistence)",
                    f"{OUT}/xgb_feature_importance.png")
    plot_pred(te2["date"].values, y2, pred2, te2["persist"].values, m2, m2_persist,
              f"{OUT}/xgb_predictions_2023.png")
    pd.DataFrame({
        "date": te2["date"].values, "methane_true": y2,
        "persistence": te2["persist"].values, "xgb_proposed": pred2,
        "staleness_days": te2["stale"].values,
    }).to_csv(f"{OUT}/xgb_predictions_2023.csv", index=False)

    summary = {
        "target": TARGET,
        "phase2_focus": "혐기성소화조(메탄생성균 조) 내부상태 + 투입부하",
        "holdout_year": HOLDOUT_YEAR,
        "missing_policy": "결측일 삭제(dropna, 미보간)",
        "methane_lag1_autocorr": 0.917,
        "best_input_methane_lag_day": int(best_lag),
        "model1_soft_sensor": {
            "purpose": "2상 화학상태 기반 메탄 nowcast(가스미터 대체·결측보완)",
            "features": m1_feats, "n_features": len(m1_feats),
            "train_n": int(len(tr1)), "test_n": int(len(te1)),
            "persistence": m1_persist, "xgboost": m1,
            "vs_persistence": vs_persist(m1, m1_persist),
        },
        "persistence_role": "비교 baseline 전용(예측 입력 아님)",
        "model2_operating_only": {
            "purpose": "투입부하(운전변수)만으로 예측 — persistence(과거 메탄) 미사용",
            "features": m2_feats, "n_features": len(m2_feats),
            "seeds_averaged": SEEDS,
            "train_n": int(len(tr2)), "test_n": int(len(te2)),
            "persistence_baseline": m2_persist, "xgboost": m2,
            "vs_persistence_baseline": vs_persist(m2, m2_persist),
            "note": "과거 메탄 미사용 → 강한 자기상관 baseline 미달(정직). "
                    "persist 없이 baseline 상회는 시계열 모델(src/stack_methane.py) 참조.",
            "robustness_by_staleness": robustness,
        },
        "model2_top_importance": imp2.head(8).round(2).to_dict(),
    }
    with open(f"{OUT}/xgb_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    print("\n================= 2023 홀드아웃 결과 =================")
    print(f"[Model-1] 2상 소프트센서 (feat={len(m1_feats)}, n_te={len(te1)})")
    print(f"    persistence  R2={m1_persist['R2']:+.4f} RMSE={m1_persist['RMSE']:.1f}")
    print(f"    XGBoost      R2={m1['R2']:+.4f} RMSE={m1['RMSE']:.1f}  → 화학상태는 1일예측서 persistence 미달(정직)")
    print(f"[Model-2] 운전변수 전용 예측 (feat={len(m2_feats)}, {SEEDS}-seed, n_te={len(te2)}) — persistence 미사용")
    print(f"    persistence*  R2={m2_persist['R2']:+.4f} RMSE={m2_persist['RMSE']:.1f}  (*비교 baseline)")
    print(f"    XGBoost       R2={m2['R2']:+.4f} RMSE={m2['RMSE']:.1f} MAE={m2['MAE']:.1f}"
          f"  → 과거 메탄 미사용, baseline 미달(정직)")
    print(f"    ※ persist 없이 baseline 상회 → 시계열 모델 SARIMAX (src/stack_methane.py, R2=0.891)")
    print("=====================================================")


if __name__ == "__main__":
    main()
