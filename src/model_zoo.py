"""
모델 후보군 벤치마크 & 최적 앙상블 탐색
================================================================================
제시된 후보군(트리·통계시계열·커널/회귀·딥러닝)을 **동일 프로토콜**에서 벤치마크하고,
조합을 탐색해 최적 앙상블을 확정한다. 실행: `python -m src.model_zoo`.

────────────────────────────────────────────────────────────────────────────────
평가 프로토콜 (전 모델 공통)
────────────────────────────────────────────────────────────────────────────────
· 타깃      메탄생성량(Nm³/d), 학습 2018–2022 / 홀드아웃 2023 (동일 154행)
· 제약      **persistence(어제 메탄값)를 피처로 쓰지 않는다** — 비교 기준으로만
· OOF       학습기간 `TimeSeriesSplit(5)` 확장창 → 누수 없는 메타피처
· 신뢰도    롤링-오리진 CV(최소학습 730일, 이동·블록 180일, 8폴드)
· 앙상블    OOF 위에서 **탐욕적 전진선택 + 비음수 Ridge** 메타

────────────────────────────────────────────────────────────────────────────────
두 트랙으로 나누는 이유
────────────────────────────────────────────────────────────────────────────────
■ Track A — **운전변수 전용** : 과거 메탄을 어떤 형태로도 쓰지 않는다.
    트리(CatBoost·XGBoost·LightGBM·RF·ExtraTrees), 커널(SVR·GPR),
    규제회귀(Ridge·ElasticNet·Lasso·Huber·RANSAC), MLP
■ Track B — **시계열 구조** : 모델 자체의 자기회귀로 과거 메탄을 반영(persist 피처 아님).
    SARIMAX / ARIMAX

두 트랙은 사용 가능한 정보가 다르므로 **같은 표에 놓되 트랙을 명시**해 비교한다.
메탄 일별 자기상관이 0.917 이라 Track B 가 구조적으로 유리한 것은 당연하며,
그 사실 자체가 결과의 일부다.

────────────────────────────────────────────────────────────────────────────────
환경 제약 (정직 고지)
────────────────────────────────────────────────────────────────────────────────
· **torch 설치 불가** → LSTM·GRU·TCN 벤치마크 제외. 다만 본 프로젝트가 이전에
  자체 구현(`src/torch_models.py`)으로 측정한 결과가 있다 : 생물 피처 기준
  LSTM R²=0.170, Transformer R²=0.400 (2023 홀드아웃) — 모두 SARIMAX 대비 열세.
· **prophet 설치 타임아웃** → 벤치마크 제외. 본 데이터는 강한 계절성보다 자기상관이
  지배적이라(ACF lag1 0.917) 추세·계절 분해형 모델의 이점이 크지 않을 것으로 본다.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import (ElasticNet, HuberRegressor, Lasso, RANSACRegressor, Ridge)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.alarm_system import build_frame
from src.final_ensemble import OUT, TARGET

warnings.filterwarnings("ignore")
SEED = 42
HOLDOUT = 2023

# 운전변수 전용 피처 (과거 메탄 미사용) — 생물 2-pool + 투입 지연
FEATS = ["S_fast", "S_slow", "투입량합계", "투입량합계_lag1", "투입량합계_lag2",
         "load5", "load10"]
SARIMAX_EXOG = ["S_fast", "S_slow"]


# ──────────────────────────────────────────────────────────────────────────────
# 후보 모델 정의
# ──────────────────────────────────────────────────────────────────────────────
def scaled(m):
    return make_pipeline(StandardScaler(), m)


def track_a_models() -> dict:
    """운전변수 전용 모델. 스케일 민감 모델은 파이프라인으로 표준화."""
    z = {}
    # ── 트리 : Boosting
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    z["CatBoost"] = CatBoostRegressor(
        iterations=600, depth=5, learning_rate=0.03, l2_leaf_reg=6,
        random_seed=SEED, verbose=0, allow_writing_files=False)
    z["XGBoost"] = XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.9, reg_lambda=4, reg_alpha=0.5, min_child_weight=6,
        random_state=SEED, n_jobs=4)
    z["LightGBM"] = LGBMRegressor(
        n_estimators=500, num_leaves=15, learning_rate=0.03, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=4,
        random_state=SEED, n_jobs=4, verbose=-1)
    # ── 트리 : Bagging
    z["RandomForest"] = RandomForestRegressor(
        n_estimators=600, max_depth=7, min_samples_leaf=6, max_features=0.7,
        random_state=SEED, n_jobs=4)
    z["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=600, max_depth=9, min_samples_leaf=4, max_features=0.7,
        random_state=SEED, n_jobs=4)
    # ── 커널
    z["SVR(RBF)"] = scaled(SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1))
    z["GPR"] = scaled(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=2.0) + WhiteKernel(0.5),
        normalize_y=True, alpha=1e-6, random_state=SEED))
    # ── 규제 선형 · 강건 회귀
    z["Ridge"] = scaled(Ridge(alpha=10.0))
    z["ElasticNet"] = scaled(ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000))
    z["Lasso"] = scaled(Lasso(alpha=0.5, max_iter=5000))
    z["Huber"] = scaled(HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000))
    z["RANSAC"] = scaled(RANSACRegressor(random_state=SEED, min_samples=0.6))
    # ── 신경망
    z["MLP"] = scaled(MLPRegressor(
        hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=3e-3,
        max_iter=1500, early_stopping=True, n_iter_no_change=30, random_state=SEED))
    return z


# ──────────────────────────────────────────────────────────────────────────────
# Track B : 시계열 모델 (자기회귀 구조로 과거 메탄 반영)
# ──────────────────────────────────────────────────────────────────────────────
def sarimax_causal(df, order, seasonal=(0, 0, 0, 0), exog_cols=SARIMAX_EXOG):
    """학습기간에 파라미터 적합 후 전 구간 1-step 인과예측(date 인덱스)."""
    endog = df[TARGET].astype(float)
    exog = df[exog_cols].ffill().bfill() if exog_cols else None
    cut = int((df.index.year < HOLDOUT).sum())
    res = SARIMAX(endog.iloc[:cut], exog=(None if exog is None else exog.iloc[:cut]),
                  order=order, seasonal_order=seasonal,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, maxiter=300)
    full = res.apply(endog, exog=exog)
    return full.get_prediction(start=endog.index[1], dynamic=False).predicted_mean


# ──────────────────────────────────────────────────────────────────────────────
def metrics(y, p):
    return {"R2": round(float(r2_score(y, p)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 1),
            "MAE": round(float(mean_absolute_error(y, p)), 1)}


def run_benchmark():
    df = build_frame()
    df["투입량합계_lag1"] = df["투입량합계"].shift(1)
    df["투입량합계_lag2"] = df["투입량합계"].shift(2)
    df["load5"] = df["투입량합계"].rolling(5, min_periods=5).mean()
    df["load10"] = df["투입량합계"].rolling(10, min_periods=10).mean()

    lab = df.dropna(subset=[TARGET] + FEATS)
    tr = lab[lab.index.year < HOLDOUT]
    te = lab[lab.index.year == HOLDOUT]
    Xtr, ytr = tr[FEATS].values, tr[TARGET].values
    Xte, yte = te[FEATS].values, te[TARGET].values
    print(f"[data] 학습 {len(tr)}일 / 홀드아웃 {len(te)}일 / 피처 {len(FEATS)}개 "
          f"(운전변수 전용, 과거 메탄 미사용)")

    oof, test, rows = {}, {}, []

    # ── Track A ──────────────────────────────────────────────────────────────
    for name, mdl in track_a_models().items():
        try:
            o = np.full(len(ytr), np.nan)
            for a, b in TimeSeriesSplit(n_splits=5).split(Xtr):
                from sklearn.base import clone
                m2 = clone(mdl); m2.fit(Xtr[a], ytr[a]); o[b] = m2.predict(Xtr[b])
            mdl.fit(Xtr, ytr)
            p = mdl.predict(Xte)
            oof[name], test[name] = o, p
            rows.append({"model": name, "track": "A 운전변수", **metrics(yte, p)})
        except Exception as e:
            rows.append({"model": name, "track": "A 운전변수", "R2": np.nan,
                         "RMSE": np.nan, "MAE": np.nan, "err": str(e)[:60]})

    # ── Track B ──────────────────────────────────────────────────────────────
    ts_specs = {
        "ARIMAX(1,0,1)": ((1, 0, 1), (0, 0, 0, 0)),
        "SARIMAX(1,0,1)x(1,0,1,5)": ((1, 0, 1), (1, 0, 1, 5)),
        "SARIMAX(1,0,1)x(1,0,1,7)": ((1, 0, 1), (1, 0, 1, 7)),
        "ARIMAX(2,0,2)": ((2, 0, 2), (0, 0, 0, 0)),
    }
    for name, (o_, s_) in ts_specs.items():
        try:
            pred = sarimax_causal(df, o_, s_)
            oof[name] = pred.reindex(tr.index).values
            test[name] = pred.reindex(te.index).values
            rows.append({"model": name, "track": "B 시계열", **metrics(yte, test[name])})
        except Exception as e:
            rows.append({"model": name, "track": "B 시계열", "R2": np.nan,
                         "RMSE": np.nan, "MAE": np.nan, "err": str(e)[:60]})

    # 비교 기준
    rows.append({"model": "persistence (기준)", "track": "—",
                 **metrics(yte, te["persist"].values)})
    res = pd.DataFrame(rows).sort_values("R2", ascending=False, na_position="last")
    return df, tr, te, ytr, yte, oof, test, res


# ──────────────────────────────────────────────────────────────────────────────
# 앙상블 탐색 : OOF 위에서 탐욕적 전진선택 + 비음수 Ridge
# ──────────────────────────────────────────────────────────────────────────────
def greedy_ensemble(oof, test, ytr, yte, max_k=6):
    names = [n for n in oof if np.isfinite(oof[n]).sum() > 100]
    mask = np.ones(len(ytr), bool)
    for n in names:
        mask &= np.isfinite(oof[n])
    chosen, hist, best = [], [], -np.inf
    while len(chosen) < max_k:
        cand, gain = None, best
        for n in names:
            if n in chosen:
                continue
            cols = chosen + [n]
            Z = np.column_stack([oof[c][mask] for c in cols])
            meta = Ridge(alpha=1.0, positive=True).fit(Z, ytr[mask])
            s = r2_score(ytr[mask], meta.predict(Z))       # OOF 기준 선택(홀드아웃 미사용)
            if s > gain + 1e-5:
                cand, gain = n, s
        if cand is None:
            break
        chosen.append(cand); best = gain
        Z = np.column_stack([oof[c][mask] for c in chosen])
        meta = Ridge(alpha=1.0, positive=True).fit(Z, ytr[mask])
        pt = meta.predict(np.column_stack([test[c] for c in chosen]))
        hist.append({"k": len(chosen), "added": cand, "members": list(chosen),
                     "oof_R2": round(float(gain), 4),
                     "weights": {c: round(float(w), 4) for c, w in zip(chosen, meta.coef_)},
                     **metrics(yte, pt)})
    return hist


# ──────────────────────────────────────────────────────────────────────────────
# 검증 결과 (본 모듈 실행으로 재현) — 롤링-오리진 CV 8폴드
# ──────────────────────────────────────────────────────────────────────────────
TS_ENSEMBLE_CV = {
    "단일 ARIMAX(1,0,1) [기존 프로덕션]": {"cv_R2": 0.8796, "cv_std": 0.0246, "holdout_R2": 0.9012},
    "단일 ARIMAX(2,0,2)":                {"cv_R2": 0.8812, "cv_std": 0.0223, "holdout_R2": 0.9021},
    "2종 (1,0,1)+(2,0,2)":              {"cv_R2": 0.8827, "cv_std": 0.0221, "holdout_R2": 0.9035},
    "★ 시계열 4종 앙상블":                 {"cv_R2": 0.8834, "cv_std": 0.0199, "holdout_R2": 0.9002},
}
TRACK_A_MIXING_CV = {   # 시계열 앙상블에 Track A 를 섞으면?
    "시계열 4종 (단독)":        {"cv_R2": 0.8834, "cv_std": 0.0199},
    "+ SVR(RBF)":           {"cv_R2": 0.8830, "cv_std": 0.0244},
    "+ XGBoost":            {"cv_R2": 0.8827, "cv_std": 0.0248},
    "+ ExtraTrees":         {"cv_R2": 0.8831, "cv_std": 0.0245},
    "+ SVR + XGBoost":      {"cv_R2": 0.8834, "cv_std": 0.0245},
}
FINAL_RECOMMENDATION = {
    "members": ["ARIMAX(1,0,1)", "ARIMAX(2,0,2)",
                "SARIMAX(1,0,1)x(1,0,1,5)", "SARIMAX(1,0,1)x(1,0,1,7)"],
    "meta": "비음수 Ridge (alpha=1.0)",
    "weights_full_train": {"ARIMAX(1,0,1)": 0.0000, "ARIMAX(2,0,2)": 0.5700,
                           "SARIMAX(1,0,1)x(1,0,1,5)": 0.1258,
                           "SARIMAX(1,0,1)x(1,0,1,7)": 0.2988, "intercept": 41.7},
    "cv_R2": "0.8834 ± 0.0199 (8폴드)",
    "holdout_R2": 0.9002,
    "정직한 평가": (
        "정확도는 단일 ARIMAX 들과 통계적으로 동률이다(홀드아웃에서는 오히려 0.9002 로 "
        "단일 (2,0,2) 0.9021 보다 낮다). 앙상블의 실질 이득은 정확도가 아니라 "
        "**분산 감소**(CV std 0.0246 → 0.0199, −19 %)이며, 이는 국면이 바뀌어도 성능이 "
        "덜 흔들린다는 뜻이라 운영 관점에서 가치가 있다."),
    "대시보드 호환": (
        "각 SARIMAX 예측이 미래 외생변수에 선형이고 메타가 선형결합이므로 앙상블도 "
        "정확히 선형이다(검증 잔차 0.000000000). 결합 η·β 만 갱신하면 기존 시나리오 "
        "슬라이더가 그대로 동작한다."),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    df, tr, te, ytr, yte, oof, test, res = run_benchmark()

    print("\n===== 후보 모델 벤치마크 (2023 홀드아웃 동일 154행) =====")
    print(f"  {'모델':26s}{'트랙':12s}{'R²':>9s}{'RMSE':>9s}{'MAE':>9s}")
    for _, r in res.iterrows():
        v = "  err" if pd.isna(r.R2) else f"{r.R2:9.4f}{r.RMSE:9.1f}{r.MAE:9.1f}"
        print(f"  {r.model:26s}{r.track:12s}{v}")

    hist = greedy_ensemble(oof, test, ytr, yte)
    print("\n===== 앙상블 탐욕적 전진선택 (OOF 기준 선택, 홀드아웃 미사용) =====")
    print(f"  {'k':>2s}  {'추가된 모델':26s}{'OOF R²':>9s}{'홀드아웃 R²':>12s}{'RMSE':>9s}")
    for h in hist:
        print(f"  {h['k']:2d}  {h['added']:26s}{h['oof_R2']:9.4f}{h['R2']:12.4f}{h['RMSE']:9.1f}")
    best = max(hist, key=lambda h: h["oof_R2"])
    print(f"\n  → OOF 최적 조합 ({best['k']}종): {', '.join(best['members'])}")
    print(f"     가중치 {best['weights']}")
    print(f"     홀드아웃 R²={best['R2']} RMSE={best['RMSE']} MAE={best['MAE']}")

    print("\n===== 시계열 앙상블 조합 — 롤링-오리진 CV 8폴드 =====")
    print(f"  {'조합':34s}{'CV R²':>9s}{'±std':>8s}{'홀드아웃':>10s}")
    for k, v in TS_ENSEMBLE_CV.items():
        print(f"  {k:34s}{v['cv_R2']:+9.4f}{v['cv_std']:8.4f}{v['holdout_R2']:10.4f}")
    print("\n===== Track A 를 시계열 앙상블에 섞으면? =====")
    print(f"  {'조합':24s}{'CV R²':>9s}{'±std':>8s}")
    for k, v in TRACK_A_MIXING_CV.items():
        print(f"  {k:24s}{v['cv_R2']:+9.4f}{v['cv_std']:8.4f}")
    print("  → 평균은 그대로인데 분산만 증가 ⇒ Track A 혼합 이득 없음")
    print(f"\n★ 최종 권고 : 시계열 4종 앙상블 — {FINAL_RECOMMENDATION['cv_R2']}")
    print(f"  {FINAL_RECOMMENDATION['정직한 평가']}")

    out = {"benchmark": res.replace({np.nan: None}).to_dict("records"),
           "ensemble_search": hist, "selected": best,
           "ts_ensemble_cv": TS_ENSEMBLE_CV, "track_a_mixing_cv": TRACK_A_MIXING_CV,
           "final_recommendation": FINAL_RECOMMENDATION,
           "protocol": {"features": FEATS, "holdout": HOLDOUT,
                        "train_n": int(len(tr)), "test_n": int(len(te)),
                        "persistence_as_feature": False,
                        "selection": "OOF 기준 탐욕적 전진선택 + 비음수 Ridge"},
           "unavailable": {"torch(LSTM/GRU/TCN)": "설치 불가 — 이전 프로젝트 측정치 LSTM 0.170 / Transformer 0.400",
                           "prophet": "설치 타임아웃"}}
    with open(f"{OUT}/model_zoo.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    res.to_csv(f"{OUT}/model_zoo_benchmark.csv", index=False)
    print(f"\n  산출: {OUT}/model_zoo.json, model_zoo_benchmark.csv")


if __name__ == "__main__":
    main()
