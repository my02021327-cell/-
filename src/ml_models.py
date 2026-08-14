# -*- coding: utf-8 -*-
"""영천 BGP 메탄생성 ML 예측 — 시계열 + 트리 + NNLS 앙상블.

방향성:
  Track A (단기 1일 운영): 시계열 계열 — persistence / persistence+Δfeed /
      ARIMA(1,1,1)+Δfeed / GBM(Δy 타깃). 평가 지표는 skill = 1 − MSE/MSE_persistence.
  Track B (주간 부하계획, 지평 7일+): 기질 기반 — 가수분해 커널 OLS / RandomForest /
      HistGradientBoosting / NNLS 스태킹 앙상블. 전일 가스량(y lag) 특징 사용 금지(W1).

검증: 시간순 분할 train 2018-2021 / valid 2022 / test 2023. K-fold 금지.
출력: outputs/ml_results.json (HTML 리포트 내장용)
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from scipy.optimize import nnls

CSV = "data/영천BGP_MASTER_2018-2023.csv"
M = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]

y = M.biogas_AB_m3d
TS = M.acid_TS_pct.interpolate(limit=3)
Q = M.feed_AB_tpd.ffill(limit=2)
TS_load = (Q * TS / 100).ffill(limit=3)


def kernel(s, k=0.35, K=120):
    tau = np.arange(K + 1)
    w = k * np.exp(-k * tau)
    w /= w.sum()
    v = np.nan_to_num(s.values)
    out = np.zeros(len(v))
    for i, wv in enumerate(w):
        out[i:] += wv * v[: len(v) - i]
    return pd.Series(out, index=s.index)


H = kernel(TS_load)


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def rmse(t, p):
    return float(np.sqrt(((t - p) ** 2).mean()))


def evaluate(name, pred, frame):
    """frame: DataFrame with y; pred aligned Series. 연도별 R²/RMSE + skill."""
    res = {"name": name}
    for label, yr in [("valid", 2022), ("test", 2023)]:
        sub = frame[frame.index.year == yr]
        p = pred.reindex(sub.index)
        m = sub["y"].notna() & p.notna()
        t, pp = sub["y"][m].values, p[m].values
        # persistence 동일 표본
        pers = y.shift(1).reindex(sub.index)[m].values
        mse_p = ((t - pers) ** 2).mean()
        res[label] = {
            "n": int(m.sum()),
            "R2": round(r2(t, pp), 3),
            "RMSE": round(rmse(t, pp)),
            "skill": round(1 - ((t - pp) ** 2).mean() / mse_p, 3),
        }
    return res


# ================================================================ Track A: 시계열
frameA = pd.DataFrame({"y": y}).dropna()
dfeed = M.feed_AB_tpd.diff()

# A1. persistence
predA1 = y.shift(1)

# A2. persistence + Δfeed (물리 시계열 기준선; 계수는 train으로)
dy_tr = (y - y.shift(1)).loc["2018":"2021"]
x_tr = dfeed.loc["2018":"2021"]
mm = dy_tr.notna() & x_tr.notna()
b1, b0 = np.polyfit(x_tr[mm].values, dy_tr[mm].values, 1)
predA2 = y.shift(1) + b1 * dfeed + b0

# A3. ARIMA(1,1,1) + Δfeed 외생변수 — 학습구간 적합, 이후 1-step 재귀 예측
from statsmodels.tsa.arima.model import ARIMA

ya = y.interpolate(limit=2)
exog = dfeed.reindex(ya.index).fillna(0.0)
tr_end = "2021-12-31"
arima_fit = ARIMA(
    ya.loc[:tr_end], exog=exog.loc[:tr_end], order=(1, 1, 1),
    enforce_stationarity=False, enforce_invertibility=False,
).fit()
# 검증·평가 구간: 상태를 관측으로 갱신하며 1-step ahead (apply + in-sample onestep)
applied = arima_fit.apply(ya.loc["2022":], exog=exog.loc["2022":], refit=False)
predA3 = applied.get_prediction().predicted_mean  # one-step-ahead within applied span

# A4. GBM(Δy 타깃): Δfeed·요일·최근 Δ부하 이력으로 Δy 학습 → ŷ = y(t-1)+Δ̂
featA = pd.DataFrame(
    {
        "dfeed": dfeed,
        "dfeed_l1": dfeed.shift(1),
        "dfeed_l2": dfeed.shift(2),
        "dTS_load": TS_load.diff(),
        "dTS_load_l1": TS_load.diff().shift(1),
        "dow": M.dow,
        "month": M.month,
        "dy_l1": (y - y.shift(1)).shift(1),
    }
)
dy = y - y.shift(1)
DA = pd.concat([featA, dy.rename("dy")], axis=1).dropna(subset=["dy", "dfeed"])
trA = DA[DA.index.year <= 2021]
gbmA = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=0)
gbmA.fit(trA.drop(columns="dy"), trA["dy"])
predA4 = y.shift(1) + pd.Series(gbmA.predict(DA.drop(columns="dy")), index=DA.index)

trackA = [
    evaluate("persistence (전일값)", predA1, frameA),
    evaluate("persistence + Δ투입 회귀", predA2, frameA),
    evaluate("ARIMA(1,1,1) + Δ투입", predA3, frameA),
    evaluate("GBM (Δy 타깃)", predA4, frameA),
]

# ================================================================ Track B: 기질 기반
featB = pd.DataFrame(
    {
        "H": H,
        "TS_load": TS_load,
        "TS_load_l1": TS_load.shift(1),
        "TS_load_l2": TS_load.shift(2),
        "TS_load_l3": TS_load.shift(3),
        "TS_load_m7": TS_load.rolling(7, min_periods=3).mean(),
        "feed": M.feed_AB_tpd,
        "acid_TS": TS,
        "dow": M.dow,
        "month": M.month,
    }
)
DB = pd.concat([featB, y.rename("y")], axis=1).dropna(subset=["y", "H", "TS_load"])
DB = DB.ffill(limit=3).dropna()
trB = DB[DB.index.year <= 2021]
vaB = DB[DB.index.year == 2022]
teB = DB[DB.index.year == 2023]
Xtr, ytr = trB.drop(columns="y"), trB["y"]

# B1. 가수분해 커널 OLS (채택 물리모델)
A = np.c_[np.ones(len(trB)), trB[["H"]].values]
cb = np.linalg.lstsq(A, ytr.values, rcond=None)[0]
predB1 = pd.Series(cb[0] + cb[1] * DB["H"], index=DB.index)

# B2. RandomForest
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=-1)
rf.fit(Xtr, ytr)
predB2 = pd.Series(rf.predict(DB.drop(columns="y")), index=DB.index)

# B3. HistGradientBoosting
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4, random_state=0)
gbm.fit(Xtr, ytr)
predB3 = pd.Series(gbm.predict(DB.drop(columns="y")), index=DB.index)

# B4. NNLS 스태킹 (검증 2022로 가중 추정 → 평가 2023 적용)
base = {"hyd": predB1, "rf": predB2, "gbm": predB3}
Pva = np.c_[[base[k].reindex(vaB.index).values for k in base]].T
w, _ = nnls(Pva, vaB["y"].values)
w = w / w.sum() if w.sum() > 0 else w
predB4 = sum(wv * base[k] for wv, k in zip(w, base))

frameB = DB[["y"]]
trackB = [
    evaluate("가수분해 커널 OLS", predB1, frameB),
    evaluate("RandomForest", predB2, frameB),
    evaluate("GBM (HistGradientBoosting)", predB3, frameB),
    evaluate("NNLS 앙상블 (물리+RF+GBM)", predB4, frameB),
]
weights = {k: round(float(wv), 3) for k, wv in zip(base, w)}

# RF 특징 중요도
imp = sorted(zip(Xtr.columns, rf.feature_importances_), key=lambda x: -x[1])
importances = [{"f": f, "v": round(float(v), 3)} for f, v in imp]

# 앙상블 시계열 (차트용)
ens_series = predB4.reindex(M.index)

out = {
    "trackA": trackA,
    "trackB": trackB,
    "ensemble_weights": weights,
    "importances": importances,
    "ens_pred": [None if pd.isna(v) else round(float(v)) for v in ens_series.values],
    "ens_dates_match": len(ens_series) == len(M),
}
with open("outputs/ml_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("=== Track A (시계열, 1일 지평) ===")
for r in trackA:
    print(f"{r['name']:32s} valid R2={r['valid']['R2']:.3f} skill={r['valid']['skill']:+.3f} | test R2={r['test']['R2']:.3f} skill={r['test']['skill']:+.3f}")
print("=== Track B (기질 기반, 주간 계획) ===")
for r in trackB:
    print(f"{r['name']:32s} valid R2={r['valid']['R2']:.3f} | test R2={r['test']['R2']:.3f} RMSE={r['test']['RMSE']}")
print("NNLS weights:", weights)
print("RF importances:", importances[:6])
