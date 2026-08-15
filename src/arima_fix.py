# -*- coding: utf-8 -*-
"""ARIMA 기준선의 시점 정렬 오류 정정.

■ 무엇이 틀렸나
   itransformer_yeongcheon.py 초판의 arima() 는 관측일 시퀀스를 등간격으로 간주하고
   직전 관측일까지의 이력으로 H스텝을 예측한 뒤 그 값을 **당일**에 대응시켰다.
   H=1에서는 맞지만 H=7에서는 사실상 '1일 전 정보로 만든 예측'을
   '7일 전 정보로 만든 persistence'와 겨루게 해 ARIMA에 유리하게 기울었다.

■ 어떻게 고쳤나
   달력일 기준으로 재정의한다. 목표일 d 에 대한 예측은 **d−H일까지의 정보만** 쓴다.
   결측을 NaN으로 둔 일별 계열에 상태공간 ARIMA를 적용하므로 관측 간격이
   불규칙해도 달력 정렬이 유지된다. 계수는 학습구간에서 한 번만 추정하고
   이후에는 refit 없이 필터만 갱신한다(재추정 시 평가구간 정보 유입 방지).
"""
import json

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
y = M.CH4_m3d.asfreq("D") if M.index.freq else M.CH4_m3d.reindex(
    pd.date_range(M.index.min(), M.index.max(), freq="D"))
TR = ("2018-01-01", "2021-12-31"); VA = ("2022-01-01", "2022-12-31")
TE = ("2023-01-01", "2023-09-17")


def metrics(t, p):
    return {"R2": round(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4),
            "RMSE": round(float(np.sqrt(((t - p) ** 2).mean())), 1),
            "MSE": round(float(((t - p) ** 2).mean()), 1),
            "MAPE": round(float(np.abs((t - p) / t).mean() * 100), 2),
            "ACC": round(float(100 * (1 - np.abs((t - p) / t).mean())), 2), "n": int(len(t))}


base = ARIMA(y.loc[:TR[1]], order=(2, 1, 2)).fit()
print(f"학습구간 계수 추정 완료 (AIC {base.aic:.1f})")

OUT = {}
for H in [1, 7]:
    rows = {}
    origins = pd.date_range(VA[0], TE[1], freq="D") - pd.Timedelta(days=H)
    for o in origins:
        tgt = o + pd.Timedelta(days=H)
        if not np.isfinite(y.get(tgt, np.nan)):
            continue
        r = base.apply(y.loc[:o], refit=False)
        rows[tgt] = float(r.forecast(H).iloc[-1])
    OUT[f"H{H}"] = pd.Series(rows)
    d = pd.concat([y.rename("y"), OUT[f"H{H}"].rename("p")], axis=1).dropna()
    m = {nm: metrics(d.loc[a:b, "y"].values, d.loc[a:b, "p"].values)
         for nm, (a, b) in [("valid", VA), ("test", TE)]}
    print(f"H={H}  (전체 관측일) 사고학습 R² {m['valid']['R2']:+.4f}  "
          f"평가 R² {m['test']['R2']:+.4f}  MAPE {m['test']['MAPE']:.2f}%  n={m['test']['n']}")

# ---------------------------------------------------------------- JSON 갱신
# 표본 정렬: ARIMA·persistence 는 특징 결측과 무관하게 계산되지만, 신경망은 30일 창의
# 11개 특징이 모두 있는 날만 표본으로 삼는다. skill score 를 같은 분모로 재려면
# 모든 모델을 **신경망과 동일한 날짜 집합**에서 채점해야 한다.
import importlib.util as _il

_sp = _il.spec_from_file_location("itr", "src/itransformer_yeongcheon.py")
P = "outputs/itransformer_results.json"
J = json.load(open(P, encoding="utf-8"))

MM = M.copy()
MM["dig_pH"] = MM[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
MM["dig_TS"] = MM[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)
MM["dig_VS"] = MM[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)
MM["dig_CODcr"] = MM[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
EXOG = ["acid_pH", "acid_TS_pct", "acid_VS_pct", "acid_CODcr_mgL", "dig_pH", "VFA_A_mgL",
        "ALK_A_mgL", "dig_TS", "dig_VS", "dig_CODcr", "feed_AB_tpd"]
Xdf = MM[EXOG].ffill(limit=7)
yt = MM.CH4_m3d


def sample_dates(L, H, use_hist):
    cols = EXOG + (["CH4_m3d"] if use_hist else [])
    A = pd.concat([Xdf, yt.ffill(limit=7).rename("CH4_m3d")], axis=1)[cols].values
    v, idx, out = yt.values, MM.index, []
    for t in range(L - 1, len(MM) - H):
        j = t + H
        if np.isfinite(v[j]) and np.isfinite(A[t - L + 1: t + 1]).all():
            out.append(idx[j])
    return pd.DatetimeIndex(out)


for k, r in J["results"].items():
    setting, Hs = k.split("|")
    H = int(Hs[1:])
    ds = sample_dates(30, H, setting.startswith("A"))
    d = pd.concat([yt.rename("y"), OUT[Hs].rename("p")], axis=1).loc[ds].dropna()
    r["ARIMA"] = {nm: metrics(d.loc[a:b, "y"].values, d.loc[a:b, "p"].values)
                  for nm, (a, b) in [("valid", VA), ("test", TE)]}
    r["ARIMA"]["sample_aligned"] = True
    pm = r["persistence"]["test"]["MSE"]
    r["skill"] = {n: round(1 - v["test"]["MSE"] / pm, 4)
                  for n, v in r.items() if isinstance(v, dict) and "test" in v}
J["arima_fix"] = {
    "issue": "초판은 관측일 시퀀스를 등간격 처리해 H=7에서 예측 원점이 목표일 1일 전에 놓였다",
    "fix": "달력일 기준으로 목표일 d의 예측에 d−H일까지의 정보만 사용. 계수는 학습구간 1회 추정 후 refit 없음",
    "issue2": "ARIMA·persistence 는 전체 관측일에서, 신경망은 특징 결측이 없는 날에서만 채점돼 분모가 달랐다",
    "fix2": "모든 모델을 신경망과 동일한 날짜 집합에서 채점",
    "before": {"H1_test_R2": 0.8770, "H7_test_R2": 0.6710},
    "after": {"H1_test_R2": J["results"]["A_논문동일_자기이력포함|H1"]["ARIMA"]["test"]["R2"],
              "H7_test_R2": J["results"]["A_논문동일_자기이력포함|H7"]["ARIMA"]["test"]["R2"]}}
json.dump(J, open(P, "w", encoding="utf-8"), ensure_ascii=False)

print("\n=== 정정 후 skill score (평가구간) ===")
for k, r in J["results"].items():
    print(f"  [{k}]")
    for n, v in sorted(r["skill"].items(), key=lambda x: -x[1]):
        print(f"     {n:22s} {v:+.4f}{'   ← persistence 미달' if v < 0 else ''}")
print(f"\n갱신: {P}")
