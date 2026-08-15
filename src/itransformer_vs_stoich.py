# -*- coding: utf-8 -*-
"""§2.8 화학양론·잔존율 모델을 iTransformer 검정과 **동일한 평가일**에서 재채점.

신경망은 30일 창의 특징 11개가 모두 존재하는 날만 표본으로 삼으므로 평가 n이 다르다.
서로 다른 표본에서 잰 R² 를 나란히 놓으면 비교가 성립하지 않는다.
여기서는 잔존율 모델의 예측을 같은 날짜 집합으로 잘라 다시 채점한다.
"""
import json

import numpy as np
import pandas as pd

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
KS = {"foodww": .40, "manure": .08, "food": .30}
SRT, TAU0, TMIX = 12.0, 0, 3.0
TR = ("2018-01-01", "2021-12-31"); VA = ("2022-01-01", "2022-12-31")
TE = ("2023-01-01", "2023-09-17")

feed = M.feed_AB_tpd.ffill(limit=2)
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = list(KS); I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)
y = M.biogas_AB_m3d * (M.CH4_pct.ffill(limit=3) / 100)


def conv(s, w):
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


w = np.exp(-np.arange(41) / TMIX); w /= w.sum()
Rm = pd.DataFrame({c: conv(R[c].ffill(limit=7), w).shift(TAU0) for c in KS})
Rm = Rm.div(Rm.sum(axis=1), axis=0)
tau = np.arange(301)
X = pd.DataFrame({k: conv((feed * Rm[k]).ffill(limit=3),
                          KS[k] * np.exp(-(1 / SRT + KS[k]) * tau)) for k in KS})
D = pd.concat([X, y.rename("y")], axis=1).dropna()
th, *_ = np.linalg.lstsq(D.loc[TR[0]:TR[1]][list(KS)].values, D.loc[TR[0]:TR[1]]["y"].values, rcond=None)
pred = pd.Series(D[list(KS)].values @ th, index=D.index)

# ---------------------------------------------------------------- 동일 평가일 산출
M2 = M.copy()
M2["dig_pH"] = M2[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
M2["dig_TS"] = M2[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)
M2["dig_VS"] = M2[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)
M2["dig_CODcr"] = M2[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
EXOG = ["acid_pH", "acid_TS_pct", "acid_VS_pct", "acid_CODcr_mgL", "dig_pH", "VFA_A_mgL",
        "ALK_A_mgL", "dig_TS", "dig_VS", "dig_CODcr", "feed_AB_tpd"]
Xf = M2[EXOG].ffill(limit=7)
yt = M2.CH4_m3d


def sample_dates(L, H, use_hist):
    cols = EXOG + (["CH4_m3d"] if use_hist else [])
    A = pd.concat([Xf, yt.ffill(limit=7).rename("CH4_m3d")], axis=1)[cols].values
    v, idx, out = yt.values, M2.index, []
    for t in range(L - 1, len(M2) - H):
        j = t + H
        if np.isfinite(v[j]) and np.isfinite(A[t - L + 1: t + 1]).all():
            out.append(idx[j])
    return pd.DatetimeIndex(out)


def met(t, p):
    return {"R2": round(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4),
            "RMSE": round(float(np.sqrt(((t - p) ** 2).mean())), 1),
            "MSE": round(float(((t - p) ** 2).mean()), 1),
            "MAPE": round(float(np.abs((t - p) / t).mean() * 100), 2),
            "ACC": round(float(100 * (1 - np.abs((t - p) / t).mean())), 2), "n": int(len(t))}


P = "outputs/itransformer_results.json"
J = json.load(open(P, encoding="utf-8"))
print("§2.8 화학양론·잔존율 모델 (파라미터 3개) — iTransformer 검정과 동일 평가일 기준")
for k, r in J["results"].items():
    setting, Hs = k.split("|")
    ds = sample_dates(30, int(Hs[1:]), setting.startswith("A"))
    d = pd.concat([y.rename("y"), pred.rename("p")], axis=1).loc[ds].dropna()
    e = {nm: met(d.loc[a:b, "y"].values, d.loc[a:b, "p"].values)
         for nm, (a, b) in [("valid", VA), ("test", TE)]}
    e["params"] = 3
    r["화학양론·잔존율(§2.8)"] = e
    pm = r["persistence"]["test"]["MSE"]
    r["skill"] = {n: round(1 - v["test"]["MSE"] / pm, 4)
                  for n, v in r.items() if isinstance(v, dict) and "test" in v}
    print(f"  [{k}]  사고학습 R² {e['valid']['R2']:+.4f}  평가 R² {e['test']['R2']:+.4f}  "
          f"MAPE {e['test']['MAPE']:.2f}%  skill {r['skill']['화학양론·잔존율(§2.8)']:+.4f}  n={e['test']['n']}")

json.dump(J, open(P, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\n갱신: {P}")
print("\n=== 최종 순위 (평가구간 skill score) ===")
for k, r in J["results"].items():
    print(f"  [{k}]")
    for n, v in sorted(r["skill"].items(), key=lambda x: -x[1]):
        print(f"     {n:24s} {v:+.4f}{'   ← persistence 미달' if v < 0 else ''}")
