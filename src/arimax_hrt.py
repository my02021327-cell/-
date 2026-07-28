"""HRT 부하 파생(EMA·Surge)을 외생변수로 넣어 검정한다.

기준선은 앞서 확정된 pool + VS_in. 여기에 EMA(장기 수준)·SURGE(단기/장기 비율)
와 COD 기준 부하를 얹어 개선 여부를 본다.

SURGE 는 비율이라 계측 드리프트에 둔감하고(연도별 0.976~1.023),
pool 과의 상관이 0.09~0.17 로 거의 직교한다 — 정보를 더할 여지가 있다.
실행: `python -m src.arimax_hrt`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

import src.vs_features as vf
from src.final_ensemble import TARGET, prepare

HORIZONS = [1, 3, 5, 7, 10, 14]
ORIGIN_STEP = 4
FOLDS = [("1 단절이전", 1100, 1500), ("2 단절걸침", 1400, 1800), ("3 단절관통", 1700, 2100)]
POOLS = ["S_fast", "S_slow"]

VARIANTS = {
    "pool+VSin (현 확정안)":       ["VS_in"],
    "pool+VSin+EMA":              ["VS_in", "VS_in_EMA"],
    "pool+VSin+SURGE":            ["VS_in", "VS_in_SURGE"],
    "pool+VSin+EMA+SURGE (원안)":  ["VS_in", "VS_in_EMA", "VS_in_SURGE"],
    "pool+EMA+SURGE (VSin 제외)":  ["VS_in_EMA", "VS_in_SURGE"],
    "pool+CODema+CODsurge":       ["COD_in_EMA", "COD_in_SURGE"],
    "pool+VSin+CODsurge":         ["VS_in", "COD_in_SURGE"],
    "pool+VSin+SURGE+유입pH":      ["VS_in", "VS_in_SURGE", "유입_pH"],
    "pool만":                     [],
}
ALL = sorted({c for v in VARIANTS.values() for c in v})


def frames():
    d = vf.build(prepare())
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    endog = d[TARGET].astype(float).reset_index(drop=True)
    X = d[POOLS + ALL].ffill().bfill().reset_index(drop=True)
    return endog, (X - X.mean()) / X.std()


def score(endog, X, held, cut, end):
    cols = POOLS + held
    ex = X[cols]
    fit = SARIMAX(endog.iloc[:cut], exog=ex.iloc[:cut], order=(1, 0, 1), trend="c",
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    P = {h: [] for h in HORIZONS}; A = {h: [] for h in HORIZONS}; B = {h: [] for h in HORIZONS}
    for i in range(cut, min(end, len(endog)) - max(HORIZONS), ORIGIN_STEP):
        past = endog.iloc[:i].dropna()
        if not len(past):
            continue
        r = fit.apply(endog.iloc[:i], exog=ex.iloc[:i], refit=False)
        fx = ex.iloc[i:i + max(HORIZONS)].copy()
        for c in held:                      # 계측 파생은 원점에서 동결
            fx[c] = ex[c].iloc[i - 1]
        fc = r.get_forecast(steps=max(HORIZONS), exog=fx).predicted_mean.values
        for h in HORIZONS:
            a = endog.iloc[i + h - 1]
            if not np.isnan(a):
                P[h].append(fc[h - 1]); A[h].append(a); B[h].append(past.iloc[-1])
    return {h: (r2_score(A[h], P[h]), r2_score(A[h], B[h]))
            for h in HORIZONS if len(A[h]) > 20}


def main():
    endog, X = frames()
    rows = []
    for tag, cut, end in FOLDS:
        print(f"── 폴드 {tag} ──")
        pers = None
        for name, held in VARIANTS.items():
            r = score(endog, X, held, cut, end)
            pers = pers or {h: v[1] for h, v in r.items()}
            print(f"   {name:28s} " + " ".join(f"h{h}={v[0]:7.3f}" for h, v in sorted(r.items())))
            for h, v in r.items():
                rows.append(dict(fold=tag, variant=name, h=h, R2=v[0], persistence=v[1]))
        print(f"   {'persistence':28s} " + " ".join(f"h{h}={v:7.3f}" for h, v in sorted(pers.items())))
        print()
    t = pd.DataFrame(rows)
    t.to_csv("outputs/arimax_hrt.csv", index=False, encoding="utf-8-sig")
    piv = t.groupby(["variant", "h"])["R2"].mean().unstack()
    per = t.groupby("h")["persistence"].mean()
    print("== 3폴드 평균 R2 ==")
    print(piv.round(3).to_string())
    print("\n== persistence 대비 이득 ==")
    print((piv - per).round(3).to_string())
    print("\npersistence:", {h: round(v, 3) for h, v in per.items()})
    print("\n== 폴드별 최저 R2 (강건성) ==")
    print(t.groupby("variant")["R2"].min().round(3).sort_values(ascending=False).to_string())
    print("\n[저장] outputs/arimax_hrt.csv")


if __name__ == "__main__":
    main()
