"""외생변수 조합 비교 — 부하(VS_in) · 산성화(VFA/Alk) · 수리(유량) 안 검정.

제안된 3종은 물리적으로 중복이 적다. 실제 상관을 보면
  VS_in ↔ pool r=0.21,  VFA/Alk ↔ pool r=-0.09  → 독립적
  투입량합계 ↔ S_fast r=0.993                    → 사실상 동일 변수
따라서 유량 포함/제외를 나눠 재고, 기존 최선(온도·VFA·CODcr)과 나란히 둔다.

미래 가용성 구분
  known : 제어입력이라 계획을 아는 변수 → 실제 미래값 사용
  held  : 계측값이라 모르는 변수        → 원점 최신값 고정
실행: `python -m src.arimax_load`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.final_ensemble import TARGET, prepare

HORIZONS = [1, 3, 5, 7, 10, 14]
ORIGIN_STEP = 4
FOLDS = [("1 (단절 이전)", 1100, 1500), ("2 (단절 걸침)", 1400, 1800),
         ("3 (단절 관통)", 1700, 2100)]
POOLS = ["S_fast", "S_slow"]

# 이름 → (known 컬럼, held 컬럼)
VARIANTS = {
    "pool만 (기준)":            (POOLS, []),
    "pool+온도·VFA·COD (기존)": (POOLS, ["소화조_온도", "소화조_VFA", "소화조_CODcr"]),
    "pool+VSin·VFAALK·유량":    (POOLS + ["투입량합계"], ["VS_in", "VFA_ALK"]),
    "pool+VSin·VFAALK":         (POOLS, ["VS_in", "VFA_ALK"]),
    "VSin·VFAALK·유량 (pool 제외)": (["투입량합계"], ["VS_in", "VFA_ALK"]),
    "pool+VSin":                (POOLS, ["VS_in"]),
    "pool+VFAALK":              (POOLS, ["VFA_ALK"]),
}
ALL = sorted({c for k, h in VARIANTS.values() for c in k + h})


def frames():
    d = prepare()
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    endog = d[TARGET].astype(float).reset_index(drop=True)
    X = d[ALL].ffill().bfill().reset_index(drop=True)
    return endog, (X - X.mean()) / X.std()


def score(endog, X, known, held, cut, end):
    cols = known + held
    ex = X[cols] if cols else None
    fit = SARIMAX(endog.iloc[:cut], exog=ex.iloc[:cut] if cols else None,
                  order=(1, 0, 1), trend="c", enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    P = {h: [] for h in HORIZONS}; A = {h: [] for h in HORIZONS}; B = {h: [] for h in HORIZONS}
    for i in range(cut, min(end, len(endog)) - max(HORIZONS), ORIGIN_STEP):
        past = endog.iloc[:i].dropna()
        if not len(past):
            continue
        r = fit.apply(endog.iloc[:i], exog=ex.iloc[:i] if cols else None, refit=False)
        if cols:
            fx = ex.iloc[i:i + max(HORIZONS)].copy()
            for c in held:                       # 계측값은 원점에서 동결
                fx[c] = ex[c].iloc[i - 1]
        else:
            fx = None
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
        for name, (k, hd) in VARIANTS.items():
            r = score(endog, X, k, hd, cut, end)
            pers = pers or {h: v[1] for h, v in r.items()}
            print(f"   {name:28s} " + " ".join(f"h{h}={v[0]:7.3f}" for h, v in sorted(r.items())))
            for h, v in r.items():
                rows.append(dict(fold=tag, variant=name, h=h, R2=v[0], persistence=v[1]))
        print(f"   {'persistence':28s} " + " ".join(f"h{h}={v:7.3f}" for h, v in sorted(pers.items())))
        print()
    pd.DataFrame(rows).to_csv("outputs/arimax_load.csv", index=False, encoding="utf-8-sig")
    print("[저장] outputs/arimax_load.csv")


if __name__ == "__main__":
    main()
