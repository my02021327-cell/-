"""ARIMAX 외생변수에 이화학을 추가 — 과거 메탄 × 이화학 × HRT 결합.

미래 시점의 이화학 값은 알 수 없다. 따라서 예측 구간에서는 **원점의 최신
관측값을 그대로 유지**한다(운전자가 실제로 가진 정보와 동일). 투입 pool 만
제어입력이므로 실제 미래값을 쓴다.

누수 방지: 폴드마다 학습구간으로만 재적합하고, 칼만 상태는 원점까지의
관측으로만 갱신한다.
실행: `python -m src.arimax_chem`
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
FOLDS = [(1100, 1500), (1400, 1800), (1700, 2100)]

POOLS = ["S_fast", "S_slow"]
# 다중공선성 정리 : MAlk(TAlk 중복), VFA_ALK(파생) 제외
CHEM = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_VFA", "소화조_TAlk", "소화조_CODcr"]
CHEM_CORE = ["소화조_온도", "소화조_VFA", "소화조_CODcr"]

VARIANTS = {
    "pool만 (기준)": (POOLS, []),
    "pool + 이화학 6종": (POOLS, CHEM),
    "pool + 이화학 3종": (POOLS, CHEM_CORE),
    "이화학만": ([], CHEM),
}


def frames():
    d = prepare()
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    endog = d[TARGET].astype(float).reset_index(drop=True)
    X = d[POOLS + CHEM].copy()
    X[CHEM] = X[CHEM].ffill()                    # 관측 시점까지의 최신값
    X = X.ffill().bfill().reset_index(drop=True)
    # 스케일 정규화 — SARIMAX 수치 안정
    X = (X - X.mean()) / X.std()
    return endog, X


def run_variant(endog, X, pool_cols, chem_cols):
    cols = pool_cols + chem_cols
    per_h = {h: ([], []) for h in HORIZONS}      # (pred, actual)
    pers_h = {h: [] for h in HORIZONS}
    for cut, end in FOLDS:
        ex = X[cols] if cols else None
        fit = SARIMAX(endog.iloc[:cut], exog=ex.iloc[:cut] if cols else None,
                      order=(1, 0, 1), trend="c", enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        hi = min(end, len(endog)) - max(HORIZONS)
        for i in range(cut, hi, ORIGIN_STEP):
            past = endog.iloc[:i].dropna()
            if not len(past):
                continue
            r = fit.apply(endog.iloc[:i], exog=ex.iloc[:i] if cols else None, refit=False)
            if cols:
                fx = ex.iloc[i:i + max(HORIZONS)].copy()
                # 이화학은 원점 최신값 고정, pool 은 실제 미래값 사용
                for c in chem_cols:
                    fx[c] = ex[c].iloc[i - 1]
            else:
                fx = None
            fc = r.get_forecast(steps=max(HORIZONS), exog=fx).predicted_mean.values
            for h in HORIZONS:
                act = endog.iloc[i + h - 1]
                if np.isnan(act):
                    continue
                per_h[h][0].append(fc[h - 1]); per_h[h][1].append(act)
                pers_h[h].append(past.iloc[-1])
    return {h: (r2_score(a, p), r2_score(a, pers_h[h]), len(a))
            for h, (p, a) in per_h.items() if len(a) > 20}


def main():
    endog, X = frames()
    print(f"달력일 {len(endog)}일, 메탄 실측 {int(endog.notna().sum())}일")
    print(f"이화학 {len(CHEM)}종, 예측구간은 원점 최신값 고정\n")
    rows = []
    for name, (p, c) in VARIANTS.items():
        res = run_variant(endog, X, p, c)
        line = "  ".join(f"h{h}={v[0]:.3f}" for h, v in sorted(res.items()))
        print(f"{name:18s} {line}")
        for h, (r2, pers, n) in res.items():
            rows.append(dict(variant=name, h=h, R2=r2, persistence=pers, n=n))
    t = pd.DataFrame(rows)
    print("\n── persistence 대비 이득 ──")
    piv = t.pivot(index="variant", columns="h", values="R2")
    per = t.groupby("h")["persistence"].first()
    print((piv - per).round(3).to_string())
    print("\npersistence:", {h: round(v, 3) for h, v in per.items()})
    t.to_csv("outputs/arimax_chem.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] outputs/arimax_chem.csv")


if __name__ == "__main__":
    main()
