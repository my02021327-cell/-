"""두 타깃(소화가스 / 메탄)을 확정 분할 위에서 각각 모델링한다.

· 데이터: src.dataset (A/B 건강계열 선택 적용, 계측 고장 보정됨)
· 분할  : src.splits 의 9폴드 롤링 오리진 (개발구간 2018~2022)
          2023 봉인 홀드아웃은 여기서 **열지 않는다**
· 평가  : 같은 지평 persistence 대비 이득

외생 구분
  known : 제어입력(투입 유량 pool) → 실제 미래값
  held  : 계측 파생 → 예측 원점에서 동결
실행: `python -m src.model_both`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.dataset import build
from src.splits import DATA_START, cv_folds

HORIZONS = [1, 3, 7, 14]
ORIGIN_STEP = 4
TARGETS = ["biogas", "methane"]

# 이름 → (known, held)
VARIANTS = {
    "유량pool":              (["S_fast", "S_slow"], []),
    "유량pool+VSin":         (["S_fast", "S_slow"], ["VS_in"]),
    "부하pool":              ([], ["L_fast", "L_slow"]),
    "부하pool+유량pool":      (["S_fast", "S_slow"], ["L_fast", "L_slow"]),
    "유량pool+이화학":         (["S_fast", "S_slow"], ["dig_온도", "dig_VFA", "dig_Alk", "dig_pH"]),
    "부하pool+이화학":         ([], ["L_fast", "L_slow", "dig_온도", "dig_VFA", "dig_Alk", "dig_pH"]),
    "유량pool+VSin+VFAALK":  (["S_fast", "S_slow"], ["VS_in", "VFA_ALK"]),
}
ALL = sorted({c for k, h in VARIANTS.values() for c in k + h})


def prep():
    g = build().loc[DATA_START:]
    X = g[ALL].ffill().bfill()
    X = (X - X.mean()) / X.std()
    return g, X


def run(g, X, target, known, held, folds):
    y = g[target].astype(float)
    cols = known + held
    idx = g.index
    P = {h: [] for h in HORIZONS}; A = {h: [] for h in HORIZONS}; B = {h: [] for h in HORIZONS}
    for tr_end, te_a, te_b in folds:
        cut = idx.get_indexer([tr_end], method="nearest")[0] + 1
        end = idx.get_indexer([te_b], method="nearest")[0] + 1
        yv, xv = y.reset_index(drop=True), X.reset_index(drop=True)
        try:
            fit = SARIMAX(yv.iloc[:cut], exog=xv[cols].iloc[:cut], order=(1, 0, 1),
                          trend="c", enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
        except Exception:                                     # noqa: BLE001
            continue
        for i in range(cut, min(end, len(yv)) - max(HORIZONS), ORIGIN_STEP):
            past = yv.iloc[:i].dropna()
            if not len(past):
                continue
            r = fit.apply(yv.iloc[:i], exog=xv[cols].iloc[:i], refit=False)
            fx = xv[cols].iloc[i:i + max(HORIZONS)].copy()
            for c in held:
                fx[c] = xv[c].iloc[i - 1]
            fc = r.get_forecast(steps=max(HORIZONS), exog=fx).predicted_mean.values
            for h in HORIZONS:
                a = yv.iloc[i + h - 1]
                if not np.isnan(a):
                    P[h].append(fc[h - 1]); A[h].append(a); B[h].append(past.iloc[-1])
    return {h: (r2_score(A[h], P[h]), r2_score(A[h], B[h]), len(A[h]))
            for h in HORIZONS if len(A[h]) > 30}


def main():
    g, X = prep()
    folds = cv_folds()
    print(f"데이터 {len(g)}일 | 폴드 {len(folds)}개 | 2023 봉인 홀드아웃 미개봉\n")
    rows = []
    for tgt in TARGETS:
        n = g[tgt].notna().sum()
        print("=" * 78)
        print(f"타깃: {tgt}  (유효 {n}일, 평균 {g[tgt].mean():.0f})")
        print("=" * 78)
        pers = None
        for name, (k, h) in VARIANTS.items():
            r = run(g, X, tgt, k, h, folds)
            if not r:
                print(f"  {name:22s} — 실패"); continue
            pers = pers or {hh: v[1] for hh, v in r.items()}
            print(f"  {name:22s} " + " ".join(f"h{hh}={v[0]:7.4f}" for hh, v in sorted(r.items())))
            for hh, v in r.items():
                rows.append(dict(target=tgt, variant=name, h=hh, R2=v[0],
                                 persistence=v[1], n=v[2]))
        print(f"  {'persistence':22s} " + " ".join(f"h{hh}={v:7.4f}" for hh, v in sorted(pers.items())))
        print()
    t = pd.DataFrame(rows)
    t.to_csv("outputs/model_both.csv", index=False, encoding="utf-8-sig")
    print("== persistence 대비 이득 ==")
    for tgt in TARGETS:
        s = t[t.target == tgt]
        piv = s.pivot_table(index="variant", columns="h", values="R2")
        per = s.groupby("h")["persistence"].first()
        print(f"\n[{tgt}]")
        print((piv - per).round(3).sort_values(by=14, ascending=False).to_string())
    print("\n[저장] outputs/model_both.csv")


if __name__ == "__main__":
    main()
