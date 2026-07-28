"""동결(freeze) 대 오라클(실제 미래값) — 열화의 원인을 특정한다.

관측된 패턴: 계측 파생 exog 는 h=1 에서 멀쩡하고 h 가 커질수록 무너진다.
가설: 원인은 이상치도 비선형도 아니라 **동결된 값의 진부화**다.
      미래를 모르는 변수를 원점 값으로 고정하면, 지평이 멀어질수록 그 값이
      틀려지고 계수가 그 오차를 증폭한다. 계수가 클수록(= 유용할수록) 더 나쁘다.

검정: 같은 구성을 (a) 원점 동결, (b) 실제 미래값(오라클)로 돌린다.
      오라클에서 성능이 회복되면 진부화가 원인으로 확정된다.
      오라클은 배치 가능한 모델이 아니라 진단용 상한선이다.
실행: `python -m src.oracle_test`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.model_robust import prep
from src.splits import cv_folds

HORIZONS = [1, 3, 7, 14]
ORIGIN_STEP = 4
CASES = {
    "유량pool (전부 known)":  (["S_fast", "S_slow"], []),
    "+VSin":                (["S_fast", "S_slow"], ["VS_in"]),
    "+부하pool 로그":          (["S_fast", "S_slow"], ["Lfast_log", "Lslow_log"]),
    "+이화학":                (["S_fast", "S_slow"], ["dig_온도", "dig_VFA", "dig_Alk"]),
}


def run(g, X, target, known, held, folds, oracle: bool):
    y = g[target].astype(float)
    cols = known + held
    idx = g.index
    P = {h: [] for h in HORIZONS}; A = {h: [] for h in HORIZONS}; B = {h: [] for h in HORIZONS}
    yv, xv = y.reset_index(drop=True), X.reset_index(drop=True)
    for tr_end, _, te_b in folds:
        cut = idx.get_indexer([tr_end], method="nearest")[0] + 1
        end = idx.get_indexer([te_b], method="nearest")[0] + 1
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
            if not oracle:
                for c in held:
                    fx[c] = xv[c].iloc[i - 1]
            fc = r.get_forecast(steps=max(HORIZONS), exog=fx).predicted_mean.values
            for h in HORIZONS:
                a = yv.iloc[i + h - 1]
                if not np.isnan(a):
                    P[h].append(fc[h - 1]); A[h].append(a); B[h].append(past.iloc[-1])
    return {h: (r2_score(A[h], P[h]), r2_score(A[h], B[h]))
            for h in HORIZONS if len(A[h]) > 30}


def main():
    g = prep()
    cols = sorted({c for k, h in CASES.values() for c in k + h})
    X = g[cols].ffill().bfill()
    X = (X - X.mean()) / X.std()
    folds = cv_folds()
    rows = []
    for tgt in ["biogas", "methane"]:
        print("=" * 78)
        print(f"타깃: {tgt}   (동결 = 배치 가능 / 오라클 = 진단용 상한)")
        print("=" * 78)
        pers = None
        for name, (k, h) in CASES.items():
            fr = run(g, X, tgt, k, h, folds, oracle=False)
            orc = run(g, X, tgt, k, h, folds, oracle=True) if h else fr
            pers = pers or {hh: v[1] for hh, v in fr.items()}
            print(f"  {name:20s} 동결  " + " ".join(f"h{hh}={v[0]:7.4f}" for hh, v in sorted(fr.items())))
            if h:
                print(f"  {'':20s} 오라클 " + " ".join(f"h{hh}={v[0]:7.4f}" for hh, v in sorted(orc.items())))
            for hh in fr:
                rows.append(dict(target=tgt, case=name, h=hh, frozen=fr[hh][0],
                                 oracle=orc[hh][0], persistence=fr[hh][1]))
        print(f"  {'persistence':20s}       " + " ".join(f"h{hh}={v:7.4f}" for hh, v in sorted(pers.items())))
        print()
    pd.DataFrame(rows).to_csv("outputs/oracle_test.csv", index=False, encoding="utf-8-sig")
    print("[저장] outputs/oracle_test.csv")


if __name__ == "__main__":
    main()
