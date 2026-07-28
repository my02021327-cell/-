"""ARIMAX 다지평 성능 — 누수 없는 재측정.

기존 `horizon_probe.py` 는 전체 계열로 적합한 결과객체를 `.apply(refit=False)`
로 재사용했다. 파라미터가 테스트 구간을 본 것이므로 다지평 수치가 부풀려졌다.
여기서는 폴드마다 **학습구간만으로 재적합**하고, 예측 시점까지의 관측으로
상태만 갱신한다(칼만 필터). 달력일 격자를 써서 h 를 달력일로 정의하고
`horizon_ensemble.py` 와 직접 비교 가능하게 한다.
실행: `python -m src.arimax_horizon`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.final_ensemble import SARIMAX_EXOG, TARGET, prepare

HORIZONS = [1, 3, 5, 7, 10, 14]
ORIGIN_STEP = 3          # 원점 간격(일)
FOLDS = [(1100, 1500), (1400, 1800), (1700, 2100)]   # (학습끝, 평가끝) 달력일 인덱스


def main():
    d = prepare()
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    endog = d[TARGET].astype(float).reset_index(drop=True)      # 결측 포함 달력일 격자
    exog = d[SARIMAX_EXOG].ffill().bfill().reset_index(drop=True)
    print(f"달력일 {len(endog)}일, 메탄 실측 {endog.notna().sum()}일\n")

    rows = []
    for h in HORIZONS:
        r2s, pers_r2 = [], []
        for cut, end in FOLDS:
            fit = SARIMAX(endog.iloc[:cut], exog=exog.iloc[:cut], order=(1, 0, 1),
                          trend="c", enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)   # 학습구간만
            P, A, B = [], [], []
            for i in range(cut, min(end, len(endog)) - h, ORIGIN_STEP):
                act = endog.iloc[i + h - 1]
                last = endog.iloc[:i].dropna()
                if np.isnan(act) or not len(last):
                    continue
                r = fit.apply(endog.iloc[:i], exog=exog.iloc[:i], refit=False)  # 파라미터 고정
                P.append(r.get_forecast(steps=h, exog=exog.iloc[i:i + h]).predicted_mean.iloc[-1])
                A.append(act)
                B.append(last.iloc[-1])                 # 같은 지평 persistence
            if len(A) > 10:
                r2s.append(r2_score(A, P)); pers_r2.append(r2_score(A, B))
        rows.append(dict(h=h, folds=len(r2s), ARIMAX=np.mean(r2s), ARIMAX_std=np.std(r2s),
                         persistence=np.mean(pers_r2), gain=np.mean(r2s) - np.mean(pers_r2)))
        r = rows[-1]
        print(f"h={h:2d}일 {r['folds']}폴드 | ARIMAX R2={r['ARIMAX']:7.4f}±{r['ARIMAX_std']:.4f}"
              f" | persistence={r['persistence']:7.4f} | 이득={r['gain']:+.4f}")
    t = pd.DataFrame(rows)
    t.to_csv("outputs/arimax_horizon.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] outputs/arimax_horizon.csv")
    return t


if __name__ == "__main__":
    main()
