"""충격부하 구간에서 모델이 변화를 따라가는가.

R2 는 정상 구간이 지배하므로 충격 성능을 가린다. 여기서는 예측을 충격/정상으로
갈라 조건부로 평가한다.

구조적 구분 (확정안 exog = S_fast, S_slow, VS_in)
  · 유량 충격 : pool 은 제어입력이라 미래값을 쓴다 → 모델이 **본다**
  · 농도 충격 : VS_in 은 계측이라 원점 동결 → 모델이 **못 본다**
두 충격을 따로 정의해 각각 추적 성능을 잰다.

실행: `python -m src.shock_test`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

import src.vs_features as vf
from src.final_ensemble import TARGET, prepare

HORIZONS = [1, 3, 5, 7, 14]
ORIGIN_STEP = 2
FOLDS = [(1100, 1500), (1400, 1800), (1700, 2100)]
EXOG = ["S_fast", "S_slow", "VS_in"]
KNOWN = ["S_fast", "S_slow"]          # 제어입력 → 실제 미래값
HELD = ["VS_in"]                      # 계측 → 원점 동결
Q_HI, Q_LO = 0.90, 0.10               # 충격 판정 분위


def collect():
    d = vf.build(prepare())
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    # 충격 지표 : 유량(제어) / 부하(농도 포함)
    d = vf.add_load_features(d, "투입량합계", prefix="Q")
    endog = d[TARGET].astype(float).reset_index(drop=True)
    X = d[EXOG].ffill().bfill().reset_index(drop=True)
    X = (X - X.mean()) / X.std()
    q_surge = d["Q_SURGE"].reset_index(drop=True)
    v_surge = d["VS_in_SURGE"].reset_index(drop=True)

    recs = []
    for cut, end in FOLDS:
        fit = SARIMAX(endog.iloc[:cut], exog=X.iloc[:cut], order=(1, 0, 1), trend="c",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        for i in range(cut, min(end, len(endog)) - max(HORIZONS), ORIGIN_STEP):
            past = endog.iloc[:i].dropna()
            if not len(past):
                continue
            r = fit.apply(endog.iloc[:i], exog=X.iloc[:i], refit=False)
            fx = X.iloc[i:i + max(HORIZONS)].copy()
            for c in HELD:
                fx[c] = X[c].iloc[i - 1]
            fc = r.get_forecast(steps=max(HORIZONS), exog=fx).predicted_mean.values
            for h in HORIZONS:
                j = i + h - 1
                a = endog.iloc[j]
                if np.isnan(a):
                    continue
                recs.append(dict(h=h, pred=fc[h - 1], act=a, persist=past.iloc[-1],
                                 q_surge=q_surge.iloc[j], v_surge=v_surge.iloc[j],
                                 base=past.iloc[-1]))
    return pd.DataFrame(recs)


def band(s, hi, lo):
    return np.where(s >= hi, "충격(과부하)", np.where(s <= lo, "충격(저부하)", "정상"))


def report(t: pd.DataFrame, key: str, title: str):
    hi, lo = t[key].quantile(Q_HI), t[key].quantile(Q_LO)
    t = t.assign(grp=band(t[key], hi, lo))
    print(f"\n{'='*74}\n{title}  (상위{Q_HI:.0%} ≥ {hi:.3f} / 하위{Q_LO:.0%} ≤ {lo:.3f})\n{'='*74}")
    print(f"{'지평':>4s} {'구간':12s} {'n':>5s} {'MAE':>7s} {'persist MAE':>11s} "
          f"{'개선율':>7s} {'편의(bias)':>10s} {'방향적중':>8s}")
    for h in HORIZONS:
        s = t[t.h == h]
        for g in ["정상", "충격(과부하)", "충격(저부하)"]:
            u = s[s.grp == g]
            if len(u) < 15:
                continue
            mae = mean_absolute_error(u.act, u.pred)
            pmae = mean_absolute_error(u.act, u.persist)
            bias = (u.pred - u.act).mean()
            # 방향 적중 : 기준값(원점 최신 실측) 대비 증감 방향
            dp, da = u.pred - u.base, u.act - u.base
            hit = ((dp > 0) == (da > 0)).mean() * 100
            print(f"{h:4d} {g:12s} {len(u):5d} {mae:7.0f} {pmae:11.0f} "
                  f"{(1-mae/pmae)*100:6.1f}% {bias:+10.0f} {hit:7.1f}%")
        print()


def main():
    t = collect()
    print(f"예측 표본 {len(t)}건 ({len(HORIZONS)}지평 × 3폴드)")
    report(t, "q_surge", "① 유량 충격 — 모델이 미래 유량을 안다(제어입력)")
    report(t, "v_surge", "② 부하 충격 — VS_in 은 원점 동결이라 농도 변화는 못 본다")
    t.to_csv("outputs/shock_test.csv", index=False, encoding="utf-8-sig")
    print("[저장] outputs/shock_test.csv")


if __name__ == "__main__":
    main()
