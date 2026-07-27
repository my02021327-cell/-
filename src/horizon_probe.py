"""ARIMAX 가 실제로 무엇을 보고 예측하는지 분해한다.

"1-step R2 0.90" 이 모델의 실력인지 소화조 자기상관인지를 가리기 위해
(1) 외생 유무 비교, (2) 외생성분(Xb) 단독 설명력, (3) 지평 h 별 성능을 측정한다.
실행: `python -m src.horizon_probe`
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error
from src.final_ensemble import prepare, SARIMAX_EXOG, TARGET, HOLDOUT_YEAR

df = prepare()
d = df.dropna(subset=[TARGET]).copy()
endog = d[TARGET].astype(float).reset_index(drop=True)
exog  = d[SARIMAX_EXOG].ffill().bfill().reset_index(drop=True)
dates = None
te_mask = (d.index.year == HOLDOUT_YEAR) if True else None
print("n =", len(endog), "| holdout n =", te_mask.sum() if te_mask is not None else "?")

def fit(order, use_exog):
    m = SARIMAX(endog, exog=exog if use_exog else None, order=order,
                trend="c", enforce_stationarity=False, enforce_invertibility=False)
    return m.fit(disp=False)

res_x  = fit((1,0,1), True)    # ARIMAX
res_nx = fit((1,0,1), False)   # ARIMA (외생 없음)

p_x  = res_x.get_prediction(start=1, dynamic=False).predicted_mean
p_nx = res_nx.get_prediction(start=1, dynamic=False).predicted_mean
y    = endog.iloc[1:]
persist = endog.shift(1).iloc[1:]
m = te_mask[1:]

print("\n=== 1-step 인과예측, 2023 홀드아웃 ===")
for name, p in [("ARIMAX(1,0,1) exog O", p_x), ("ARIMA(1,0,1) exog X", p_nx),
                ("persistence", persist)]:
    print(f"  {name:24s} R2={r2_score(y[m], p[m]):.4f}  MAE={mean_absolute_error(y[m],p[m]):7.1f}")

print("\n=== 계수 ===")
print(res_x.params.round(4).to_string())

# 외생 성분만 (X*beta + const) 의 설명력
b = res_x.params
Xb = b["intercept"] + exog["S_fast"]*b["S_fast"] + exog["S_slow"]*b["S_slow"]
print(f"\n외생성분만(Xβ+c) R2 (홀드아웃) = {r2_score(y[m], Xb.iloc[1:][m]):.4f}")
print(f"  Xβ 범위 {Xb.min():.0f}~{Xb.max():.0f} / y 평균 {endog.mean():.0f}")

# 1-step 예측의 분산 기여 분해
u = endog - Xb                       # ARMA 잔차 성분
print(f"\ny 분산 {endog.var():.0f} | Xβ 분산 {Xb.var():.0f} ({Xb.var()/endog.var()*100:.1f}%) | u 분산 {u.var():.0f}")

# ── 다단계(dynamic) 예측: AR 기여가 decay 하는 구간 ──
print("\n=== h일 앞 예측 (dynamic, 홀드아웃 시작점부터 rolling) ===")
cut = int(np.argmax(te_mask))
idx = np.arange(len(endog))
rows=[]
for h in [1,2,3,5,7,10,14]:
    preds_x, preds_nx, acts, pers = [],[],[],[]
    for s in range(cut, len(endog)-h, 5):     # 5일 간격 origin
        r = res_x.apply(endog.iloc[:s], exog=exog.iloc[:s], refit=False)
        f = r.get_forecast(steps=h, exog=exog.iloc[s:s+h])
        preds_x.append(f.predicted_mean.iloc[-1])
        r2_ = res_nx.apply(endog.iloc[:s], refit=False)
        preds_nx.append(r2_.get_forecast(steps=h).predicted_mean.iloc[-1])
        acts.append(endog.iloc[s+h-1]); pers.append(endog.iloc[s-1])
    rows.append((h, r2_score(acts,preds_x), r2_score(acts,preds_nx), r2_score(acts,pers)))
    print(f"  h={h:2d}일  ARIMAX={rows[-1][1]:7.4f}  ARIMA(exogX)={rows[-1][2]:7.4f}  persist={rows[-1][3]:7.4f}")
