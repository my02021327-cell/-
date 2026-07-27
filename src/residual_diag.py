"""예측 잔차가 소화조 화학상태를 설명하는가 = 예측·진단 연결 가능성 검정.

예측 모델의 외생입력은 투입 유량 하나뿐이다. 그렇다면 '투입으로 설명되지 않는
메탄'(잔차 u = y − Xβ)이 소화조 화학상태를 담고 있어야 예측기와 상태감시기를
인과적으로 연결할 수 있다. 두 가지로 검정한다.
  ① u 와 상태변수의 상관 (자기상관 보정: 이동블록 부트스트랩 95% CI)
  ② 상태변수를 exog 로 추가했을 때 홀드아웃 R2 변화
실행: `python -m src.residual_diag`
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
from src.final_ensemble import prepare, SARIMAX_EXOG, TARGET, HOLDOUT_YEAR

df = prepare()
d = df.dropna(subset=[TARGET]).copy()
endog = d[TARGET].astype(float).reset_index(drop=True)
exog = d[SARIMAX_EXOG].ffill().bfill().reset_index(drop=True)
res = SARIMAX(endog, exog=exog, order=(1,0,1), trend="c",
              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
b = res.params
Xb = b["intercept"] + exog["S_fast"]*b["S_fast"] + exog["S_slow"]*b["S_slow"]
u = (endog - Xb)                      # 투입으로 설명 안 되는 메탄
u.index = d.index

STATE = ["소화조_pH","VFA_ALK","소화조_온도","소화조_NH4N","소화조_VFA","소화조_TAlk","소화조_VS"]

def block_boot_r(x, y, B=2000, L=30, seed=0):
    rng = np.random.default_rng(seed); n=len(x); rs=[]
    nb = int(np.ceil(n/L))
    for _ in range(B):
        st = rng.integers(0, max(n-L,1), nb)
        idx = np.concatenate([np.arange(s, min(s+L,n)) for s in st])[:n]
        if len(idx)<10: continue
        xi, yi = x[idx], y[idx]
        if xi.std()==0 or yi.std()==0: continue
        rs.append(np.corrcoef(xi, yi)[0,1])
    return np.percentile(rs, [2.5, 97.5])

print("잔차 u (= 실측 − 투입설명분) vs 소화조 상태변수")
print(f"{'변수':14s} {'n':>5s} {'r':>7s} {'95%CI(블록부트)':>22s}  판정")
for c in STATE:
    m = d[c].notna().values & u.notna().values
    if m.sum() < 60: print(f"{c:14s} {m.sum():5d}   표본부족"); continue
    x = d[c].values[m].astype(float); y = u.values[m]
    r = np.corrcoef(x,y)[0,1]
    lo,hi = block_boot_r(x,y)
    sig = "유의" if lo*hi>0 else "무의미(0 포함)"
    print(f"{c:14s} {m.sum():5d} {r:7.3f}   [{lo:6.3f}, {hi:6.3f}]  {sig}")

# ── 상태변수를 외생으로 추가하면 예측이 좋아지는가 ──
print("\n상태변수를 exog 에 추가했을 때 홀드아웃 R2 (1-step)")
te = (d.index.year == HOLDOUT_YEAR)[1:]
y1 = endog.iloc[1:]
base_p = res.get_prediction(start=1, dynamic=False).predicted_mean
print(f"  {'기준 S_fast+S_slow':32s} R2={r2_score(y1[te], base_p[te]):.4f}")
for c in ["소화조_pH","VFA_ALK","소화조_온도"]:
    ex = pd.concat([exog, d[c].astype(float).ffill().bfill().reset_index(drop=True)], axis=1)
    try:
        r2m = SARIMAX(endog, exog=ex, order=(1,0,1), trend="c",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        p = r2m.get_prediction(start=1, dynamic=False).predicted_mean
        print(f"  {'+ '+c:32s} R2={r2_score(y1[te], p[te]):.4f}   계수={r2m.params[c]:.1f}")
    except Exception as e:
        print(f"  + {c}: 실패 {e}")
