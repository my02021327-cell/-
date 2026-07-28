"""소화조 내부 이화학 데이터만으로 메탄 예측이 되는가 — 설계 전 타당성 측정.

투입 유량 기반 모델과 달리, 상태변수는 '지금 조 안이 어떤가'를 직접 잰 값이다.
이것이 메탄을 설명하는지, 그리고 결측·시점 제약이 실제로 어떤지 확인한다.
실행: `python -m src.state_probe`
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from src.final_ensemble import prepare, TARGET, HOLDOUT_YEAR

STATE = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_TS", "소화조_VFA",
         "소화조_TAlk", "소화조_MAlk", "소화조_CODcr", "VFA_ALK"]
POOLS = ["S_fast", "S_slow"]

df = prepare()
# 메탄이 있는 날 중 상태변수도 있는 날 (결측일 삭제 원칙)
d = df.dropna(subset=[TARGET] + STATE).copy()
print(f"[데이터] 메탄+상태 모두 존재: {len(d)}일 / 메탄 존재일 {df[TARGET].notna().sum()}일")
print(f"        기간 {d.index.min().date()} ~ {d.index.max().date()}")

tr, te = d[d.index.year < HOLDOUT_YEAR], d[d.index.year == HOLDOUT_YEAR]
print(f"        학습 {len(tr)}일 / 홀드아웃 {len(te)}일\n")

def models():
    return {
        "Ridge":      make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR(RBF)":   make_pipeline(StandardScaler(), SVR(C=100, gamma="scale", epsilon=0.1)),
        "RandomForest": RandomForestRegressor(n_estimators=600, max_depth=8, min_samples_leaf=5,
                                              max_features=0.7, random_state=42, n_jobs=4),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, max_depth=None, min_samples_leaf=3,
                                          max_features=0.7, random_state=42, n_jobs=4),
        "XGBoost":    XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                                   colsample_bytree=0.9, reg_lambda=4.0, min_child_weight=6,
                                   random_state=42),
    }

def bench(feats, tag):
    Xtr, ytr = tr[feats].values, tr[TARGET].values
    Xte, yte = te[feats].values, te[TARGET].values
    print(f"── {tag}  (feature {len(feats)}개, 과거 메탄 미사용) ──")
    best = None
    for n, m in models().items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        r2 = r2_score(yte, p)
        print(f"   {n:14s} R2={r2:7.4f}  MAE={mean_absolute_error(yte,p):7.1f}")
        best = max(best or -9, r2)
    print(f"   → 최고 {best:.4f}\n")
    return best

bench(STATE, "A. 소화조 상태변수만")
bench(POOLS, "B. 투입 pool 만")
bench(STATE + POOLS, "C. 상태 + 투입 pool")

# ── ARIMAX: 상태변수를 exog 로 (과거 메탄 사용) ──
endog = d[TARGET].astype(float).reset_index(drop=True)
te_m = (d.index.year == HOLDOUT_YEAR)[1:]
y1 = endog.iloc[1:]
print("── D. ARIMAX(1,0,1) exog 별 1-step 홀드아웃 (과거 메탄 사용) ──")
for tag, cols in [("투입 pool", POOLS), ("상태변수", STATE), ("상태+pool", STATE + POOLS)]:
    ex = d[cols].astype(float).reset_index(drop=True)
    r = SARIMAX(endog, exog=ex, order=(1, 0, 1), trend="c",
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    p = r.get_prediction(start=1, dynamic=False).predicted_mean
    # 외생성분 단독(과거 메탄 배제)
    xb = r.params["intercept"] + sum(ex[c] * r.params[c] for c in cols)
    print(f"   {tag:10s} 전체 R2={r2_score(y1[te_m], p[te_m]):.4f}   "
          f"외생성분만 R2={r2_score(y1[te_m], xb.iloc[1:][te_m]):.4f}")
print(f"   {'persistence':10s} 전체 R2={r2_score(y1[te_m], endog.shift(1).iloc[1:][te_m]):.4f}")
