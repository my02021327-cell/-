"""다지평 직접예측 앙상블 — 과거 메탄 + 과거 이화학 + HRT 지연을 결합한다.

설계 원칙
  · AR 보정형(어제값 ± δ)이 아니라 **직접 다지평 예측**: y(t+h) 를 시점 t 의
    정보만으로 회귀한다. h 가 커질수록 자기상관은 스스로 무력해지므로,
    같은 지평의 persistence 대비 이득이 곧 모델의 실력이 된다.
  · 과거 메탄은 '수준·추세' feature 로 참여할 뿐 단독 지배하지 않는다.
  · 이화학은 시점 t 까지 관측된 값만 쓴다(운전자가 실제로 아는 정보).
  · 투입량은 제어입력이라 미래 계획을 아는 것으로 본다(대시보드와 동일 가정).
  · HRT 기반 기질가용성 2-pool(τ=1, 8일)로 지연을 명시한다.

실행: `python -m src.horizon_ensemble`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.bio_lag import substrate_pool
from src.final_ensemble import FEED, TARGET, prepare

HORIZONS = [1, 3, 5, 7, 10, 14]
CHEM = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_TS", "소화조_VFA",
        "소화조_TAlk", "소화조_CODcr", "VFA_ALK"]
CHEM_FFILL = 5            # 이화학은 평일 계측 → 최대 5일까지 최신값 유지
MIN_TRAIN, STEP, BLOCK = 350, 55, 90


# ──────────────────────────────────────────────────────────────────────────────
# 피처
# ──────────────────────────────────────────────────────────────────────────────
def build(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """시점 t 에서 알 수 있는 정보만으로 피처를 만든다."""
    f = pd.DataFrame(index=d.index)
    y = d[TARGET]

    # ① 과거 메탄 — 수준과 추세. 단독 지배하지 않도록 평활 위주로 구성.
    f["m_last"] = y.ffill()
    for w in (3, 7, 14, 30):
        f[f"m_ewm{w}"] = y.ffill().ewm(span=w, adjust=False).mean()
    f["m_trend7"] = f["m_ewm7"] - f["m_ewm30"]
    f["m_std14"] = y.ffill().rolling(14).std()
    f["m_age"] = y.notna().cumsum().pipe(lambda s: s.groupby(s).cumcount())  # 최신 실측 경과일

    # ② 과거 이화학 — 관측 시점까지의 값과 그 평활·변화
    c = d[CHEM].ffill(limit=CHEM_FFILL)
    for col in CHEM:
        f[f"c_{col}"] = c[col]
        f[f"c_{col}_ewm7"] = c[col].ewm(span=7, adjust=False).mean()
        f[f"c_{col}_d7"] = c[col] - c[col].shift(7)

    # ③ HRT 기반 기질가용성 — 지연 구조를 명시
    f["S_fast"] = d["S_fast"]
    f["S_slow"] = d["S_slow"]
    f["feed"] = d[FEED]
    f["load5"] = d[FEED].rolling(5).mean()
    f["load10"] = d[FEED].rolling(10).mean()
    f["load40"] = d[FEED].rolling(40).mean()          # HRT 40.6일 창

    # ④ 상호작용 — 기질이 있어도 상태가 나쁘면 전환이 안 된다
    f["S_slow_x_pH"] = f["S_slow"] * f["c_소화조_pH"]
    f["S_slow_x_T"] = f["S_slow"] * f["c_소화조_온도"]
    f["S_slow_x_VFAALK"] = f["S_slow"] * f["c_VFA_ALK"]
    return f, list(f.columns)


def future_pools(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """t+h 시점의 기질가용성 — 투입계획은 알려진 것으로 본다."""
    return pd.DataFrame({"S_fast_h": d["S_fast"].shift(-h),
                         "S_slow_h": d["S_slow"].shift(-h),
                         "feed_h": d[FEED].shift(-h)}, index=d.index)


# ──────────────────────────────────────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────────────────────────────────────
def bases():
    return {
        "XGB": XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0,
                            min_child_weight=6, random_state=42, n_jobs=4),
        "RF": RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=4,
                                    max_features=0.5, random_state=42, n_jobs=4),
        "ET": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.6,
                                  random_state=42, n_jobs=4),
        "SVR": make_pipeline(StandardScaler(), SVR(C=200, gamma="scale", epsilon=0.05)),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def stack(Xtr, ytr, Xte, n_splits=4):
    """시간순 OOF → 비음 Ridge 메타. 누수 없음."""
    n = len(Xtr)
    edges = np.linspace(n // (n_splits + 1), n, n_splits + 1).astype(int)
    names = list(bases())
    oof = np.full((n, len(names)), np.nan)
    for i in range(n_splits):
        a, b = edges[i], edges[i + 1]
        for j, nm in enumerate(names):
            m = bases()[nm]
            m.fit(Xtr[:a], ytr[:a])
            oof[a:b, j] = m.predict(Xtr[a:b])
    ok = ~np.isnan(oof).any(axis=1)
    meta = Ridge(alpha=1.0, positive=True).fit(oof[ok], ytr[ok])
    te = np.column_stack([bases()[nm].fit(Xtr, ytr).predict(Xte) for nm in names])
    return meta.predict(te), dict(zip(names, meta.coef_.round(4)))


# ──────────────────────────────────────────────────────────────────────────────
def run():
    d = prepare()
    d["S_fast"] = substrate_pool(d[FEED], 1)
    d["S_slow"] = substrate_pool(d[FEED], 8)
    F, cols = build(d)

    rows = []
    for h in HORIZONS:
        fp = future_pools(d, h)
        X = pd.concat([F, fp], axis=1)
        yt = d[TARGET].shift(-h)                      # 예측 대상 : h일 뒤 메탄
        keep = yt.notna() & d[TARGET].notna() & X.notna().all(axis=1)
        Xd, yd, base = X[keep], yt[keep], d[TARGET][keep]
        n = len(Xd)
        feats = cols + list(fp.columns)
        r2s, pers, maes, ws = [], [], [], []
        for s in range(MIN_TRAIN, n - BLOCK + 1, STEP):
            tr = slice(0, s); te = slice(s, s + BLOCK)
            p, w = stack(Xd[feats].values[tr], yd.values[tr], Xd[feats].values[te])
            r2s.append(r2_score(yd.values[te], p))
            maes.append(mean_absolute_error(yd.values[te], p))
            pers.append(r2_score(yd.values[te], base.values[te]))   # 같은 지평 persistence
            ws.append(w)
        if not r2s:
            print(f"h={h:2d}일  n={n:4d} — 폴드 부족, 건너뜀")
            continue
        rows.append(dict(h=h, n=n, folds=len(r2s),
                         R2=np.mean(r2s), R2_std=np.std(r2s), MAE=np.mean(maes),
                         persist=np.mean(pers), gain=np.mean(r2s) - np.mean(pers),
                         weights={k: round(float(np.mean([w[k] for w in ws])), 3)
                                  for k in ws[0]}))
        r = rows[-1]
        print(f"h={h:2d}일  n={n:4d} {r['folds']}폴드 | 앙상블 R2={r['R2']:.4f}±{r['R2_std']:.4f} "
              f"MAE={r['MAE']:6.1f} | persistence={r['persist']:7.4f} | 이득={r['gain']:+.4f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("다지평 직접예측 앙상블 (XGB/RF/ET/SVR/Ridge → 비음 Ridge 메타)")
    print("롤링-오리진 CV, 같은 지평 persistence 대비 이득으로 평가\n")
    t = run()
    t.to_csv("outputs/horizon_ensemble.csv", index=False, encoding="utf-8-sig")
    print("\n메타 가중치(폴드 평균):")
    for _, r in t.iterrows():
        print(f"  h={r['h']:2d}: {r['weights']}")
    print("\n[저장] outputs/horizon_ensemble.csv")
