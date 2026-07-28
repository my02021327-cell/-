"""메탄 = 소화가스량 × CH4함량 — 2인자 분해 모델링.

원자료 계산식이 methane = biogas × CH4%/100 임이 확인됐다(1264일 오차 0).
두 인자는 물리적으로 다른 것에 지배된다.
  · 소화가스량(양)  ← 투입 기질 부하    → 투입 pool 로 예측
  · CH4 함량(질)    ← 소화조 내부 상태  → 이화학 상태변수로 예측
따라서 따로 예측해 곱하는 것이 단일 모델보다 타당한지 검정한다.
과거 메탄/가스는 입력에 쓰지 않는다(자기상관 배제).
실행: `python -m src.two_factor`
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

from src.final_ensemble import prepare, TARGET

STATE = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_TS", "소화조_VFA",
         "소화조_TAlk", "소화조_CODcr", "VFA_ALK"]
POOLS = ["S_fast", "S_slow"]
HOLDOUT = 2023


def models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR": make_pipeline(StandardScaler(), SVR(C=100, gamma="scale")),
        "RF": RandomForestRegressor(n_estimators=600, max_depth=8, min_samples_leaf=5,
                                    max_features=0.7, random_state=42, n_jobs=4),
        "ET": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=3, max_features=0.7,
                                  random_state=42, n_jobs=4),
        "XGB": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                            subsample=0.8, reg_lambda=4.0, min_child_weight=6,
                            random_state=42),
    }


def best_fit(tr, te, feats, target):
    """홀드아웃 최고 모델과 그 예측을 돌려준다."""
    out = {}
    for n, m in models().items():
        m.fit(tr[feats].values, tr[target].values)
        p = m.predict(te[feats].values)
        out[n] = (r2_score(te[target], p), p, m)
    best = max(out, key=lambda k: out[k][0])
    return best, out


def load():
    d = prepare()
    e = pd.read_excel("data/monitor_extra.xlsx")
    e["date"] = pd.to_datetime(e["date"])
    d = d.join(e.set_index("date")[["CH4_pct"]])
    d = d[d[TARGET].notna() & d["CH4_pct"].notna()]
    d["biogas"] = d[TARGET] / (d["CH4_pct"] / 100)
    return d


def main():
    d = load()
    print(f"[검산] methane == biogas × CH4%/100  최대오차 "
          f"{(d[TARGET] - d['biogas']*d['CH4_pct']/100).abs().max():.9f}")

    d = d.dropna(subset=STATE + POOLS)
    tr, te = d[d.index.year < HOLDOUT], d[d.index.year == HOLDOUT]
    print(f"[데이터] 학습 {len(tr)}일 / 홀드아웃 {HOLDOUT} {len(te)}일 "
          f"(상태·pool 결측일 삭제)\n")

    print("── ① 소화가스량(양) ← 투입 pool ──")
    b_best, b_all = best_fit(tr, te, POOLS, "biogas")
    for n, (r2, _, _) in b_all.items():
        print(f"   {n:6s} R2={r2:7.4f}")
    print(f"   최고: {b_best} ({b_all[b_best][0]:.4f})\n")

    print("── ② CH4 함량(질) ← 소화조 상태변수 ──")
    c_best, c_all = best_fit(tr, te, STATE, "CH4_pct")
    for n, (r2, _, _) in c_all.items():
        print(f"   {n:6s} R2={r2:7.4f}")
    print(f"   최고: {c_best} ({c_all[c_best][0]:.4f})")
    print(f"   참고: CH4% 학습평균으로 고정 예측 시 R2="
          f"{r2_score(te['CH4_pct'], np.full(len(te), tr['CH4_pct'].mean())):.4f}\n")

    print("── ③ 결합: 메탄 = ①×②/100  vs  단일 모델 직접예측 ──")
    yte = te[TARGET].values
    comb = b_all[b_best][1] * c_all[c_best][1] / 100
    print(f"   2인자 분해        R2={r2_score(yte, comb):7.4f}  MAE={mean_absolute_error(yte, comb):7.1f}")
    comb_fix = b_all[b_best][1] * tr["CH4_pct"].mean() / 100
    print(f"   가스예측×평균CH4%  R2={r2_score(yte, comb_fix):7.4f}  MAE={mean_absolute_error(yte, comb_fix):7.1f}")
    for tag, feats in [("단일: pool 만", POOLS), ("단일: pool+상태", POOLS + STATE)]:
        n, a = best_fit(tr, te, feats, TARGET)
        print(f"   {tag:16s} R2={a[n][0]:7.4f}  MAE={mean_absolute_error(yte, a[n][1]):7.1f}  ({n})")


# ──────────────────────────────────────────────────────────────────────────────
# 롤링-오리진 교차검증 — 단일 홀드아웃은 연도별 편차가 커서 결론을 못 낸다
# ──────────────────────────────────────────────────────────────────────────────
MIN_TRAIN, STEP, BLOCK = 400, 120, 120


def rolling_cv():
    d = load().dropna(subset=STATE + POOLS).sort_index()
    n = len(d)
    specs = {
        "pool만": ("single", POOLS),
        "상태만": ("single", STATE),
        "pool+상태": ("single", POOLS + STATE),
        "2인자 분해": ("two", None),
        "가스(pool)×평균CH4%": ("fix", None),
    }
    acc = {k: [] for k in specs}
    folds = 0
    for s in range(MIN_TRAIN, n - BLOCK + 1, STEP):
        tr, te = d.iloc[:s], d.iloc[s:s + BLOCK]
        folds += 1
        yte = te[TARGET].values
        bb, ba = best_fit(tr, te, POOLS, "biogas")
        cb, ca = best_fit(tr, te, STATE, "CH4_pct")
        for name, (kind, feats) in specs.items():
            if kind == "single":
                k, a = best_fit(tr, te, feats, TARGET)
                acc[name].append(a[k][0])
            elif kind == "two":
                acc[name].append(r2_score(yte, ba[bb][1] * ca[cb][1] / 100))
            else:
                acc[name].append(r2_score(yte, ba[bb][1] * tr["CH4_pct"].mean() / 100))
    print(f"\n── 롤링-오리진 CV ({folds}폴드, 학습 최소 {MIN_TRAIN}일, 검증창 {BLOCK}일) ──")
    print("   과거 메탄·가스는 어느 구성에도 입력으로 쓰지 않는다.")
    res = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
    for k, (m, s) in sorted(res.items(), key=lambda x: -x[1][0]):
        print(f"   {k:20s} R2 = {m:7.4f} ± {s:.4f}")
    return res


if __name__ == "__main__":
    main()
    res = rolling_cv()
    pd.DataFrame([{"구성": k, "CV_R2": m, "CV_std": sd}
                  for k, (m, sd) in res.items()]).sort_values(
        "CV_R2", ascending=False).to_csv(
        "outputs/two_factor_cv.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] outputs/two_factor_cv.csv")
