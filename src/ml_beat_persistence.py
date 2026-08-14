# -*- coding: utf-8 -*-
"""persistence 정박 잔차학습 (Persistence-Anchored Residual Learning, 다지평).

목표: 모든 운영 지평(1·3·7·14일)에서 persistence를 이기거나 대등할 것.

방법론 (자체 고안):
  예측식  ŷ(t+h) = y(t) + r̂_h(t)          ← persistence에 '정박'
  잔차    r_h(t) = y(t+h) − y(t) 를 GBM으로 학습.
  모델이 0을 출력하면 정확히 persistence와 동률 → 구조적으로 크게 질 수 없고,
  학습이 신호를 찾는 만큼 persistence를 초과한다.

  핵심 특징 1 — 기질 평형 복귀항(gap): 가수분해 커널 기대치 (a·H+c) − y(t).
    현재 가스량이 기질이 지지하는 수준보다 높으면 하락, 낮으면 상승이 예상된다.
    (반응조 관성 때문에 y는 기질 평형으로 서서히 복귀 — HRT 40.5일 CSTR 물리)
  핵심 특징 2 — 내부 이화학 상태: 소화조 pH·VFA·알칼리도·VFA/Alk·COD·TS,
    산발효조 pH·TS·COD, CH4함량, TAN 추세지수(알칼리도 역산: (ALK−14089)/0.640).
    실험실 항목은 주말 구조 결측 → 최근 관측값 유지(ffill) + 경과일(age) 채널 병행.
    (주말 선형보간 금지 원칙 준수 — 미래값을 쓰지 않는 인과적 처리)

데이터 3분할 (시간순, 혼합 금지):
  학습(train)        2018-01-01 ~ 2021-12-31   — 잔차모델 적합
  사고학습(valid)    2022-01-01 ~ 2022-12-31   — 하이퍼파라미터·모델 선택 전용
  평가(test)         2023-01-01 ~ 2023-09-17   — 최종 1회 평가, 선택에 불사용

평가: skill_h = 1 − MSE_모델 / MSE_persistence (동일 표본, 지평별).
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CSV = "data/영천BGP_MASTER_2018-2023.csv"
M = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
y = M.biogas_AB_m3d

# ---------------------------------------------------------------- 기질 부하·커널
TS = M.acid_TS_pct.interpolate(limit=3)
Q = M.feed_AB_tpd.ffill(limit=2)
TS_load = (Q * TS / 100).ffill(limit=3)


def kernel(s, k=0.35, K=120):
    tau = np.arange(K + 1)
    w = k * np.exp(-k * tau)
    w /= w.sum()
    v = np.nan_to_num(s.values)
    out = np.zeros(len(v))
    for i, wv in enumerate(w):
        out[i:] += wv * v[: len(v) - i]
    return pd.Series(out, index=s.index)


H = kernel(TS_load)
D0 = pd.concat([H.rename("H"), y.rename("y")], axis=1).dropna()
tr0 = D0[D0.index.year <= 2021]
A = np.c_[np.ones(len(tr0)), tr0[["H"]].values]
c0, a1 = np.linalg.lstsq(A, tr0["y"].values, rcond=None)[0]
eq = c0 + a1 * H                      # 기질 평형 수준
gap = (eq - y).rename("gap")          # 평형 복귀항 (핵심 특징)

# ---------------------------------------------------------------- 내부 이화학 특징
def lab(s, limit=10):
    """실험실 항목: 최근 관측 유지 + 경과일 채널 (인과적, 미래 미사용)."""
    obs = s.notna()
    age = (~obs).groupby(obs.cumsum()).cumcount()
    return s.ffill(limit=limit), age.where(s.ffill(limit=limit).notna())


dig_pH, dig_pH_age = lab(M[["dig_pH_A", "dig_pH_B"]].mean(axis=1))
VFA, VFA_age = lab(M.VFA_A_mgL)
ALK, ALK_age = lab(M.ALK_A_mgL)          # B계열 2023 오염(DQ-05) → A만 사용
vfaalk = (VFA / ALK).rename("vfaalk")
digCOD, _ = lab(M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1))
digTS, _ = lab(M.dig_TS_A_pct)
acid_pH, _ = lab(M.acid_pH)
acidCOD, _ = lab(M.acid_CODcr_mgL)
ch4pct, _ = lab(M.CH4_pct)
tan_proxy = ((ALK - 14089) / 0.640).rename("tan_proxy")   # 알칼리도 역산 추세지수

F = pd.DataFrame(
    {
        # persistence 정박 잔차 학습의 물리 특징
        "gap": gap,
        "gap_m3": gap.rolling(3, min_periods=1).mean(),
        "dy1": y.diff(1),
        "dy3": y.diff(3),
        "dy7": y.diff(7),
        "y_dev7": y.rolling(7, min_periods=3).mean() - y,
        "H": H,
        "TS_load": TS_load,
        "TS_load_m7": TS_load.rolling(7, min_periods=3).mean(),
        "dfeed3": M.feed_AB_tpd.diff(3),
        # 내부 이화학 상태 (진단 계층 변수의 예측 활용)
        "dig_pH": dig_pH,
        "dig_pH_age": dig_pH_age,
        "VFA": VFA,
        "ALK": ALK,
        "ALK_age": ALK_age,
        "vfaalk": vfaalk,
        "digCOD": digCOD,
        "digTS": digTS,
        "acid_pH": acid_pH,
        "acidCOD": acidCOD,
        "ch4pct": ch4pct,
        "tan_proxy": tan_proxy,
        "dALK14": ALK.diff(14),
        "dVFA7": VFA.diff(7),
        # 달력
        "dow": M.dow,
        "month": M.month,
    }
)
INTERNAL = ["dig_pH", "dig_pH_age", "VFA", "ALK", "ALK_age", "vfaalk", "digCOD",
            "digTS", "acid_pH", "acidCOD", "ch4pct", "tan_proxy", "dALK14", "dVFA7"]

TR = slice("2018-01-01", "2021-12-31")   # 학습
VA = slice("2022-01-01", "2022-12-31")   # 사고학습(검증) — 선택 전용
TE = slice("2023-01-01", "2023-09-17")   # 평가 — 최종 1회

HORIZONS = [1, 3, 7, 14]


def build(h, cols):
    """지평별 설계행렬. 투입량·기질 TS는 운전 계획으로 기지(설정 변수)라는
    가정 하에 미래 평형항·계획 투입 변화를 특징에 포함한다
    (persistence+Δ투입 모델과 동일한 시나리오 조건부 예측 가정)."""
    r = (y.shift(-h) - y).rename("r")            # 잔차 타깃
    feed = M.feed_AB_tpd
    fut = pd.DataFrame(
        {
            "gap_h": eq.shift(-h) - y,                       # 계획 부하 기준 평형 복귀항
            "dfeed_h": feed.shift(-h) - feed,                # 계획 투입 변화
            "feed_fut_m": sum(feed.shift(-j) for j in range(1, h + 1)) / h - feed,
        }
    )
    X = pd.concat([F[cols], fut, r], axis=1)
    X = X[X["r"].notna() & F["gap"].notna() & fut["gap_h"].notna()]
    return X


def skill_eval(pred_abs, idx):
    """pred_abs: ŷ(t+h) 시계열(측정시점 t 인덱스). 동일 표본 persistence 대비."""
    out = {}
    for name, sl in [("valid", VA), ("test", TE)]:
        p = pred_abs.loc[sl]
        t_future = p["yt_h"]
        m = p["pred"].notna() & t_future.notna()
        tt, pp, pers = t_future[m].values, p["pred"][m].values, p["y0"][m].values
        mse_m = ((tt - pp) ** 2).mean()
        mse_p = ((tt - pers) ** 2).mean()
        out[name] = {
            "n": int(m.sum()),
            "R2": round(float(1 - mse_m / ((tt - tt.mean()) ** 2).sum() * len(tt)), 3),
            "RMSE": round(float(np.sqrt(mse_m))),
            "skill": round(float(1 - mse_m / mse_p), 3),
            "pers_R2": round(float(1 - mse_p / ((tt - tt.mean()) ** 2).sum() * len(tt)), 3),
        }
    return out


results = {}
imp_h7 = None
series_h7 = None
GRID = [
    {"learning_rate": 0.03, "max_depth": 2, "max_iter": 200, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_depth": 3, "max_iter": 400, "l2_regularization": 0.5},
    {"learning_rate": 0.08, "max_depth": 4, "max_iter": 300, "l2_regularization": 1.0},
]

for h in HORIZONS:
    per_model = {}

    def run(cols, tag):
        X = build(h, cols)
        Xtr = X.loc[TR]
        best, best_skill = None, -9e9
        for gcfg in GRID:
            g = HistGradientBoostingRegressor(random_state=0, **gcfg)
            g.fit(Xtr.drop(columns="r"), Xtr["r"])
            rp = pd.Series(g.predict(X.drop(columns="r")), index=X.index)
            frame = pd.DataFrame({"y0": y.reindex(X.index),
                                  "yt_h": y.shift(-h).reindex(X.index),
                                  "pred": y.reindex(X.index) + rp})
            ev = skill_eval(frame, X.index)
            if ev["valid"]["skill"] > best_skill:
                best_skill, best = ev["valid"]["skill"], (g, ev, gcfg, frame)
        g, ev, gcfg, frame = best
        per_model[tag] = ev | {"config": gcfg}
        return g, frame

    # (a) 선형 정박: r̂ = b0 + b1·gap_h  (train 적합)
    Xl = build(h, ["gap"])
    bl = np.polyfit(Xl.loc[TR]["gap_h"].values, Xl.loc[TR]["r"].values, 1)
    rp = bl[0] * Xl["gap_h"] + bl[1]
    frame_lin = pd.DataFrame({"y0": y.reindex(Xl.index),
                              "yt_h": y.shift(-h).reindex(Xl.index),
                              "pred": y.reindex(Xl.index) + rp})
    per_model["linear_gap"] = skill_eval(frame_lin, Xl.index)

    # (b) GBM — 부하·gap만 (내부 이화학 제외, ablation)
    run([c for c in F.columns if c not in INTERNAL], "gbm_no_internal")

    # (c) GBM — 전체 (내부 이화학 포함) ← 제안 모델
    g_full, frame_full = run(list(F.columns), "gbm_full")

    results[h] = per_model
    if h == 7:
        imp = sorted(zip([c for c in F.columns],
                         getattr(g_full, "feature_importances_", np.zeros(len(F.columns)))),
                     key=lambda x: -x[1])
        series_h7 = frame_full

# HistGB에는 feature_importances_가 없음 → permutation importance (h=7, valid 구간)
from sklearn.inspection import permutation_importance

X7 = build(7, list(F.columns))
Xtr7 = X7.loc[TR]
best_cfg = results[7]["gbm_full"]["config"]
g7 = HistGradientBoostingRegressor(random_state=0, **best_cfg)
g7.fit(Xtr7.drop(columns="r"), Xtr7["r"])
Xva7 = X7.loc[VA]
pi = permutation_importance(g7, Xva7.drop(columns="r"), Xva7["r"], n_repeats=10, random_state=0)
imp_h7 = sorted(zip(Xva7.drop(columns="r").columns, pi.importances_mean), key=lambda x: -x[1])
importances = [{"f": f, "v": round(float(v), 4)} for f, v in imp_h7]

# h=7 차트용: 실측 y(t+h)와 예측 ŷ(t+h)를 도착일(t+h) 기준으로 재배열
arr_pred = pd.Series(series_h7["pred"].values,
                     index=series_h7.index + pd.Timedelta(days=7)).reindex(M.index)
out_series = [None if pd.isna(v) else round(float(v)) for v in arr_pred.values]

out = {
    "split": {"train": "2018-01-01~2021-12-31", "valid": "2022-01-01~2022-12-31",
              "test": "2023-01-01~2023-09-17"},
    "horizons": {str(h): results[h] for h in HORIZONS},
    "importances_h7": importances,
    "pred_h7": out_series,
}
with open("outputs/parl_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("지평 | 모델            | 사고학습(2022) R2/skill | 평가(2023) R2/skill | pers R2")
for h in HORIZONS:
    for tag in ["linear_gap", "gbm_no_internal", "gbm_full"]:
        r = results[h][tag]
        print(f"h={h:2d} | {tag:15s} | {r['valid']['R2']:.3f} {r['valid']['skill']:+.3f}"
              f" | {r['test']['R2']:.3f} {r['test']['skill']:+.3f} | {r['test']['pers_R2']:.3f}")
print("\nh=7 permutation importance (top 10):")
for f_, v in imp_h7[:10]:
    print(f"  {f_:14s} {v:.4f}")
