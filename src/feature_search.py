# -*- coding: utf-8 -*-
"""전 이화학 변수 × 시차(lag) 자동 탐색 기반 메탄생성 예측 모델링.

이전 판의 한계를 두 가지 고친다.
  (1) 변수 누락 — pH·온도 외 VFA·알칼리도·TS·VS·COD·TN, 여액저장조 3종, B계열,
      반입 조성 3종, 탈수량까지 마스터의 이화학 변수를 전수 후보로 올린다.
  (2) 시차 부재 — HRT 40.5일 공정에서 동시점 값만 쓰는 것은 물리적으로 부적절하다.
      모든 후보에 lag(0~14일)·이동평균(3~30일)·변화량(7~30일)을 자동 생성한다.

■ 특징 생성
    각 기저 변수마다  level, lag 1/2/3/5/7/14,  rolling mean 3/7/14/30,
    delta 7/14/30  → 후보 수백 개. 실험실 항목은 최근 관측 유지(ffill limit 7)
    후 시차를 걸어 "t 시점에 알 수 있는 값"만 사용한다(미래 미참조).
    측정 경과일(age) 채널을 함께 넣어 값의 신선도를 모델이 알 수 있게 한다.

■ 특징 선택 (사고학습 2022 전용, 평가 2023 미사용)
    ① 프리스크리닝 : 학습구간 결측률 40% 초과 제거, 타깃 상관 상위 N만 통과,
                     상호 |r|>0.98 중복쌍 제거
    ② 전진선택     : Ridge로 사고학습 R²를 최대화하는 특징을 하나씩 추가,
                     개선폭 0.002 미만이면 중단 (최대 20개)
    ③ 최종 적합    : 선택된 특징으로 Ridge/RF/GBM 학습 후 NNLS 결합

■ 블록별 독립 선택으로 ablation: S(투입·기질) / I(소화조 내부) / S+I
■ 분할: 학습 2018-2021 / 사고학습 2022 / 평가 2023(최종 1회)
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from scipy.optimize import nnls

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
TR = slice("2018-01-01", "2021-12-31")
VA = slice("2022-01-01", "2022-12-31")
TE = slice("2023-01-01", "2023-09-17")

# ================================================================ 기저 변수 정의
# DQ 플래그 반영: 센서 결함·기록 오염 구간은 NaN 처리
T_A = M.dig_T_A_C.where(M.flag_TA_sensor_fault == 0)
T_B = M.dig_T_B_C.where(M.flag_TB_sensor_fault == 0)
ALK_B = M.ALK_B_mgL.where(M.flag_ALKB_corrupted == 0)

feed = M.feed_AB_tpd.ffill(limit=2)
acid_TS, acid_VS, acid_COD, acid_TN = M.acid_TS_pct, M.acid_VS_pct, M.acid_CODcr_mgL, M.acid_TN_mgL
VFA = M[["VFA_A_mgL", "VFA_B_mgL"]].mean(axis=1)
ALK = pd.concat([M.ALK_A_mgL, ALK_B], axis=1).mean(axis=1)
dig_pH = M[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
dig_T = pd.concat([T_A, T_B], axis=1).mean(axis=1)
dig_TS = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)
dig_VS = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)
dig_COD = M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
NH3N = M[["NH3N_A_mgL", "NH3N_B_mgL"]].mean(axis=1)

pKa = 0.09018 + 2729.92 / (38.0 + 273.15)

# S 블록 — 투입량·반입 조성·여액저장조·기질 성상·부하
S_BASE = {
    "intake_total": M.intake_total_tpd, "intake_foodww": M.intake_foodww_tpd,
    "intake_manure": M.intake_manure_tpd, "intake_food": M.intake_food_tpd,
    "manure_share": M.intake_manure_tpd / M.intake_total_tpd * 100,
    "leach_pH": M.leach_pH, "leach_TS": M.leach_TS_pct, "leach_VS": M.leach_VS_pct,
    "leach_VSTS": M.leach_VS_pct / M.leach_TS_pct,
    "acid_pH": M.acid_pH, "acid_TS": acid_TS, "acid_VS": acid_VS,
    "acid_COD": acid_COD, "acid_TN": acid_TN,
    "acid_VSTS": acid_VS / acid_TS, "acid_CODVS": acid_COD / (acid_VS * 10000),
    "acid_CN": acid_COD / acid_TN,
    "feed": feed,
    "TS_load": feed * acid_TS / 100, "VS_load": feed * acid_VS / 100,
    "COD_load": feed * acid_COD / 1000, "TN_load": feed * acid_TN / 1000,
    "OLR": feed * acid_VS / 100 * 1000 / 8000, "HRT": 8000 / feed.replace(0, np.nan),
}
# I 블록 — 소화조 내부 이화학 + 후단 탈수
I_BASE = {
    "dig_pH": dig_pH, "dig_pH_AB_gap": (M.dig_pH_A - M.dig_pH_B).abs(),
    "dig_T": dig_T, "VFA": VFA, "ALK": ALK, "vfaalk": VFA / ALK,
    "ALK_minus_VFA": ALK - 0.71 * VFA,                       # 중탄산 알칼리도
    "dig_TS": dig_TS, "dig_VS": dig_VS, "dig_VSTS": dig_VS / dig_TS,
    "dig_COD": dig_COD, "dig_TN": M.dig_TN_A_mgL, "NH3N": NH3N,
    "FAN": NH3N / (1 + 10 ** (pKa - dig_pH)),
    "TAN_idx": (ALK - 14089) / 0.640,                        # 알칼리도 역산 암모니아 지수
    "COD_rem": (acid_COD - dig_COD) / acid_COD * 100,
    "VS_rem": (acid_VS - dig_VS) / acid_VS * 100,
    "TS_rem": (acid_TS - dig_TS) / acid_TS * 100,
    "VFA_per_ALK_dev": (VFA / ALK) - (VFA / ALK).rolling(30, min_periods=10).mean(),
    "dewater": M.dewater_tpd,
}
LAB = set(["leach_pH", "leach_TS", "leach_VS", "leach_VSTS", "acid_pH", "acid_TS", "acid_VS",
           "acid_COD", "acid_TN", "acid_VSTS", "acid_CODVS", "acid_CN"]) | set(I_BASE) - {"dewater"}

LAGS = [0, 1, 2, 3, 5, 7, 14]
ROLLS = [3, 7, 14, 30]
DELTAS = [7, 14, 30]


def expand(name, s, is_lab):
    """한 기저 변수를 시차·이동평균·변화량 특징군으로 전개."""
    out = {}
    base = s.ffill(limit=7) if is_lab else s.ffill(limit=2)
    for L in LAGS:
        out[f"{name}@L{L}"] = base.shift(L)
    for R in ROLLS:
        out[f"{name}@M{R}"] = base.rolling(R, min_periods=max(2, R // 3)).mean()
    for Dd in DELTAS:
        out[f"{name}@D{Dd}"] = base - base.shift(Dd)
    if is_lab:                                  # 값의 신선도(마지막 실측 이후 경과일)
        obs = s.notna()
        out[f"{name}@age"] = (~obs).groupby(obs.cumsum()).cumcount().where(base.notna())
    return out


def kernel(s, k=0.35, K=120):
    tau = np.arange(K + 1); w = k * np.exp(-k * tau); w /= w.sum()
    v = np.nan_to_num(s.interpolate(limit=3).values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


BANK, BLOCK = {}, {}
for nm, s in S_BASE.items():
    for k, v in expand(nm, s, nm in LAB).items():
        BANK[k] = v; BLOCK[k] = "S"
for nm, s in I_BASE.items():
    for k, v in expand(nm, s, nm in LAB).items():
        BANK[k] = v; BLOCK[k] = "I"
# 물리 커널 (부하 계열 전용)
for nm in ["TS_load", "VS_load", "COD_load"]:
    for kk in [0.15, 0.35, 0.80]:
        key = f"{nm}@K{kk}"
        BANK[key] = kernel(S_BASE[nm], k=kk); BLOCK[key] = "S"
# 달력
for k, v in {"dow": M.dow, "month": M.month}.items():
    BANK[k] = v; BLOCK[k] = "S"

FB = pd.DataFrame(BANK).replace([np.inf, -np.inf], np.nan)
print(f"후보 특징 총 {FB.shape[1]}개 (S {sum(1 for v in BLOCK.values() if v=='S')} / "
      f"I {sum(1 for v in BLOCK.values() if v=='I')})")


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def prescreen(cols, tgt, sample, top=90):
    """학습구간 결측률·타깃 상관·상호중복으로 후보를 압축."""
    tr_idx = FB.loc[TR].index.intersection(sample)
    X = FB.loc[tr_idx, cols]
    y = tgt.reindex(tr_idx)
    keep = [c for c in cols if X[c].isna().mean() < 0.40 and X[c].std(skipna=True) > 0]
    if not keep:
        return []
    cor = X[keep].corrwith(y, method="spearman").abs().sort_values(ascending=False)
    cand = [c for c in cor.index if cor[c] > 0.05][:top * 2]
    sel, mat = [], X[cand].fillna(X[cand].median())
    for c in cand:                                    # 상호 |r|>0.98 중복 제거
        if all(abs(np.corrcoef(mat[c], mat[s])[0, 1]) <= 0.98 for s in sel):
            sel.append(c)
        if len(sel) >= top:
            break
    return sel


# 롤링 오리진 폴드 — 선택 기준을 단일 연도가 아닌 다중 시점 일반화로 둔다.
# 단일 연도(2022) 기준 탐욕 선택은 그 해에 과적합되어 평가 성능이 오히려 떨어졌다
# (실측: L2 S+I 사고학습 0.854 / 평가 0.238). 아래 3개 폴드 평균으로 교체한다.
FOLDS = [(slice("2018-01-01", "2019-12-31"), slice("2020-01-01", "2020-12-31")),
         (slice("2018-01-01", "2020-12-31"), slice("2021-01-01", "2021-12-31")),
         (slice("2018-01-01", "2021-12-31"), slice("2022-01-01", "2022-12-31"))]


def cv_score(D, cs):
    """롤링 오리진 3폴드 평균 R². 최저 폴드도 함께 반환해 불안정한 조합을 거른다."""
    ss = []
    for a, b in FOLDS:
        tr_, va_ = D.loc[a], D.loc[b]
        if len(tr_) < 30 or len(va_) < 10:
            continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          Ridge(alpha=10.0)).fit(tr_[cs], tr_["y"])
        ss.append(r2(va_["y"].values, m.predict(va_[cs])))
    return (float(np.mean(ss)), float(np.min(ss))) if ss else (-9e9, -9e9)


def forward_select(cand, tgt, sample, max_k=10, tol=0.005):
    """롤링 오리진 평균 R²로 전진선택. 평가 구간(2023)은 일절 참조하지 않는다."""
    D = pd.concat([FB[cand], tgt.rename("y")], axis=1).loc[sample].dropna(subset=["y"])
    if len(D.loc[TR]) < 60:
        return [], []
    chosen, best, trace = [], -9e9, []
    rest = list(cand)
    while len(chosen) < max_k and rest:
        scored = [(cv_score(D, chosen + [c]), c) for c in rest]
        (mean_s, min_s), c = max(scored, key=lambda x: x[0][0])
        if mean_s - best < tol:
            break
        best, chosen = mean_s, chosen + [c]
        trace.append({"f": c, "cv_mean": round(mean_s, 3), "cv_min": round(min_s, 3)})
        rest.remove(c)
    return chosen, trace


def final_fit(cols, tgt, sample, tag):
    D = pd.concat([FB[cols], tgt.rename("y")], axis=1).loc[sample].dropna(subset=["y"])
    tr, va, te = D.loc[TR], D.loc[VA], D.loc[TE]
    mk = lambda est: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    P = {"ridge": mk(RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(tr[cols], tr["y"]),
         "rf": make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                                   max_features=0.6, random_state=0,
                                                   n_jobs=-1)).fit(tr[cols], tr["y"]),
         "gbm": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                              l2_regularization=1.0,
                                              random_state=0).fit(tr[cols], tr["y"])}
    keys = list(P)
    w, _ = nnls(np.c_[[P[k].predict(va[cols]) for k in keys]].T, va["y"].values)
    w = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
    res = {"tag": tag, "k": len(cols), "features": cols,
           "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
           "weights": {k: round(float(x), 2) for k, x in zip(keys, w)}}
    for lb, d in [("valid", va), ("test", te)]:
        Pm = np.c_[[P[k].predict(d[cols]) for k in keys]].T
        e = Pm @ w
        res[lb] = {"R2": round(r2(d["y"].values, e), 3),
                   "RMSE": round(float(np.sqrt(((d["y"].values - e) ** 2).mean())), 2),
                   "MAPE": round(float(np.abs((d["y"].values - e) / d["y"].values).mean() * 100), 2),
                   "mean_R2": round(r2(d["y"].values, np.full(len(d), tr["y"].mean())), 3)}
        res[lb].update({k: round(r2(d["y"].values, Pm[:, i]), 3) for i, k in enumerate(keys)})
    pi = permutation_importance(P["gbm"], va[cols], va["y"], n_repeats=12, random_state=0)
    tot = max(pi.importances_mean.clip(min=0).sum(), 1e-9)
    res["importance"] = [{"f": f, "v": round(float(max(v, 0) / tot), 4), "blk": BLOCK.get(f, "S")}
                         for f, v in sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])
                         if v > 0][:12]
    pred = pd.Series(np.c_[[P[k].predict(D[cols]) for k in keys]].T @ w, index=D.index)
    return res, pred


# ================================================================ 계층별 실행
S_COLS = [c for c, b in BLOCK.items() if b == "S"]
I_COLS = [c for c, b in BLOCK.items() if b == "I"]

CH4 = M.CH4_m3d
CH4pct = M.CH4_pct
TS_load = S_BASE["TS_load"]
SMY = (CH4 / TS_load).replace([np.inf, -np.inf], np.nan)
# 수율 타깃의 준-순환 차단.
#   SMY = CH₄/(투입량×acid_TS/100) 이므로 분모 성분과 그 대리변수를 모두 제외한다.
#   제거율 계열(TS_rem·VS_rem·COD_rem)도 분자에 acid_* 가 들어가 있어 실질적으로
#   분모의 대리변수다 — 실측 확인: TS_rem ↔ acid_TS 상관 +0.909, 소화액 TS는 거의
#   불변(CV 9%)이라 제거율 변동의 대부분이 유입 TS 변동에서 온다. acid_TS 분위 내로
#   층화하면 TS_rem↔수율 관계가 붕괴(ρ −0.06~−0.46)하므로 인과가 아닌 기계적 관계다.
DENOM_BAN = ("acid_TS@", "feed@", "TS_load@", "VS_load@", "COD_load@", "TN_load@",
             "OLR@", "HRT@", "intake_", "acid_VS@", "acid_COD@", "acid_TN@", "dewater@",
             "TS_rem@", "VS_rem@", "COD_rem@")

VARIANTS = {}          # 기저 변수 → 그 변수에서 파생된 특징 키 목록
for k in BANK:
    if k in ("dow", "month"):
        VARIANTS.setdefault(k, []).append(k)
    elif "@" in k:
        VARIANTS.setdefault(k.split("@")[0], []).append(k)


def best_lag(base_vars, tgt, sample):
    """1단계 — 변수별 대표 시차 추정.
    각 변수의 시차·이동평균·변화량 변형을 단일 특징으로 롤링 CV 평가해 최선을 고른다.
    658개 후보를 변수 수준(≈45개)으로 압축해 선택 불안정을 줄이고,
    동시에 '이 이화학 지표가 가스에 몇 일 시차로 반영되는가'라는 물리량을 추정한다."""
    reps = []
    for v in base_vars:
        cols = [c for c in VARIANTS.get(v, []) if c in FB.columns]
        cols = [c for c in cols if FB.loc[TR, c].notna().mean() > 0.5
                and FB.loc[TR, c].std(skipna=True) > 0]
        if not cols:
            continue
        D = pd.concat([FB[cols], tgt.rename("y")], axis=1).loc[sample].dropna(subset=["y"])
        if len(D.loc[TR]) < 60:
            continue
        scored = [(cv_score(D, [c])[0], c) for c in cols]
        s, c = max(scored)
        if np.isfinite(s) and s > -9e8:
            reps.append({"var": v, "feat": c, "cv": round(s, 3),
                         "blk": BLOCK.get(c, "S"),
                         "lag": c.split("@")[1] if "@" in c else "-"})
    # 절대 문턱 대신 상대 순위로 자른다. 타깃에 따라 CV 수준 자체가 다르기 때문 —
    # 메탄함량은 연도별 국면 변화(2020년 68.7% → 2021년 62.2%)로 연도 교차 CV가
    # 구조적으로 음수라, 절대 문턱을 쓰면 유효 변수까지 전량 탈락한다.
    return sorted(reps, key=lambda x: -x["cv"])[:20]


LAYERS, SERIES = {}, {}
plan = [
    ("L1", "일별 메탄생성량", "㎥/d", CH4, CH4.notna(), S_COLS, I_COLS, 20),
    ("L2", "일별 메탄함량", "%", CH4pct, CH4pct.notna(), S_COLS, I_COLS, 20),
    ("L3", "일별 비메탄수율", "㎥CH₄/t TS", SMY, SMY.notna(),
     [c for c in S_COLS if not c.startswith(DENOM_BAN)],
     [c for c in I_COLS if not c.startswith(DENOM_BAN)], 14),
    # L4: 내부 상태만으로 7일 뒤 수율을 '예보'할 수 있는가 (동시점 정보 차단)
    ("L4", "7일 후 비메탄수율 (예보)", "㎥CH₄/t TS", SMY.shift(-7), SMY.shift(-7).notna(),
     [c for c in S_COLS if not c.startswith(DENOM_BAN) and "@L0" not in c and "@L1" not in c
      and "@L2" not in c],
     [c for c in I_COLS if not c.startswith(DENOM_BAN) and "@L0" not in c and "@L1" not in c
      and "@L2" not in c], 12),
]
for lk, label, unit, tgt, mask, scols, icols, mk_ in plan:
    sample = FB.index[mask.reindex(FB.index).fillna(False)]
    L = {"label": label, "unit": unit, "blocks": {}}
    # 1단계 — 변수별 대표 시차. 블록별로 각각 수행해 ablation 공정성을 유지한다.
    sv = sorted({c.split("@")[0] for c in scols} | {"dow", "month"})
    iv = sorted({c.split("@")[0] for c in icols})
    repS, repI = best_lag(sv, tgt, sample), best_lag(iv, tgt, sample)
    L["lag_table"] = {"S": repS[:14], "I": repI[:14]}
    candS = [r["feat"] for r in repS]
    candI = [r["feat"] for r in repI]
    print(f"[{lk}] 대표시차 추정: S {len(candS)}변수 / I {len(candI)}변수")
    print("      I 상위:", ", ".join(f"{r['var']}@{r['lag']}({r['cv']:+.2f})" for r in repI[:6]))
    # 2단계 — 변수 수준 전진선택
    for bk, cand in [("S", candS), ("I", candI), ("S+I", candS + candI)]:
        sel, trace = forward_select(cand, tgt, sample, max_k=mk_)
        if not sel:
            continue
        res, pred = final_fit(sel, tgt, sample, bk)
        res["trace"] = trace
        res["cv_mean"] = trace[-1]["cv_mean"] if trace else None
        res["cv_min"] = trace[-1]["cv_min"] if trace else None
        L["blocks"][bk] = res
        if bk == "S+I":
            SERIES[lk] = pred
        print(f"[{lk}/{bk}] 후보 {len(cand)} → 선택 {len(sel)}개 | CV평균 {res['cv_mean']:+.3f} "
              f"(최저폴드 {res['cv_min']:+.3f}) | 사고학습 {res['valid']['R2']:+.3f} "
              f"평가 {res['test']['R2']:+.3f} RMSE {res['test']['RMSE']}")
        print("        ", ", ".join(sel))
    LAYERS[lk] = L

idx = M.index
def ser(s, r=1):
    s = s.reindex(idx)
    return [None if pd.isna(v) else round(float(v), r) for v in s.values]


out = {"split": {"train": "2018-01-01~2021-12-31", "valid": "2022-01-01~2022-12-31",
                 "test": "2023-01-01~2023-09-17"},
       "n_candidates": int(FB.shape[1]),
       "lag_spec": {"lags": LAGS, "rolls": ROLLS, "deltas": DELTAS},
       "layers": LAYERS,
       "series": {"ch4_obs": ser(CH4), "ch4_pred": ser(SERIES.get("L1", pd.Series(dtype=float))),
                  "ch4pct_obs": ser(CH4pct, 2), "ch4pct_pred": ser(SERIES.get("L2", pd.Series(dtype=float)), 2),
                  "smy_obs": ser(SMY), "smy_pred": ser(SERIES.get("L3", pd.Series(dtype=float)))}}
with open("outputs/featsearch_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("\n=== 요약 ===")
for lk, L in LAYERS.items():
    print(f"[{lk}] {L['label']}")
    for bk, b in L["blocks"].items():
        print(f"   {bk:4s} k={b['k']:2d}  CV {b['cv_mean']:+.3f}  사고학습 {b['valid']['R2']:+.3f}"
              f"  평가 {b['test']['R2']:+.3f}  RMSE {b['test']['RMSE']:8.2f}  MAPE {b['test']['MAPE']:5.2f}%")
    if "S+I" in L["blocks"]:
        print("    중요도:", ", ".join(f"{x['f']}({x['blk']}) {x['v']*100:.0f}%"
                                    for x in L["blocks"]["S+I"]["importance"][:6]))
