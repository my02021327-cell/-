# -*- coding: utf-8 -*-
"""누적(cumulative) 프레임 기반 메탄생성 예측·수율 진단 모델링.

시차(lag) 계열 대신 누적량으로 공정을 기술한다. 혐기소화에서 물리적으로 의미 있는
누적량은 세 가지다.

  ① 체류 누적 유기물  — 지금 반응조 안에 남아 있는 기질량.
       CSTR 희석을 반영한 지수가중 누적:  Σ load(t−i)·exp(−i/HRT)
       그리고 단순 창 누적:              Σ_{i<W} load(t−i)
  ② 창 누적 투입 ↔ 창 누적 생산 — 수율의 적분 형태.
       일별 측정 노이즈가 적분으로 상쇄되어 전환효율이 안정적으로 드러난다.
  ③ 누적 저해물질 노출 — Σ FAN·dt, Σ TAN·dt.
       만성 암모니아 저해는 '순간 농도'가 아니라 '노출 총량'의 문제이므로
       이 시설의 지배적 실패 모드에 물리적으로 대응하는 표현이다.

■ 누적-누적 회귀의 함정 (§0에서 실증)
    누적 메탄과 누적 투입은 둘 다 단조증가하므로 회귀 R²가 자동으로 1에 가까워진다.
    이 R²는 성능이 아니다. 따라서 누적 곡선은 R²가 아니라 **기울기(=수율)** 로만
    해석하고, 예측 모델은 반드시 '창 누적'(구간 합) 위에서 세운다.

■ 중첩 창 문제
    30일 창을 매일 계산하면 인접일이 29/30 겹쳐 사실상 독립 표본이 아니다.
    학습은 중첩 창으로 하되, 평가는 **비중첩 창**으로도 병기한다.

■ 분할: 학습 2018-2021 / 사고학습 2022 / 평가 2023(최종 1회). 무작위 분할 금지.
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
from scipy import stats

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
TR = slice("2018-01-01", "2021-12-31")
VA = slice("2022-01-01", "2022-12-31")
TE = slice("2023-01-01", "2023-09-17")
HRT = 40.5

# ------------------------------------------------------------------ 기본량
feed = M.feed_AB_tpd.ffill(limit=2)
acid_TS, acid_VS, acid_COD = M.acid_TS_pct, M.acid_VS_pct, M.acid_CODcr_mgL
TS_load = (feed * acid_TS / 100).interpolate(limit=3)          # t TS/d
VS_load = (feed * acid_VS / 100).interpolate(limit=3)          # t VS/d
COD_load = (feed * acid_COD / 1000).interpolate(limit=3)       # kg COD/d
CH4 = M.CH4_m3d
biogas = M.biogas_AB_m3d

dig_pH = M[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
VFA = M[["VFA_A_mgL", "VFA_B_mgL"]].mean(axis=1)
ALK = pd.concat([M.ALK_A_mgL, M.ALK_B_mgL.where(M.flag_ALKB_corrupted == 0)], axis=1).mean(axis=1)
dig_COD = M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
dig_TS = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)
dig_VS = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)
dig_T = pd.concat([M.dig_T_A_C.where(M.flag_TA_sensor_fault == 0),
                   M.dig_T_B_C.where(M.flag_TB_sensor_fault == 0)], axis=1).mean(axis=1)
TAN_idx = (ALK - 14089) / 0.640
pKa = 0.09018 + 2729.92 / (38.0 + 273.15)
FAN_idx = TAN_idx / (1 + 10 ** (pKa - dig_pH))

OUT = {}

# ================================================================ §0 누적-누적 함정 실증
cum_ch4 = CH4.fillna(0).cumsum()
cum_vs = VS_load.fillna(0).cumsum()
cum_ts = TS_load.fillna(0).cumsum()
m = (CH4.notna()) & (VS_load.notna())
sl, ic, r, p, se = stats.linregress(cum_vs[m], cum_ch4[m])
# 대조군: 투입과 무관한 순수 잡음의 누적합도 같은 결과가 나오는가
rng = np.random.default_rng(0)
noise_cum = pd.Series(rng.normal(size=len(M)), index=M.index).cumsum()
sl2, _, r2n, p2, _ = stats.linregress(noise_cum[m], cum_ch4[m])
OUT["trap"] = {
    "cum_r2": round(float(r ** 2), 4),
    "cum_slope": round(float(sl), 1),
    "noise_r2": round(float(r2n ** 2), 4),
    "daily_r2": round(float(stats.linregress(VS_load[m], CH4[m])[2] ** 2), 4),
    "note": "누적-누적 R²는 단조증가가 만드는 값이며 성능 지표가 아니다",
}

# 연도별 누적곡선 기울기(=실효 수율) — 누적 데이터의 올바른 사용법.
# 정규화 기준(VS/TS/COD)을 모두 산출한다. 기준에 따라 결론이 뒤집히기 때문이다.
slopes = {}
for nm, Ld, dec in [("VS", VS_load, 1), ("TS", TS_load, 1), ("COD", COD_load, 3)]:
    row = {}
    for yr, g in M.groupby(M.index.year):
        mm = CH4.loc[g.index].notna() & Ld.loc[g.index].notna()
        if mm.sum() < 30:
            continue
        s_, i_, r_, p_, se_ = stats.linregress(Ld.loc[g.index][mm].cumsum(),
                                               CH4.loc[g.index][mm].cumsum())
        row[int(yr)] = {"slope": round(float(s_), dec), "ci": round(float(1.96 * se_), dec),
                        "n": int(mm.sum())}
    v = [x["slope"] for x in row.values()]
    slopes[nm] = {"by_year": row, "change_pct": round((v[-1] / v[0] - 1) * 100, 1)}
OUT["yearly_slope"] = slopes
OUT["vsts_by_year"] = {int(k): round(float(v), 3) for k, v in
                       (acid_VS / acid_TS).groupby(M.index.year).median().items()}

# ================================================================ 누적 특징
WINS = [7, 14, 30, 45, 60, 90]


def csum(s, w):
    return s.rolling(w, min_periods=max(3, w // 3)).sum()


def cmean(s, w):
    return s.rolling(w, min_periods=max(3, w // 3)).mean()


def hrt_weighted(s, tau=HRT, K=180):
    """CSTR 희석 반영 체류 누적: Σ load(t−i)·exp(−i/τ). 지금 반응조에 남은 기질량."""
    w = np.exp(-np.arange(K + 1) / tau)
    v = np.nan_to_num(s.values)
    o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


F = {}
# ① 체류 누적 유기물 (S)
F["retain_TS"] = hrt_weighted(TS_load)
F["retain_VS"] = hrt_weighted(VS_load)
F["retain_COD"] = hrt_weighted(COD_load)
for w in WINS:
    F[f"cumTS_{w}"] = csum(TS_load, w)
    F[f"cumVS_{w}"] = csum(VS_load, w)
    F[f"cumCOD_{w}"] = csum(COD_load, w)
    F[f"cumFeed_{w}"] = csum(feed, w)
# 창 평균 기질 성상 (S) — 양이 아닌 질
for w in [7, 30, 90]:
    F[f"acidpH_{w}"] = cmean(M.acid_pH, w)
    F[f"CODVS_{w}"] = cmean(acid_COD / (acid_VS * 10000), w)
    F[f"VSTS_in_{w}"] = cmean(acid_VS / acid_TS, w)
    F[f"manure_{w}"] = cmean(M.intake_manure_tpd / M.intake_total_tpd * 100, w)
# ② 누적 처리 이력 (S) — 반응조 순치·노후 대리
F["total_processed"] = TS_load.fillna(0).cumsum() / 1000
F["days_operated"] = pd.Series(np.arange(len(M)), index=M.index) / 365
# ③ 창 평균 내부 상태 + 누적 저해 노출 (I)
for w in [7, 30, 90]:
    F[f"digpH_{w}"] = cmean(dig_pH, w)
    F[f"VFA_{w}"] = cmean(VFA, w)
    F[f"ALK_{w}"] = cmean(ALK, w)
    F[f"vfaalk_{w}"] = cmean(VFA / ALK, w)
    F[f"digCOD_{w}"] = cmean(dig_COD, w)
    F[f"digTS_{w}"] = cmean(dig_TS, w)
    F[f"digVSTS_{w}"] = cmean(dig_VS / dig_TS, w)
    F[f"digT_{w}"] = cmean(dig_T, w)
for w in [30, 90, 180]:
    F[f"expFAN_{w}"] = csum(FAN_idx.ffill(limit=7), w) / 1000      # 누적 FAN 노출 [g·d/L]
    F[f"expTAN_{w}"] = csum(TAN_idx.ffill(limit=7), w) / 1000      # 누적 TAN 노출
    F[f"expVFA_{w}"] = csum(VFA.ffill(limit=7), w) / 1000
F["expFAN_total"] = FAN_idx.ffill(limit=7).fillna(0).cumsum() / 1000   # 전기간 누적 노출
F["dewater_30"] = cmean(M.dewater_tpd, 30)

FB = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
S_KEYS = [k for k in FB if k.startswith(("retain_", "cumTS", "cumVS", "cumCOD", "cumFeed",
                                         "acidpH", "CODVS", "VSTS_in", "manure",
                                         "total_processed", "days_operated"))]
I_KEYS = [k for k in FB if k not in S_KEYS]
print(f"누적 특징 {FB.shape[1]}개 (S {len(S_KEYS)} / I {len(I_KEYS)})")

# ================================================================ 타깃 (창 누적)
W = 30
Y_cum = csum(CH4, W)                                    # 창 누적 메탄 [㎥]
Y_yield = (Y_cum / csum(TS_load, W)).replace([np.inf, -np.inf], np.nan)   # 창 누적 수율
# 수율 타깃의 분모 성분 제외 (준-순환 차단)
YIELD_BAN = tuple(["cumTS", "cumVS", "cumCOD", "cumFeed", "retain_"])


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


FOLDS = [(slice("2018-01-01", "2019-12-31"), slice("2020-01-01", "2020-12-31")),
         (slice("2018-01-01", "2020-12-31"), slice("2021-01-01", "2021-12-31")),
         (slice("2018-01-01", "2021-12-31"), slice("2022-01-01", "2022-12-31"))]


def cv_mean(D, cs):
    ss = []
    for a, b in FOLDS:
        tr_, va_ = D.loc[a], D.loc[b]
        if len(tr_) < 40 or len(va_) < 15:
            continue
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            Ridge(alpha=10.0)).fit(tr_[cs], tr_["y"])
        ss.append(r2(va_["y"].values, mdl.predict(va_[cs])))
    return float(np.mean(ss)) if ss else -9e9


def forward(cand, tgt, max_k=8, tol=0.005):
    D = pd.concat([FB[cand], tgt.rename("y")], axis=1).dropna(subset=["y"])
    if len(D.loc[TR]) < 100:
        return [], -9e9
    chosen, best, rest = [], -9e9, list(cand)
    while len(chosen) < max_k and rest:
        sc, c = max(((cv_mean(D, chosen + [c]), c) for c in rest), key=lambda x: x[0])
        if sc - best < tol:
            break
        best, chosen = sc, chosen + [c]
        rest.remove(c)
    return chosen, best


def final(cols, tgt, cvm, tag):
    D = pd.concat([FB[cols], tgt.rename("y")], axis=1).dropna(subset=["y"])
    tr, va, te = D.loc[TR], D.loc[VA], D.loc[TE]
    mk = lambda e: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), e)
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
    res = {"tag": tag, "k": len(cols), "features": cols, "cv_mean": round(cvm, 3),
           "n_train": int(len(tr)), "n_test": int(len(te))}
    for lb, d in [("valid", va), ("test", te)]:
        e = np.c_[[P[k].predict(d[cols]) for k in keys]].T @ w
        res[lb] = {"R2": round(r2(d["y"].values, e), 3),
                   "RMSE": round(float(np.sqrt(((d["y"].values - e) ** 2).mean())), 2),
                   "MAPE": round(float(np.abs((d["y"].values - e) / d["y"].values).mean() * 100), 2)}
    # 비중첩 창 평가 (평가구간을 W일 간격으로 솎아냄 → 표본 간 겹침 제거)
    te_ns = te.iloc[::W]
    if len(te_ns) >= 4:
        e = np.c_[[P[k].predict(te_ns[cols]) for k in keys]].T @ w
        res["test_nonoverlap"] = {"n": int(len(te_ns)),
                                  "R2": round(r2(te_ns["y"].values, e), 3),
                                  "MAPE": round(float(np.abs((te_ns["y"].values - e) / te_ns["y"].values).mean() * 100), 2)}
    pi = permutation_importance(P["gbm"], va[cols], va["y"], n_repeats=12, random_state=0)
    tot = max(pi.importances_mean.clip(min=0).sum(), 1e-9)
    res["importance"] = [{"f": f, "v": round(float(max(v, 0) / tot), 4),
                          "blk": "I" if f in I_KEYS else "S"}
                         for f, v in sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])
                         if v > 0][:10]
    pred = pd.Series(np.c_[[P[k].predict(D[cols]) for k in keys]].T @ w, index=D.index)
    return res, pred


LAYERS, SERIES = {}, {}
plan = [("C1", f"{W}일 누적 메탄생성량", "㎥", Y_cum, S_KEYS, I_KEYS),
        ("C2", f"{W}일 누적 비메탄수율", "㎥CH₄/t TS", Y_yield,
         [k for k in S_KEYS if not k.startswith(YIELD_BAN)], I_KEYS)]
for lk, label, unit, tgt, sk, ik in plan:
    L = {"label": label, "unit": unit, "blocks": {}}
    for bk, pool in [("S", sk), ("I", ik), ("S+I", sk + ik)]:
        sel, cvm = forward(pool, tgt)
        if not sel:
            continue
        res, pred = final(sel, tgt, cvm, bk)
        L["blocks"][bk] = res
        if bk == "S+I":
            SERIES[lk] = pred
        ns = res.get("test_nonoverlap", {})
        print(f"[{lk}/{bk}] k={len(sel)} CV {cvm:+.3f} | 사고학습 {res['valid']['R2']:+.3f} "
              f"평가 {res['test']['R2']:+.3f} (비중첩 {ns.get('R2','—')}) MAPE {res['test']['MAPE']}%")
        print("        ", ", ".join(sel))
    LAYERS[lk] = L

# ================================================================ 누적 노출 ↔ 수율
expo = {}
for k in ["expFAN_30", "expFAN_90", "expFAN_180", "expTAN_90", "expVFA_90", "expFAN_total",
          "total_processed"]:
    d = pd.concat([FB[k], Y_yield.rename("y")], axis=1).dropna()
    rho, pv = stats.spearmanr(d[k], d["y"])
    dm = d.groupby(d.index.year).transform(lambda x: x - x.mean())
    rw, pw = stats.spearmanr(dm[k], dm["y"])
    expo[k] = [round(float(rho), 3), round(float(pv), 5), round(float(rw), 3), round(float(pw), 5)]
OUT["exposure_corr"] = expo

idx = M.index
def ser(s, r=1):
    s = s.reindex(idx)
    return [None if pd.isna(v) else round(float(v), r) for v in s.values]


OUT.update({
    "split": {"train": "2018-01-01~2021-12-31", "valid": "2022-01-01~2022-12-31",
              "test": "2023-01-01~2023-09-17"},
    "window": W, "n_features": int(FB.shape[1]),
    "layers": LAYERS,
    "series": {"cum_obs": ser(Y_cum), "cum_pred": ser(SERIES.get("C1", pd.Series(dtype=float))),
               "yield_obs": ser(Y_yield), "yield_pred": ser(SERIES.get("C2", pd.Series(dtype=float))),
               # 누적 곡선용: 두 변수가 모두 실측된 날만 누적한다.
               # (2018년 COD 실측이 51일뿐이라 결측을 0으로 채우면 곡선이 왜곡된다)
               "cum_cod": ser(COD_load[COD_load.notna() & CH4.notna()].cumsum()),
               "cum_ch4_obs": ser(CH4[COD_load.notna() & CH4.notna()].cumsum()),
               "retain_TS": ser(FB.retain_TS, 2), "expFAN_90": ser(FB.expFAN_90, 2)},
})
with open("outputs/cumulative_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)

print("\n=== §0 누적-누적 회귀의 함정 ===")
t = OUT["trap"]
print(f"  누적메탄~누적VS부하 R² = {t['cum_r2']}  (기울기 {t['cum_slope']} ㎥CH₄/t VS)")
print(f"  누적메탄~순수잡음누적 R² = {t['noise_r2']}   ← 무관한 변수도 같은 수준")
print(f"  일별 메탄~일별 VS부하 R² = {t['daily_r2']}   ← 실제 설명력")
print("\n=== 연도별 누적곡선 기울기 (실효 수율) — 정규화 기준별 ===")
for nm, blk in slopes.items():
    print(f"  [{nm} 기준] 2018→2023 {blk['change_pct']:+.1f}%  " +
          "  ".join(f"{y}:{v['slope']}" for y, v in blk["by_year"].items()))
print("\n=== 누적 노출 ↔ 창 누적 수율 (전체 / 연도내) ===")
for k, v in sorted(expo.items(), key=lambda x: -abs(x[1][2])):
    print(f"  {k:16s} 전체 {v[0]:+.3f}(p={v[1]:.4f})  연도내 {v[2]:+.3f}(p={v[3]:.4f})")
print("\n=== 요약 ===")
for lk, L in LAYERS.items():
    print(f"[{lk}] {L['label']}")
    for bk, b in L["blocks"].items():
        ns = b.get("test_nonoverlap", {})
        print(f"   {bk:4s} k={b['k']}  CV {b['cv_mean']:+.3f}  사고학습 {b['valid']['R2']:+.3f}"
              f"  평가 {b['test']['R2']:+.3f}  비중첩 {ns.get('R2','—')}  MAPE {b['test']['MAPE']}%")
    if "S+I" in L["blocks"]:
        print("    중요도:", ", ".join(f"{x['f']}({x['blk']}) {x['v']*100:.0f}%"
                                    for x in L["blocks"]["S+I"]["importance"][:6]))
