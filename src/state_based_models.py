# -*- coding: utf-8 -*-
"""소화조 내부 상태 + 투입량·기질 성상 기반 메탄생성 예측 및 내부 진단 모델링.

이전 persistence 계열과의 결정적 차이: 자기회귀항(전일 가스/메탄) 전면 배제.
예측은 오직 ① 투입량·기질 성상(S) ② 소화조 내부 상태(I) 로부터만 이뤄진다.

■ 3계층 구조 (시간 스케일별로 답이 다르다)
  L1 일별 메탄생성량   CH4_m3d      ← 부하 지배. 내부 기여 검정
  L2 일별 메탄함량     CH4_pct      ← 가스 품질(경로 상태). 내부 기여 검정
  L3 월별 비메탄수율   SMY_month    ← 전환효율. 만성 저해가 드러나는 스케일

■ 수율 타깃의 순환성 차단 (중요)
  SMY = CH4 / (feed × acid_TS/100) 이므로 분모 성분(feed·acid_TS·TS_load·OLR·HRT·
  VS_load·COD_load·H)을 특징에서 전면 제외한다. 남는 기질 특징은 '양'이 아닌
  '성상 비율'뿐: 기질 환원도(COD/VS), VS/TS, 산발효조 pH.
  → "같은 부하에서 왜 전환효율이 달라지는가"라는 질문이 정직하게 성립한다.

■ ablation: S단독 / I단독 / S+I  → 내부 데이터의 기여를 계층별로 정량화
■ 진단: D1 수율손실 귀인 / D2 상태 판별기 / D3 내부상태 이상탐지
■ 분할: 학습 2018-2021 / 사고학습 2022(선택 전용) / 평가 2023(최종 1회)
■ 표본: 실측일만 사용 (실험실 주말 결측 보간 금지)
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import nnls
from scipy.stats import spearmanr

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]

TR = slice("2018-01-01", "2021-12-31")
VA = slice("2022-01-01", "2022-12-31")
TE = slice("2023-01-01", "2023-09-17")

# ------------------------------------------------------------------ 투입·기질(S)
feed = M.feed_AB_tpd.ffill(limit=2)
acid_TS, acid_VS, acid_COD = M.acid_TS_pct, M.acid_VS_pct, M.acid_CODcr_mgL
TS_load = feed * acid_TS / 100.0
COD_load = feed * acid_COD / 1000.0


def kernel(s, k=0.35, K=120):
    tau = np.arange(K + 1); w = k * np.exp(-k * tau); w /= w.sum()
    v = np.nan_to_num(s.interpolate(limit=3).values); out = np.zeros(len(v))
    for i, wv in enumerate(w):
        out[i:] += wv * v[: len(v) - i]
    return pd.Series(out, index=s.index)


H_TS = kernel(TS_load)
CODVS = acid_COD / (acid_VS * 10000)          # 기질 환원도 (ADM1 지질분획 프록시)
VSTS_in = acid_VS / acid_TS                   # 유기물 분율

# ------------------------------------------------------------------ 내부 상태(I)
dig_pH = M[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
VFA, ALK = M.VFA_A_mgL, M.ALK_A_mgL           # B계열 알칼리도 2023 오염(DQ-05) → A만
vfaalk = VFA / ALK
dig_COD = M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1).ffill(limit=3)
COD_rem = (acid_COD.ffill(limit=3) - dig_COD) / acid_COD.ffill(limit=3) * 100
dig_TS, dig_VS = M.dig_TS_A_pct, M.dig_VS_A_pct
dig_VSTS = dig_VS / dig_TS
TAN_idx = (ALK - 14089) / 0.640               # 알칼리도 역산 암모니아 부하지수(추세용)
pKa = 0.09018 + 2729.92 / (38.0 + 273.15)
FAN_idx = TAN_idx / (1 + 10 ** (pKa - dig_pH))
T_B = M.dig_T_B_C.where(M.flag_TB_sensor_fault == 0)

S_LOAD = {"TS_load": TS_load, "COD_load": COD_load, "H_TS": H_TS, "feed": feed}
S_QUAL = {"CODVS": CODVS.ffill(limit=3), "VSTS_in": VSTS_in, "acid_pH": M.acid_pH}
I_ALL = {"dig_pH": dig_pH, "VFA": VFA, "ALK": ALK, "vfaalk": vfaalk,
         "dig_COD": dig_COD, "COD_rem": COD_rem, "dig_TS": dig_TS, "dig_VSTS": dig_VSTS,
         "TAN_idx": TAN_idx, "FAN_idx": FAN_idx, "T_B": T_B}
F = pd.DataFrame({**S_LOAD, **S_QUAL, **I_ALL})
S_FULL, S_QUAL_K, I_K = list(S_LOAD) + list(S_QUAL), list(S_QUAL), list(I_ALL)

CH4 = M.CH4_m3d
SMY = (CH4 / TS_load).replace([np.inf, -np.inf], np.nan)
CORE = dig_pH.notna() & VFA.notna() & ALK.notna() & acid_TS.notna()


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def fit_block(cols, tgt, mask=None, small=False):
    """Ridge/RF/GBM 학습 → 사고학습에서 NNLS 결합 → 평가 1회. 표본 작으면 Ridge만."""
    D = pd.concat([F[cols], tgt.rename("y")], axis=1)
    D = D[(mask if mask is not None else CORE) & tgt.notna()]
    tr, va, te = D.loc[TR], D.loc[VA], D.loc[TE]
    if len(tr) < 25 or len(te) < 8:
        return None
    # 결측 대치·표준화를 파이프라인에 넣어 학습/재현 경로가 어긋날 수 없게 한다
    med = tr[cols].median()
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(tr[cols], tr["y"])
    P = {"ridge": lambda d: ridge.predict(d[cols])}
    if not small:
        rf = make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestRegressor(n_estimators=400, min_samples_leaf=6,
                                                 max_features=0.6, random_state=0,
                                                 n_jobs=-1)).fit(tr[cols], tr["y"])
        gbm = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3,
                                            l2_regularization=2.0, random_state=0).fit(tr[cols], tr["y"])
        P["rf"] = lambda d: rf.predict(d[cols])
        P["gbm"] = lambda d: gbm.predict(d[cols])
    keys = list(P)
    Pva = np.c_[[P[k](va) for k in keys]].T
    w, _ = nnls(Pva, va["y"].values)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(keys)) / len(keys)
    res = {"n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
           "weights": {k: round(float(v), 2) for k, v in zip(keys, w)}}
    for lab, d in [("valid", va), ("test", te)]:
        Pm = np.c_[[P[k](d) for k in keys]].T
        ens = Pm @ w
        res[lab] = {"R2": round(r2(d["y"].values, ens), 3),
                    "RMSE": round(float(np.sqrt(((d["y"].values - ens) ** 2).mean())), 2),
                    "MAPE": round(float(np.abs((d["y"].values - ens) / d["y"].values).mean() * 100), 2),
                    "mean_R2": round(r2(d["y"].values, np.full(len(d), tr["y"].mean())), 3)}
        res[lab].update({k: round(r2(d["y"].values, Pm[:, i]), 3) for i, k in enumerate(keys)})
    model = gbm if not small else ridge
    return res, (model, D, cols, med, "gbm" if not small else "ridge")


def imp_of(pack, cols_all=None):
    model, D, cols, med, kind = pack
    va = D.loc[VA]
    pi = permutation_importance(model, va[cols], va["y"], n_repeats=15, random_state=0)
    tot = max(pi.importances_mean.clip(min=0).sum(), 1e-9)
    return [{"f": f, "v": round(float(max(v, 0) / tot), 4), "blk": "I" if f in I_K else "S"}
            for f, v in sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1]) if v > 0][:10]


LAYERS, IMPS = {}, {}

# ---------------------------------------------------- L1 일별 메탄생성량
L1 = {"label": "일별 메탄생성량", "unit": "㎥/d", "blocks": {}}
for bk, cols in [("S", S_FULL), ("I", I_K), ("S+I", S_FULL + I_K)]:
    out = fit_block(cols, CH4)
    if out:
        L1["blocks"][bk] = out[0]
        if bk == "S+I":
            IMPS["L1"] = imp_of(out[1], cols)
LAYERS["L1"] = L1

# ---------------------------------------------------- L2 일별 메탄함량
L2 = {"label": "일별 메탄함량", "unit": "%", "blocks": {}}
for bk, cols in [("S", S_FULL), ("I", I_K), ("S+I", S_FULL + I_K)]:
    out = fit_block(cols, M.CH4_pct)
    if out:
        L2["blocks"][bk] = out[0]
        if bk == "S+I":
            IMPS["L2"] = imp_of(out[1], cols)
LAYERS["L2"] = L2

# ---------------------------------------------------- L3 월별 비메탄수율
# 월 집계(중앙값). 특징에 부하량 계열 없음 — 기질 성상 비율 + 내부 상태만.
mon = pd.concat([F[S_QUAL_K + I_K], SMY.rename("SMY")], axis=1)[CORE]
mon = mon.resample("MS").median()
mon = mon[mon["SMY"].notna()]
nobs = SMY[CORE].resample("MS").count()
mon = mon[nobs.reindex(mon.index).fillna(0) >= 8]      # 월 8일 이상 실측 월만

F_bak = F
F = mon.drop(columns="SMY")
MASK_M = pd.Series(True, index=mon.index)
L3 = {"label": "월별 비메탄수율", "unit": "㎥CH₄/t TS", "blocks": {}}
for bk, cols in [("기질성상", S_QUAL_K), ("내부상태", I_K), ("기질+내부", S_QUAL_K + I_K)]:
    out = fit_block(cols, mon["SMY"], mask=MASK_M, small=True)
    if out:
        L3["blocks"][bk] = out[0]
        if bk == "기질+내부":
            IMPS["L3"] = imp_of(out[1], cols)
            L3_pack = out[1]
LAYERS["L3"] = L3

# 월별 지표 ↔ 수율 상관. 연도 고정효과 통제(연평균 제거) 병기 — 추세 허위상관 배제
def corr_pair(col):
    s = mon[[col, "SMY"]].dropna()
    a = spearmanr(s[col], s["SMY"])
    dm = s.groupby(s.index.year).transform(lambda x: x - x.mean())
    w = spearmanr(dm[col], dm["SMY"])
    return [round(float(a[0]), 3), round(float(a[1]), 4),
            round(float(w[0]), 3), round(float(w[1]), 4)]


L3["corr"] = {k: corr_pair(k) for k in S_QUAL_K + I_K}   # [rho_all, p_all, rho_within, p_within]
mon_series = {"dates": mon.index.strftime("%Y-%m").tolist(),
              "smy": [round(float(v), 1) for v in mon["SMY"]],
              "ALK": [None if pd.isna(v) else round(float(v)) for v in mon["ALK"]],
              "FAN_idx": [None if pd.isna(v) else round(float(v)) for v in mon["FAN_idx"]],
              "vfaalk": [None if pd.isna(v) else round(float(v), 3) for v in mon["vfaalk"]]}
# 월별 예측 재현 (차트용)
_m, _D, _c, _med, _k = L3_pack
mon_series["smy_pred"] = [None if pd.isna(v) else round(float(v), 1)
                          for v in pd.Series(_m.predict(_D[_c]), index=_D.index).reindex(mon.index)]
F = F_bak

# ---------------------------------------------------- D1 수율손실 귀인 (월 단위)
qual_pack = fit_block(S_QUAL_K, mon["SMY"], mask=MASK_M, small=True) if False else None
F = mon.drop(columns="SMY")
outq = fit_block(S_QUAL_K, mon["SMY"], mask=MASK_M, small=True)
mq, Dq, cq, medq, _ = outq[1]
exp_q = pd.Series(mq.predict(Dq[cq]), index=Dq.index)
loss = (exp_q - mon["SMY"].reindex(exp_q.index)).rename("loss")     # 양수 = 기질 대비 부족
F = pd.concat([mon[I_K]], axis=1)
outl = fit_block(I_K, loss, mask=pd.Series(True, index=loss.index), small=True)
D1 = {"n": int(len(loss)), "expl_valid": outl[0]["valid"]["R2"], "expl_test": outl[0]["test"]["R2"]}
D1["corr"] = {k: [round(float(spearmanr(mon[k].reindex(loss.index), loss, nan_policy="omit")[0]), 3),
                  round(float(spearmanr(mon[k].reindex(loss.index), loss, nan_policy="omit")[1]), 4)]
              for k in I_K}
F = F_bak

# ---------------------------------------------------- D2 상태 판별기 (내부 변수만, 월 단위)
thr_lo = float(mon["SMY"].loc[TR].quantile(1 / 3))
lab_m = (mon["SMY"] < thr_lo).astype(int)              # 1 = 저조
Dc = pd.concat([mon[I_K], lab_m.rename("lab")], axis=1)
trc, vac, tec = Dc.loc[TR], Dc.loc[VA], Dc.loc[TE]
medc = trc[I_K].median()
scc = StandardScaler().fit(trc[I_K].fillna(medc))
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(Cs=8, cv=4, max_iter=2000, random_state=0).fit(
    scc.transform(trc[I_K].fillna(medc)), trc["lab"])


def clf_m(d):
    if d["lab"].nunique() < 2:
        return {"n": int(len(d)), "acc": round(float(accuracy_score(
            d["lab"], clf.predict(scc.transform(d[I_K].fillna(medc))))), 3), "auc": None}
    pr = clf.predict_proba(scc.transform(d[I_K].fillna(medc)))[:, 1]
    return {"n": int(len(d)), "auc": round(float(roc_auc_score(d["lab"], pr)), 3),
            "acc": round(float(accuracy_score(d["lab"], (pr > 0.5).astype(int))), 3),
            "pos": int(d["lab"].sum())}


coef = dict(zip(I_K, clf.coef_[0]))
D2 = {"threshold": round(thr_lo, 1), "train": clf_m(trc), "valid": clf_m(vac), "test": clf_m(tec),
      "coef": [{"f": k, "v": round(float(v), 3)} for k, v in
               sorted(coef.items(), key=lambda x: -abs(x[1]))]}

# ---------------------------------------------------- D3 내부상태 이상탐지 (일별)
NORM = ["dig_pH", "VFA", "ALK", "vfaalk", "dig_TS", "dig_VSTS", "COD_rem"]
base = F_bak[NORM][CORE].loc[TR].dropna()
smy_tr = SMY.reindex(base.index)
ref = base[smy_tr >= smy_tr.median()]                   # 정상 운전기(수율 중앙 이상)
mu, Si = ref.mean(), np.linalg.pinv(np.cov(ref.values, rowvar=False))
Xn = F_bak[NORM][CORE].dropna()
dev = Xn.values - mu.values
md = pd.Series(np.sqrt(np.einsum("ij,jk,ik->i", dev, Si, dev)), index=Xn.index)
thr = float(np.percentile(md.loc[TR], 95))
D3 = {"threshold_p95": round(thr, 2), "vars": NORM, "n": int(len(md)),
      "rate_by_year": {int(y): round(float((g > thr).mean() * 100), 1) for y, g in md.groupby(md.index.year)},
      "corr_with_SMY": round(float(spearmanr(md, SMY.reindex(md.index), nan_policy="omit")[0]), 3)}

# ---------------------------------------------------- 시계열 내보내기
idx = M.index
def ser(s, r=1):
    s = s.reindex(idx)
    return [None if pd.isna(v) else round(float(v), r) for v in s.values]


packs = {}
for lay, cols in [("L1", S_FULL + I_K), ("L2", S_FULL + I_K)]:
    o = fit_block(cols, CH4 if lay == "L1" else M.CH4_pct)
    m_, D_, c_, md_, k_ = o[1]
    packs[lay] = pd.Series(m_.predict(D_[c_]), index=D_.index)

# ---------------------------------------------------------- 계산기용 투명 선형모델
def export_linear(cols, tgt, src, mask):
    """표준화 Ridge 계수를 그대로 내보내 브라우저에서 동일 계산이 되게 한다."""
    D = pd.concat([src[cols], tgt.rename("y")], axis=1)[mask & tgt.notna()].dropna()
    tr, te = D.loc[TR], D.loc[TE]
    med = tr[cols].median()
    sc = StandardScaler().fit(tr[cols])
    rg = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(tr[cols]), tr["y"])
    pr = lambda d: rg.predict(sc.transform(d[cols]))
    return {"cols": cols, "mean": [float(v) for v in sc.mean_],
            "scale": [float(v) for v in sc.scale_],
            "coef": [float(v) for v in rg.coef_], "intercept": float(rg.intercept_),
            "test_R2": round(r2(te["y"].values, pr(te)), 3),
            "test_MAPE": round(float(np.abs((te["y"].values - pr(te)) / te["y"].values).mean() * 100), 2),
            "n_train": int(len(tr)),
            "p1_p99": {c: [round(float(tr[c].quantile(.01)), 3), round(float(tr[c].quantile(.99)), 3)]
                       for c in cols}}


CALC = {
    "ch4": export_linear(["TS_load", "H_TS", "CODVS", "acid_pH", "VSTS_in"], CH4, F_bak, CORE),
    "smy": export_linear(["CODVS", "VSTS_in", "acid_pH"], mon["SMY"], mon, MASK_M),
}

out = {
    "split": {"train": "2018-01-01~2021-12-31", "valid": "2022-01-01~2022-12-31",
              "test": "2023-01-01~2023-09-17"},
    "calc": CALC,
    "layers": LAYERS, "importances": IMPS,
    "D1": D1, "D2": D2, "D3": D3,
    "blocks": {"S_load": list(S_LOAD), "S_qual": S_QUAL_K, "I": I_K},
    "monthly": mon_series,
    "series": {"ch4_obs": ser(CH4), "ch4_pred": ser(packs["L1"]),
               "ch4pct_obs": ser(M.CH4_pct, 2), "ch4pct_pred": ser(packs["L2"], 2),
               "md": ser(md, 2)},
    "md_threshold": round(thr, 2),
}
with open("outputs/state_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("=== 계층별 ablation (앙상블 R²) ===")
for lk, L in LAYERS.items():
    print(f"[{lk}] {L['label']} ({L['unit']})")
    for bk, b in L["blocks"].items():
        print(f"   {bk:10s} n_tr={b['n_train']:4d}  사고학습 {b['valid']['R2']:+.3f}"
              f"  평가 {b['test']['R2']:+.3f}  RMSE {b['test']['RMSE']:8.2f}  MAPE {b['test']['MAPE']:5.2f}%")
print("\n=== 중요도 (전체 특징 모델) ===")
for k, v in IMPS.items():
    print(f"{k}:", ", ".join(f"{x['f']}({x['blk']}) {x['v']*100:.0f}%" for x in v[:6]))
print("\n=== L3 월별 수율 ↔ 지표 Spearman (전체 / 연도내) ===")
for k, v in sorted(LAYERS["L3"]["corr"].items(), key=lambda x: -abs(x[1][2])):
    blk = "I" if k in I_K else "S"
    print(f"   {k:10s}({blk}) 전체 {v[0]:+.3f}(p={v[1]:.4f})   연도내 {v[2]:+.3f}(p={v[3]:.4f})")
print("\n=== D1 수율손실 귀인(월) ===", D1["expl_valid"], D1["expl_test"], "n=", D1["n"])
for k, (rho, p) in sorted(D1["corr"].items(), key=lambda x: -abs(x[1][0]))[:6]:
    print(f"   loss vs {k:10s} rho={rho:+.3f} p={p:.4f}")
print("\n=== D2 판별기(월, 내부만) ===", "임계", D2["threshold"], D2["train"], D2["valid"], D2["test"])
print("   계수:", ", ".join(f"{c['f']} {c['v']:+.2f}" for c in D2["coef"][:6]))
print("\n=== D3 이상탐지 ===", D3["threshold_p95"], D3["rate_by_year"], "corr_SMY", D3["corr_with_SMY"])
print("\n=== 계산기 선형모델 ===")
for k, v in CALC.items():
    print(f"  {k}: {v['cols']}  test R²={v['test_R2']} MAPE={v['test_MAPE']}% n_tr={v['n_train']}")
