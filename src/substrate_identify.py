# -*- coding: utf-8 -*-
"""기질별 생분해도 분리 재시도 — 모델 구조 수정 + 고대비 구간 학습.

■ 앞선 판정의 정정
    "반입 조성 변동이 작아 기질을 분리할 수 없다"는 진단은 **연평균만 보고 내린 오판**이었다.
    일별 반입 비율은 가축분뇨 0~100%(표준편차 19.5%p), 하위 10% <12.6% / 상위 10% >54.4%로
    극단일이 각각 185일씩 존재한다. 설계행렬 조건수도 11로 심한 공선성이 아니다.
    따라서 BD가 경계에 붙은 원인은 조성 변동 부족이 아니라 **모델 구조**를 의심해야 한다.

■ 구조 결함 — 체류시간 분포와 반응속도의 혼동
    앞선 모델은 소화조를 CSTR 체류시간 분포 g(τ)=(1/HRT)e^(−τ/HRT) 만으로 표현했다.
    이 커널은 가중치 합이 1이므로 HRT를 바꿔도 **총 메탄량이 변하지 않는다** — 즉
    '체류시간이 짧으면 전환율이 떨어진다'는 가장 기본적인 반응공학이 모델에 없었다.
    올바른 형태는 1차 분해 반응과 희석 유출을 함께 푼 것이다.

        dS/dt = (S_in − S)/τ_H − k·S
        메탄 임펄스 응답  h(s) = k·exp(−(1/τ_H + k)·s)
        Σ_s h(s) = k·τ_H/(1 + k·τ_H) = 전환율 X          ← HRT와 k가 함께 전환율을 결정

    이 형태는 부수적으로 **식별가능성도 개선**한다. 기질마다 분해속도 k가 다르므로
    (가용성 음폐수 빠름, 리그노셀룰로스 분뇨 느림) 세 기질이 서로 다른 동적 신호를 갖는다.

■ 고대비 구간 학습
    조성 대비가 큰 날에 가중치를 주거나 그런 구간만 학습해 BD 분리를 재시도한다.
    대비 지표: |log(분뇨비율 / 음폐수비율)| 의 이탈도.

■ 불확실성: 블록 부트스트랩(30일 블록)으로 BD 신뢰구간을 산출한다.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
TR = slice("2018-01-01", "2021-12-31")
VA = slice("2022-01-01", "2022-12-31")
TE = slice("2023-01-01", "2023-09-17")
HRT = 12.0

# --- 화학양론 (앞 스크립트와 동일) -----------------------------------------
COMP = {"carb": (6, 10, 5, 0), "prot": (5, 7, 2, 1), "lipid": (57, 104, 6, 0)}
def buswell(c, h, o, n):
    return 22.414 * (c / 2 + h / 8 - o / 4 - 3 * n / 8) / (12.011 * c + 1.008 * h + 15.999 * o + 14.007 * n)
BTH = {k: buswell(*v) for k, v in COMP.items()}

SUB = {
    "foodww": {"name": "음폐수", "carb": .62, "prot": .16, "lipid": .22, "BD": .82,
               "VSfrac": .055 * .88, "k": 0.40},
    "manure": {"name": "가축분뇨", "carb": .52, "prot": .24, "lipid": .07, "BD": .45,
               "VSfrac": .045 * .75, "k": 0.08},
    "food":   {"name": "음식물", "carb": .58, "prot": .18, "lipid": .24, "BD": .69,
               "VSfrac": .180 * .90, "k": 0.30},
}
for s in SUB.values():
    s["B_th"] = sum(BTH[c] * s[c] for c in ["carb", "prot", "lipid"])

feed = M.feed_AB_tpd.ffill(limit=2)
CH4 = M.CH4_m3d
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = list(SUB)
I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)

VS_obs = (feed * M.acid_VS_pct.interpolate(limit=3) / 100)
_lit = sum(feed * R[k] * SUB[k]["VSfrac"] for k in SUB)
_d = pd.concat([_lit.rename("l"), VS_obs.rename("o")], axis=1).loc[TR].dropna()
K_VS = float(_d["o"].mean() / _d["l"].mean())
for k in SUB:
    SUB[k]["VSfrac"] *= K_VS


def mix(df, tau0, tmix, K=40):
    src = df.ffill(limit=7)
    w = np.exp(-np.arange(K + 1) / tmix); w /= w.sum()
    out = {}
    for c in src.columns:
        v = np.nan_to_num(src[c].values); o = np.zeros(len(v))
        for i, wv in enumerate(w):
            o[i:] += wv * v[: len(v) - i]
        out[c] = pd.Series(o, index=src.index).shift(tau0)
    Rm = pd.DataFrame(out)
    return Rm.div(Rm.sum(axis=1), axis=0)


def kinetic_kernel(s, k, hrt, K=200):
    """1차 분해 + 희석 유출:  h(τ) = k·exp(−(1/HRT + k)τ).
    가중치 합 = k·HRT/(1+k·HRT) = 전환율. RTD만 쓰던 앞 모델과 달리
    HRT가 짧으면 총 메탄량 자체가 줄어든다."""
    tau = np.arange(K + 1)
    w = k * np.exp(-(1.0 / hrt + k) * tau)
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


def design(tau0=0, tmix=3.0, hrt=HRT, ks=None, rtd_only=False):
    """열 i = B_th,i · VS_i 를 동역학 커널로 통과시킨 값. 계수는 BD_i."""
    rr = mix(R, tau0, tmix)
    cols = {}
    for k in SUB:
        vs = (feed * rr[k] * SUB[k]["VSfrac"]).ffill(limit=3)
        if rtd_only:                       # 앞 모델(구조 결함 재현용)
            w = np.exp(-np.arange(121) / hrt); w /= w.sum()
            v = np.nan_to_num((SUB[k]["B_th"] * vs * 1000).values); o = np.zeros(len(v))
            for i, wv in enumerate(w):
                o[i:] += wv * v[: len(v) - i]
            cols[k] = pd.Series(o, index=vs.index)
        else:
            kk = (ks or {}).get(k, SUB[k]["k"])
            cols[k] = kinetic_kernel(SUB[k]["B_th"] * vs * 1000, kk, hrt)
    return pd.DataFrame(cols)


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def fit(X, sl, w=None, base=True):
    d = pd.concat([X, CH4.rename("y")], axis=1).loc[sl].dropna()
    if len(d) < 40:
        return None
    A = d[list(SUB)].values
    lo, hi = [0.] * 3, [1.] * 3
    if base:
        A = np.c_[A, np.ones(len(d))]; lo, hi = lo + [0.], hi + [np.inf]
    y = d["y"].values
    if w is not None:
        ww = np.sqrt(w.reindex(d.index).fillna(0).values)
        A, y = A * ww[:, None], y * ww
    r = lsq_linear(A, y, bounds=(np.array(lo), np.array(hi)), max_iter=800)
    return {"bd": {k: float(v) for k, v in zip(SUB, r.x[:3])},
            "base": float(r.x[3]) if base else 0.0, "n": int(len(d))}


def sc(X, sl, par):
    d = pd.concat([X, CH4.rename("y")], axis=1).loc[sl].dropna()
    p = sum(par["bd"][k] * d[k] for k in SUB) + par["base"]
    t = d["y"].values
    return {"n": int(len(d)), "R2": round(r2(t, p.values), 3),
            "RMSE": round(float(np.sqrt(((t - p.values) ** 2).mean()))),
            "MAPE": round(float(np.abs((t - p.values) / t).mean() * 100), 2)}


OUT = {}
print("=== 0. 반입 조성 변동 실태 (앞선 '변동 부족' 판정의 검증) ===")
stat = (R * 100).describe(percentiles=[.1, .5, .9]).round(1)
for k in SUB:
    print(f"  {SUB[k]['name']:6s} 일별 평균 {stat.loc['mean', k]:5.1f}%  표준편차 {stat.loc['std', k]:5.1f}%p"
          f"  10~90% {stat.loc['10%', k]:5.1f}~{stat.loc['90%', k]:5.1f}%")
ann = (R * 100).groupby(R.index.year).mean().round(1)
print("  연평균은 안정적:", {int(y): float(ann.loc[y, 'manure']) for y in ann.index}, "(가축분뇨 %)")
OUT["variation"] = {"daily_std": {k: float(stat.loc["std", k]) for k in SUB},
                    "annual_mean_manure": {int(y): float(ann.loc[y, "manure"]) for y in ann.index}}

# ================================================================ 1. 구조 비교
print("\n=== 1. 모델 구조 비교 — RTD만 vs 1차분해+희석 ===")
res_struct = {}
for tag, kw in [("RTD만 (앞 모델)", {"rtd_only": True}), ("1차분해+희석 (수정)", {})]:
    X = design(**kw)
    par = fit(X, TR)
    conv = {k: SUB[k]["k"] * HRT / (1 + SUB[k]["k"] * HRT) for k in SUB}
    res_struct[tag] = {"bd": {k: round(par["bd"][k], 3) for k in SUB},
                       "base": round(par["base"]),
                       "train": sc(X, TR, par), "valid": sc(X, VA, par), "test": sc(X, TE, par),
                       "conv": {k: round(v, 3) for k, v in conv.items()}}
    r = res_struct[tag]
    print(f"  [{tag}] BD " + " ".join(f"{SUB[k]['name']} {r['bd'][k]:.2f}" for k in SUB) +
          f" | 기저 {r['base']} | 사고학습 R² {r['valid']['R2']:+.3f} 평가 {r['test']['R2']:+.3f}")
OUT["structure"] = res_struct

X = design()          # 수정 구조를 기준 설계행렬로
BASE_PAR = fit(X, TR)

# ================================================================ 2. 고대비 구간
print("\n=== 2. 고대비 구간 학습 ===")
Rm = mix(R, 0, 3.0)                                   # 소화조 도달 조성
lr = np.log((Rm["manure"] + 1e-3) / (Rm["foodww"] + 1e-3))
contrast = (lr - lr.rolling(90, min_periods=30).mean()).abs()      # 국소 평균 대비 이탈
res_ctr = {}
for q in [0.0, 0.3, 0.5, 0.7]:
    thr = contrast.loc[TR].quantile(q)
    sel = contrast >= thr
    idx = X.index[sel.reindex(X.index).fillna(False)]
    sub_tr = idx[(idx >= pd.Timestamp("2018-01-01")) & (idx <= pd.Timestamp("2021-12-31"))]
    d = pd.concat([X, CH4.rename("y")], axis=1).loc[sub_tr].dropna()
    if len(d) < 80:
        continue
    A = np.c_[d[list(SUB)].values, np.ones(len(d))]
    r_ = lsq_linear(A, d["y"].values, bounds=(np.array([0, 0, 0, 0.]), np.array([1, 1, 1, np.inf])))
    par = {"bd": {k: float(v) for k, v in zip(SUB, r_.x[:3])}, "base": float(r_.x[3])}
    res_ctr[f"상위 {int((1-q)*100)}%"] = {
        "n_train": int(len(d)), "thr": round(float(thr), 3),
        "bd": {k: round(par["bd"][k], 3) for k in SUB}, "base": round(par["base"]),
        "valid": sc(X, VA, par), "test": sc(X, TE, par),
        "at_bound": [k for k in SUB if par["bd"][k] >= .999 or par["bd"][k] <= .001]}
    rr = res_ctr[f"상위 {int((1-q)*100)}%"]
    print(f"  대비 상위 {int((1-q)*100):3d}% (n={rr['n_train']:3d})  BD " +
          " ".join(f"{SUB[k]['name']} {rr['bd'][k]:.2f}" for k in SUB) +
          f" | 기저 {rr['base']:5d} | 평가 R² {rr['test']['R2']:+.3f}"
          f" | 경계 {rr['at_bound'] or '없음'}")
OUT["contrast"] = res_ctr

# ================================================================ 3. 블록 부트스트랩
print("\n=== 3. 블록 부트스트랩 신뢰구간 (30일 블록, 400회) ===")
d_tr = pd.concat([X, CH4.rename("y")], axis=1).loc[TR].dropna()
blocks = [g for _, g in d_tr.groupby(pd.Grouper(freq="30D")) if len(g) > 5]
rng = np.random.default_rng(0)
boot = []
for _ in range(400):
    pick = rng.integers(0, len(blocks), len(blocks))
    dd = pd.concat([blocks[i] for i in pick])
    A = np.c_[dd[list(SUB)].values, np.ones(len(dd))]
    r_ = lsq_linear(A, dd["y"].values, bounds=(np.array([0, 0, 0, 0.]), np.array([1, 1, 1, np.inf])))
    boot.append(list(r_.x))
B = np.array(boot)
ci = {}
for i, k in enumerate(SUB):
    lo, med, hi = np.percentile(B[:, i], [2.5, 50, 97.5])
    at_lo = float((B[:, i] <= 0.001).mean() * 100)
    at_hi = float((B[:, i] >= 0.999).mean() * 100)
    ci[k] = {"median": round(float(med), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
             "pct_at_0": round(at_lo, 1), "pct_at_1": round(at_hi, 1),
             "lit": SUB[k]["BD"]}
    print(f"  {SUB[k]['name']:6s} 중앙 {med:.2f}  95%CI [{lo:.2f}, {hi:.2f}]  "
          f"경계 접촉 0에서 {at_lo:.0f}% / 1에서 {at_hi:.0f}%  (문헌 {SUB[k]['BD']:.2f})")
ci["base"] = {"median": round(float(np.median(B[:, 3]))),
              "ci": [round(float(np.percentile(B[:, 3], 2.5))), round(float(np.percentile(B[:, 3], 97.5)))]}
print(f"  기저항  중앙 {ci['base']['median']} ㎥/d  95%CI {ci['base']['ci']}")
OUT["bootstrap"] = ci

# ================================================================ 4. k 동시 추정
print("\n=== 4. 분해속도 k 동시 탐색 (동적 신호로 기질 분리 시도) ===")
best = None
for k_ww in [0.2, 0.4, 0.8]:
    for k_mn in [0.03, 0.08, 0.15]:
        for k_fd in [0.15, 0.30, 0.60]:
            Xk = design(ks={"foodww": k_ww, "manure": k_mn, "food": k_fd})
            par = fit(Xk, TR)
            if not par:
                continue
            v = sc(Xk, VA, par)["R2"]
            if best is None or v > best["valid_R2"]:
                best = {"k": {"foodww": k_ww, "manure": k_mn, "food": k_fd},
                        "bd": {kk: round(par["bd"][kk], 3) for kk in SUB},
                        "base": round(par["base"]), "valid_R2": v,
                        "test": sc(Xk, TE, par)}
print(f"  최적 k: " + " ".join(f"{SUB[k]['name']} {v}" for k, v in best["k"].items()))
print(f"  BD: " + " ".join(f"{SUB[k]['name']} {best['bd'][k]:.2f}" for k in SUB) +
      f" | 사고학습 R² {best['valid_R2']:+.3f} 평가 {best['test']['R2']:+.3f}")
OUT["k_search"] = best

# 전환율 (수정 구조에서의 물리량)
OUT["conversion"] = {k: {"k": SUB[k]["k"], "HRT": HRT,
                         "X": round(SUB[k]["k"] * HRT / (1 + SUB[k]["k"] * HRT), 3)} for k in SUB}
OUT["sub"] = {k: {"name": SUB[k]["name"], "B_th": round(SUB[k]["B_th"], 4),
                  "BD_lit": SUB[k]["BD"], "k": SUB[k]["k"]} for k in SUB}
print("\n=== 전환율 X = k·HRT/(1+k·HRT), HRT 12일 ===")
for k, v in OUT["conversion"].items():
    print(f"  {SUB[k]['name']:6s} k={v['k']:.2f}/d → X={v['X']:.3f}  "
          f"(실효수율 {SUB[k]['B_th']*SUB[k]['BD']*v['X']:.3f} m³CH₄/kg VS)")

with open("outputs/identify_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/identify_results.json")


# ================================================================ 5. COD 기준 재구축
# VS 기준 모델은 열역학적으로 성립하지 않는다: 관측 수율 0.745 m³CH₄/kg VS가
# 혼합기질 이론 상한(0.57)의 131%이고 96%의 날이 이를 초과한다. acid_VS 측정이
# 구조적으로 과소평가된 결과이며(DQ-08), 그래서 BD를 모두 1.0으로 밀어도 총량이
# 모자라 부족분이 전부 기저항으로 흘러갔다 — 이것이 BD가 경계에 붙은 진짜 원인이다.
# COD 기준은 물질수지가 닫힌다: 투입 COD 기준 0.242(이론 0.35의 69%),
# 제거 COD 기준 0.298(85%, 문헌 창 83~90% 이내), 상한 초과일 3%/14%.
B_COD = 0.35          # m³CH₄/kg COD — Buswell 화학양론상 모든 유기물에 공통인 상한
CODC = {"foodww": 140.0, "manure": 50.0, "food": 250.0}     # kg COD/t 습중량 (문헌 중앙값)
COD_obs = (feed * M.acid_CODcr_mgL / 1000)
_l = sum(feed * R[k] * CODC[k] for k in CODC)
_dd = pd.concat([_l.rename("l"), COD_obs.rename("o")], axis=1).loc[TR].dropna()
KC = float(_dd["o"].mean() / _dd["l"].mean())
CODC = {k: v * KC for k, v in CODC.items()}

Rm3 = mix(R, 0, 3.0)
Xc = pd.DataFrame({k: kinetic_kernel((feed * Rm3[k] * CODC[k] * B_COD).ffill(limit=3),
                                     SUB[k]["k"], HRT) for k in CODC})


def fit_c(sl, base=True, fixed=None, idx=None):
    d = pd.concat([Xc, CH4.rename("y")], axis=1).loc[sl].dropna()
    if idx is not None:
        d = d.loc[d.index.intersection(idx)]
    fixed = fixed or {}
    free = [k for k in CODC if k not in fixed]
    off = sum(fixed[k] * d[k] for k in fixed) if fixed else 0
    A = d[free].values
    lo, hi = [0.] * len(free), [1.] * len(free)
    if base:
        A = np.c_[A, np.ones(len(d))]; lo, hi = lo + [0.], hi + [np.inf]
    y = d["y"].values - (off.values if hasattr(off, "values") else off)
    r = lsq_linear(A, y, bounds=(np.array(lo), np.array(hi)))
    p = {"bd": dict(fixed), "base": float(r.x[-1]) if base else 0.0, "n": int(len(d))}
    for i, k in enumerate(free):
        p["bd"][k] = float(r.x[i])
    return p


def sc_c(p, sl):
    d = pd.concat([Xc, CH4.rename("y")], axis=1).loc[sl].dropna()
    q = sum(p["bd"][k] * d[k] for k in CODC) + p["base"]
    t = d["y"].values
    return round(float(1 - ((t - q.values) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 3)


par_c = fit_c(TR)
COD_RES = {"K_COD": round(KC, 3), "codc": {k: round(v, 1) for k, v in CODC.items()},
           "bd": {k: round(par_c["bd"][k], 3) for k in CODC}, "base": round(par_c["base"]),
           "train": sc_c(par_c, TR), "valid": sc_c(par_c, VA), "test": sc_c(par_c, TE)}
# 부트스트랩
dcb = pd.concat([Xc, CH4.rename("y")], axis=1).loc[TR].dropna()
blk = [g for _, g in dcb.groupby(pd.Grouper(freq="30D")) if len(g) > 5]
rng2 = np.random.default_rng(0); Bb = []
for _ in range(300):
    ss = pd.concat([blk[i] for i in rng2.integers(0, len(blk), len(blk))])
    A = np.c_[ss[list(CODC)].values, np.ones(len(ss))]
    Bb.append(lsq_linear(A, ss["y"].values,
                         bounds=(np.array([0, 0, 0, 0.]), np.array([1, 1, 1, np.inf]))).x)
Bb = np.array(Bb)
COD_RES["boot"] = {}
for i, k in enumerate(CODC):
    lo, me, hi = np.percentile(Bb[:, i], [2.5, 50, 97.5])
    COD_RES["boot"][k] = {"median": round(float(me), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                          "at0": round(float((Bb[:, i] <= .001).mean() * 100), 1),
                          "at1": round(float((Bb[:, i] >= .999).mean() * 100), 1),
                          "identified": bool((lo > 0.02) and (hi < 0.98))}
COD_RES["boot"]["base"] = {"median": round(float(np.median(Bb[:, 3]))),
                           "share_pct": round(float(np.median(Bb[:, 3]) / CH4.mean() * 100), 1)}
# 수율 정합성 (열역학 상한 대비)
_vs = (feed * M.acid_VS_pct / 100)
_y = pd.DataFrame({"c": CH4, "vs": _vs * 1000, "cod": COD_obs}).dropna()
COD_RES["yield_check"] = {
    "VS_median": round(float((_y.c / _y.vs).median()), 3), "VS_ceiling": 0.57,
    "VS_exceed_pct": round(float(((_y.c / _y.vs) > 0.57).mean() * 100), 1),
    "COD_median": round(float((_y.c / _y.cod).median()), 3), "COD_ceiling": B_COD,
    "COD_exceed_pct": round(float(((_y.c / _y.cod) > B_COD).mean() * 100), 1)}
OUT["cod_basis"] = COD_RES
with open("outputs/identify_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)

print("\n=== 5. COD 기준 재구축 ===")
print(f"  수율 정합성: VS 기준 {COD_RES['yield_check']['VS_median']} (상한 0.57, 초과 "
      f"{COD_RES['yield_check']['VS_exceed_pct']}%)  ← 열역학 위반")
print(f"               COD 기준 {COD_RES['yield_check']['COD_median']} (상한 0.35, 초과 "
      f"{COD_RES['yield_check']['COD_exceed_pct']}%)  ← 정합")
print(f"  BD: " + " ".join(f"{SUB[k]['name']} {COD_RES['bd'][k]:.2f}" for k in CODC) +
      f" | 기저 {COD_RES['base']} | 평가 R² {COD_RES['test']:+.3f}")
for k in CODC:
    b = COD_RES["boot"][k]
    print(f"    {SUB[k]['name']:6s} 95%CI [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]  "
          f"{'식별됨' if b['identified'] else '식별 불가'}")
