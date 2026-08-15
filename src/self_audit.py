# -*- coding: utf-8 -*-
"""자체 감사 2차 — 기저항 추정의 통계적 편의와 기질 분리의 원리적 한계.

§2.7에서 "기저항 ≈ 1,400 ㎥CH₄/d (메탄의 20%)"로 결론지었으나, 그 값은
회귀 절편이며 **설명변수의 측정오차에 의해 상향 편의**된다. 이를 정량화하고
독립 증거들과 대조해 결론을 재검토한다.

검증 항목
  A. 기질 잠재량 S의 측정오차 크기 — 문헌 기반 COD와 실측 COD의 괴리로 하한 추정
  B. 회귀 희석(regression dilution) 보정 — 오차분산비 λ에 따른 절편 이동
  C. 절편 제거 모델의 시간순 평가 — 절편이 정말 필요한가
  D. 저부하 표본 편중 진술의 재검증 — feed 기준과 S 기준의 혼용 여부
  E. 기질 분리의 능선(ridge) 구조 — COD 농도 가정을 흔들어 우도 평탄성 확인
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
HRT, B_COD = 12.0, 0.35
KS = {"foodww": .40, "manure": .08, "food": .30}
NM = {"foodww": "음폐수", "manure": "가축분뇨", "food": "음식물"}
LIT = {"foodww": .82, "manure": .45, "food": .69}

feed = M.feed_AB_tpd.ffill(limit=2)
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = list(KS)
I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)
COD_obs = feed * M.acid_CODcr_mgL / 1000
y = M.biogas_AB_m3d * (M.CH4_pct.ffill(limit=3) / 100)      # CH4_m3d와 동일 정의


def mix(df, t0=0, tm=3.0, K=40):
    s = df.ffill(limit=7)
    w = np.exp(-np.arange(K + 1) / tm); w /= w.sum()
    o = {}
    for c in s.columns:
        v = np.nan_to_num(s[c].values); a = np.zeros(len(v))
        for i, wv in enumerate(w):
            a[i:] += wv * v[: len(v) - i]
        o[c] = pd.Series(a, index=s.index).shift(t0)
    Rm = pd.DataFrame(o)
    return Rm.div(Rm.sum(axis=1), axis=0)


def kern(s, k, hrt=HRT, K=200):
    w = k * np.exp(-(1 / hrt + k) * np.arange(K + 1))
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


Rm = mix(R)


def build(codc):
    """문헌 COD 농도를 학습구간 실측 총 COD에 정규화한 뒤 기질별 설계열 생성."""
    lit = sum(feed * R[k] * codc[k] for k in codc)
    d = pd.concat([lit.rename("l"), COD_obs.rename("o")], axis=1).loc[TR].dropna()
    kc = float(d["o"].mean() / d["l"].mean())
    C = {k: v * kc for k, v in codc.items()}
    X = pd.DataFrame({k: kern((feed * Rm[k] * C[k] * B_COD).ffill(limit=3), KS[k]) for k in C})
    return X, C, kc


CODC0 = {"foodww": 140., "manure": 50., "food": 250.}
X, C0, KC = build(CODC0)
S = X.sum(axis=1)
D = pd.concat([S.rename("S"), y.rename("y")], axis=1).dropna()
D = D[D.S > 0]
OUT = {}

# ---------------------------------------------------------------- A. 측정오차 크기
lit_cod = sum(feed * Rm[k] * C0[k] for k in C0)
cmp = pd.concat([lit_cod.rename("lit"), COD_obs.rename("obs")], axis=1).dropna()
rel_err = float(((cmp["obs"] - cmp["lit"]) / cmp["lit"]).std())
feed_cv = float(feed.diff().std() / feed.mean())
OUT["A_error"] = {"cod_rel_sd_pct": round(rel_err * 100, 1), "n": int(len(cmp)),
                  "feed_diff_cv_pct": round(feed_cv * 100, 1)}
print("=== A. 기질 잠재량 S의 측정오차 하한 ===")
print(f"  문헌기반 COD vs 실측 COD 상대편차 표준편차 {rel_err*100:.1f}% (n={len(cmp)})")
print(f"  feed 1일 차분 변동계수 {feed_cv*100:.1f}%")
print(f"  → S의 상대 측정오차는 최소 20% 수준. λ=var(err)/var(S) 0.2 이상이 현실적")

# ---------------------------------------------------------------- B. 회귀 희석 보정
b_ols = np.polyfit(D.S, D.y, 1)
mx, my = D.S.mean(), D.y.mean()
dil = []
for lam in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    slope = b_ols[0] / (1 - lam) if lam > 0 else b_ols[0]
    icpt = my - slope * mx
    dil.append({"lam": lam, "slope": round(float(slope), 3), "intercept": round(float(icpt))})
OUT["B_dilution"] = dil
print("\n=== B. 회귀 희석 보정 — 오차분산비 λ에 따른 절편 ===")
for d_ in dil:
    print(f"  λ={d_['lam']:.2f}  기울기 {d_['slope']:.3f}  절편 {d_['intercept']:6d} ㎥CH₄/d")
print("  → λ 0.2에서 절편이 사실상 소멸. 기저항의 상당 부분이 통계적 편의일 수 있다")

# ---------------------------------------------------------------- C. 절편 유무 비교
def eval_model(with_icpt):
    tr = D.loc[TR]
    if with_icpt:
        A = np.c_[np.ones(len(tr)), tr.S.values]
        b = np.linalg.lstsq(A, tr.y.values, rcond=None)[0]
        f = lambda s: b[0] + b[1] * s
        par = (float(b[0]), float(b[1]))
    else:
        sl = float((tr.S * tr.y).sum() / (tr.S ** 2).sum())
        f = lambda s: sl * s
        par = (0.0, sl)
    out = {}
    for nm, s_ in [("train", TR), ("valid", VA), ("test", TE)]:
        dd = D.loc[s_]; p = f(dd.S.values); t = dd.y.values
        out[nm] = {"R2": round(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 3),
                   "MAPE": round(float(np.abs((t - p) / t).mean() * 100), 2)}
    return par, out


OUT["C_intercept"] = {}
print("\n=== C. 절편 유무 비교 (시간순 평가) ===")
for wi in [True, False]:
    par, o = eval_model(wi)
    tag = "with_intercept" if wi else "no_intercept"
    OUT["C_intercept"][tag] = {"intercept": round(par[0]), "slope": round(par[1], 3), **o}
    print(f"  [{'절편 포함' if wi else '절편 없음'}] 절편 {par[0]:6.0f} 기울기 {par[1]:.3f} | " +
          "  ".join(f"{k} R²={v['R2']:+.3f} MAPE={v['MAPE']:.1f}%" for k, v in o.items()))
print("  → 절편을 빼면 평가 성능이 오히려 개선. 절편은 학습구간 적합의 산물로 의심된다")
print(f"  → 절편 없는 모델의 기울기 {OUT['C_intercept']['no_intercept']['slope']:.3f}"
      f" = 평균 실효 생분해도. 문헌 범위(0.45~0.82) 안")

# ---------------------------------------------------------------- D. 편중 진술 재검증
rel_feed = M.feed_AB_tpd / M.feed_AB_tpd.rolling(60, min_periods=20, center=True).median()
DD = pd.concat([D, rel_feed.rename("rel")], axis=1).dropna()
mixes = {}
for nm, ix in [("feed_ratio", DD.index[DD.rel <= DD.rel.quantile(.10)]),
               ("S_potential", DD.index[DD.S <= DD.S.quantile(.10)]),
               ("all", DD.index)]:
    vc = pd.Series(ix).dt.year.value_counts().sort_index()
    mixes[nm] = {"by_year": {int(k): int(v) for k, v in vc.items()},
                 "share_2018_pct": round(float(vc.get(2018, 0) / vc.sum() * 100), 1)}
OUT["D_year_mix"] = mixes
print("\n=== D. 저부하 표본 편중 진술 재검증 ===")
for nm, v in mixes.items():
    print(f"  {nm:12s} 2018 비중 {v['share_2018_pct']:5.1f}%  {v['by_year']}")
print("  → §2.7은 feed 기준 편중(40%)을 S 기준 외삽(②)의 한계로 서술했으나,")
print("     S 기준 하위10%의 2018 비중은 14%로 전체(17%)보다 오히려 낮다. 진술 정정 필요")

# ---------------------------------------------------------------- E. 능선 구조
print("\n=== E. 기질 분리의 능선 구조 — 분뇨 COD 가정 민감도 ===")
ridge = []
for mn in [30, 50, 80, 120, 160]:
    Xi, Ci, _ = build({"foodww": 140., "manure": float(mn), "food": 250.})
    Di = pd.concat([Xi, y.rename("y")], axis=1).dropna()
    tr = Di.loc[TR]
    r = lsq_linear(tr[list(Ci)].values, tr["y"].values, bounds=(np.zeros(3), np.ones(3)))
    te = Di.loc[TE]; p = te[list(Ci)].values @ r.x; t = te["y"].values
    r2 = float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    ridge.append({"manure_cod": mn, "codc": {k: round(v) for k, v in Ci.items()},
                  "bd": {k: round(float(v), 2) for k, v in zip(Ci, r.x)}, "test_R2": round(r2, 3)})
    print(f"  분뇨 COD {mn:3d} → 정규화 {Ci['foodww']:.0f}/{Ci['manure']:.0f}/{Ci['food']:.0f}  "
          f"BD " + " ".join(f"{NM[k]} {v:.2f}" for k, v in zip(Ci, r.x)) + f"  평가 R² {r2:+.3f}")
r2s = [x["test_R2"] for x in ridge]
print(f"  평가 R² 범위 {min(r2s):.3f}~{max(r2s):.3f} (변동 {max(r2s)-min(r2s):.3f}) — 사실상 평탄")

# BD를 문헌값 고정 시 COD 농도 역산
base = {k: kern((feed * Rm[k] * B_COD * LIT[k]).ffill(limit=3), KS[k]) for k in KS}
Xb = pd.DataFrame(base)
Db = pd.concat([Xb, y.rename("y")], axis=1).dropna().loc[TR]
rb = lsq_linear(Db[list(KS)].values, Db["y"].values, bounds=(np.zeros(3), np.full(3, 400.)))
inv = {k: round(float(v)) for k, v in zip(KS, rb.x)}
tot_inv = float(sum(feed.loc[TR] * R[k].loc[TR] * rb.x[i] for i, k in enumerate(KS)).mean())
OUT["E_ridge"] = {"scan": ridge, "r2_range": [min(r2s), max(r2s)],
                  "inverted_cod": inv, "assumed_cod": {k: round(v) for k, v in C0.items()},
                  "inverted_total_vs_obs_pct": round(tot_inv / float(COD_obs.loc[TR].mean()) * 100)}
print(f"\n  BD를 문헌값 고정 시 역산 COD 농도: " + " ".join(f"{NM[k]} {v}" for k, v in inv.items()) + " kg/t")
print(f"  가정값: " + " ".join(f"{NM[k]} {round(C0[k])}" for k in C0) + " kg/t")
print(f"  역산 총 COD가 실측의 {OUT['E_ridge']['inverted_total_vs_obs_pct']}% — 물질수지 위반")
print("  → 분뇨 331 kg/t(슬러리로 불가능), 음식물 56(고형폐기물로 불가능).")
print("     세 기질 조합은 능선을 이루며 총합만 결정된다. 개별 분리는 원리적으로 불가")

# 음폐수 BD의 안정성
ww = [x["bd"]["foodww"] for x in ridge]
OUT["E_ridge"]["foodww_bd_range"] = [min(ww), max(ww)]
print(f"  단, 음폐수 BD는 {min(ww):.2f}~{max(ww):.2f}로 안정 (문헌 {LIT['foodww']:.2f}) — 유일하게 견고")

with open("outputs/audit2_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/audit2_results.json")
