# -*- coding: utf-8 -*-
"""기저항(base load) 분해 — 투입 급감 구간을 이용한 직접 추정.

■ 왜 필요한가
    기질별 화학양론 모델에서 추정된 기저항은 메탄의 34%(2,312 ㎥/d)에 달하는데,
    그 실체가 확인되지 않았다. 회귀로 얻은 값은 생분해도(BD)와 서로 맞바꿔지므로
    (분뇨 BD를 0→1로 강제해도 학습 R² 변동 0.003, 기저항이 정확히 보상)
    기질별 기여를 확정하려면 기저항을 **독립적으로** 고정해야 한다.

■ 원리
    투입이 급감하면 기질 유래 메탄은 감쇠하지만 기저항은 남는다. 따라서
    저부하 구간은 y절편(=기저항)에 대한 지렛대를 제공한다. 관건은
    **잔류 기질의 감쇠**와 **진짜 기저항**을 구분하는 것이다.

        y(t) = base + Σ_i BD_i·B_i·[기질 커널 통과분](t)
        투입 중단 후:  기질항은 시상수 1/(1/HRT + k)로 감쇠 → 잔여가 base

■ 세 갈래로 교차 검증한다
    ① 이벤트 스터디  — 저부하 에피소드를 정렬해 감쇠 궤적을 관찰
    ② 부하 구간별 외삽 — 기질 잠재량 S를 구간화해 S→0 극한을 비모수 추정
    ③ 감쇠곡선 적합  — y = base + A·exp(−t/τ) 를 에피소드에 직접 적합

■ 대상 지표는 바이오가스(일별 완전 관측, n=2,086)를 주로 쓴다.
    메탄생성량은 메탄함량 측정일(60.6%)에만 존재해 저부하 구간 표본이 얇다.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, curve_fit

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
TR = slice("2018-01-01", "2021-12-31")
HRT = 12.0
B_COD = 0.35

feed = M.feed_AB_tpd
gas = M.biogas_AB_m3d
ch4 = M.CH4_m3d
ch4pct = M.CH4_pct

I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = ["foodww", "manure", "food"]
I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)
CODC = {"foodww": 140.0, "manure": 50.0, "food": 250.0}
KS = {"foodww": 0.40, "manure": 0.08, "food": 0.30}
NM = {"foodww": "음폐수", "manure": "가축분뇨", "food": "음식물"}
COD_obs = (feed.ffill(limit=2) * M.acid_CODcr_mgL / 1000)
_l = sum(feed.ffill(limit=2) * R[k] * CODC[k] for k in CODC)
_d = pd.concat([_l.rename("l"), COD_obs.rename("o")], axis=1).loc[TR].dropna()
KC = float(_d["o"].mean() / _d["l"].mean())
CODC = {k: v * KC for k, v in CODC.items()}

OUT = {}

# ---------------------------------------------------------------- 저부하 구간 탐지
base60 = feed.rolling(60, min_periods=20, center=True).median()
rel = feed / base60


def episodes(thr=0.70, min_days=3, pad=25):
    low = (rel < thr).fillna(False)
    grp = (low != low.shift()).cumsum()
    out = []
    for _, idx in low.groupby(grp).groups.items():
        if not low.loc[idx].iloc[0] or len(idx) < min_days:
            continue
        s, e = idx[0], idx[-1]
        out.append({"start": s, "end": e, "n": len(idx),
                    "feed": float(feed.loc[s:e].mean()),
                    "base_feed": float(base60.loc[s:e].mean()),
                    "ratio": float(feed.loc[s:e].mean() / base60.loc[s:e].mean()),
                    "pre": s - pd.Timedelta(days=pad), "post": e + pd.Timedelta(days=pad)})
    return sorted(out, key=lambda x: x["ratio"])


EPS = episodes()
print("=== 저부하 에피소드 (60일 중앙값 대비 <70%, 3일 이상) ===")
for e in EPS:
    pre_gas = gas.loc[e["pre"]:e["start"] - pd.Timedelta(days=1)].mean()
    in_gas = gas.loc[e["start"]:e["end"]].mean()
    print(f"  {e['start']:%Y-%m-%d}~{e['end']:%Y-%m-%d} ({e['n']}일) 투입 {e['ratio']:.0%} "
          f"| 직전 25일 가스 {pre_gas:6.0f} → 구간 {in_gas:6.0f} ({in_gas/pre_gas-1:+.0%}) "
          f"{'  ※2018 고분산 국면' if e['start'].year == 2018 else ''}")
    e["pre_gas"] = round(float(pre_gas)); e["in_gas"] = round(float(in_gas))
    e["drop_pct"] = round(float(in_gas / pre_gas - 1) * 100, 1)
OUT["episodes"] = [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                    for k, v in e.items()} for e in EPS]

# ---------------------------------------------------------------- ① 이벤트 스터디
print("\n=== ① 이벤트 스터디 — 저부하 시작일 기준 정렬 (가스, 직전 25일 평균=100) ===")
rows = {}
for off in range(-10, 26):
    vals, fv = [], []
    for e in EPS:
        d = e["start"] + pd.Timedelta(days=off)
        if d in gas.index and not pd.isna(gas.get(d)) and e["pre_gas"] > 0:
            vals.append(float(gas[d]) / e["pre_gas"] * 100)
            if not pd.isna(feed.get(d)):
                fv.append(float(feed[d]) / e["base_feed"] * 100)
    if vals:
        rows[off] = {"gas": round(float(np.mean(vals)), 1), "n": len(vals),
                     "feed": round(float(np.mean(fv)), 1) if fv else None}
for off in range(-6, 22, 2):
    if off in rows:
        r = rows[off]
        print(f"  D{off:+3d}  투입 {str(r['feed'])+'%':>7s}  가스 {r['gas']:5.1f}%  (n={r['n']})")
OUT["event_study"] = rows
mins = min((v["gas"], k) for k, v in rows.items() if k >= 0)
print(f"  최저점: D{mins[1]:+d} 에서 가스 {mins[0]:.1f}%")
OUT["event_min"] = {"day": int(mins[1]), "gas_pct": round(float(mins[0]), 1)}

# ---------------------------------------------------------------- 기질 잠재량 S
def mix(df, tau0, tmix, K=40):
    src = df.ffill(limit=7)
    w = np.exp(-np.arange(K + 1) / tmix); w /= w.sum()
    o = {}
    for c in src.columns:
        v = np.nan_to_num(src[c].values); a = np.zeros(len(v))
        for i, wv in enumerate(w):
            a[i:] += wv * v[: len(v) - i]
        o[c] = pd.Series(a, index=src.index).shift(tau0)
    Rm = pd.DataFrame(o)
    return Rm.div(Rm.sum(axis=1), axis=0)


def kern(s, k, hrt, K=200):
    w = k * np.exp(-(1 / hrt + k) * np.arange(K + 1))
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


Rm = mix(R, 0, 3.0)
fe = feed.ffill(limit=2)
S_cols = {k: kern((fe * Rm[k] * CODC[k] * B_COD).ffill(limit=3), KS[k], HRT) for k in CODC}
S_tot = sum(S_cols.values())            # 기질 유래 잠재 메탄 [㎥CH₄/d], BD 미적용
ch4_eq = gas * (ch4pct.ffill(limit=3) / 100)     # 결측일 보완한 메탄 환산

# ---------------------------------------------------------------- ② 부하 구간별 외삽
print("\n=== ② 기질 잠재량 구간별 관측 메탄 (S→0 외삽) ===")
D = pd.concat([S_tot.rename("S"), ch4_eq.rename("y"), feed.rename("f")], axis=1).dropna()
D = D[D.S > 0]
q = np.percentile(D.S, [0, 2, 5, 10, 20, 35, 50, 65, 80, 92, 100])
bins = []
for i in range(len(q) - 1):
    seg = D[(D.S >= q[i]) & (D.S < q[i + 1] if i < len(q) - 2 else D.S <= q[i + 1])]
    if len(seg) < 15:
        continue
    bins.append({"S": round(float(seg.S.mean())), "y": round(float(seg.y.mean())),
                 "y_lo": round(float(seg.y.quantile(.25))), "n": int(len(seg)),
                 "feed": round(float(seg.f.mean()), 1)})
for b in bins:
    print(f"  S {b['S']:6d}  투입 {b['feed']:6.1f} t/d  관측메탄 {b['y']:6d} ㎥/d  (n={b['n']:4d})")
lowb = [b for b in bins[:4]]
A = np.c_[np.ones(len(lowb)), [b["S"] for b in lowb]]
coef = np.linalg.lstsq(A, np.array([b["y"] for b in lowb]), rcond=None)[0]
print(f"  → 하위 4구간 선형 외삽 절편 = {coef[0]:.0f} ㎥CH₄/d (기울기 {coef[1]:.3f})")
A2 = np.c_[np.ones(len(bins)), [b["S"] for b in bins]]
coef2 = np.linalg.lstsq(A2, np.array([b["y"] for b in bins]), rcond=None)[0]
print(f"  → 전 구간 선형 외삽 절편 = {coef2[0]:.0f} ㎥CH₄/d (기울기 {coef2[1]:.3f})")
OUT["bins"] = bins
OUT["extrap"] = {"low4_intercept": round(float(coef[0])), "low4_slope": round(float(coef[1]), 3),
                 "all_intercept": round(float(coef2[0])), "all_slope": round(float(coef2[1]), 3)}

# ---------------------------------------------------------------- ③ 감쇠 점근선 식별 가능성
# 앞선 시도에서 y = base + A·exp(−t/τ) 를 에피소드 시작 + 20일 창에 적합했더니
# base가 8,184~14,278 ㎥/d (평균 메탄의 120~210%)로 나왔다. 창 안에 **회복 구간**이
# 들어가 점근선이 회복된 수준을 잡은 것이다. 저부하가 3~8일밖에 지속되지 않으므로
# 감쇠 점근선은 애초에 식별될 수 없다. 얼마나 필요한지를 대신 계산한다.
print("\n=== ③ 감쇠 점근선 식별 가능성 검토 ===")
tau_eff = {k: 1.0 / (1.0 / HRT + KS[k]) for k in KS}
need = {k: round(float(-tau_eff[k] * np.log(0.10)), 1) for k in KS}   # 잔여 10%까지
for k in KS:
    print(f"  {NM[k]:6s} k={KS[k]:.2f}/d, HRT {HRT:.0f}일 → 유효 시상수 {tau_eff[k]:.1f}일, "
          f"잔여 10%까지 {need[k]:.0f}일 필요")
maxlen = max(e["n"] for e in EPS)
print(f"  관측된 최장 저부하 지속 {maxlen}일 → 가장 느린 기질(가축분뇨) 기준 잔여 "
      f"{np.exp(-maxlen/tau_eff['manure'])*100:.0f}%")
print("  → 현재 데이터로는 감쇠 점근선(=기저항) 직접 관측 불가. "
      f"최소 {max(need.values()):.0f}일 연속 저부하가 필요하다.")
OUT["decay_feasibility"] = {"tau_eff": {k: round(v, 2) for k, v in tau_eff.items()},
                            "days_to_10pct": need, "max_observed_episode": int(maxlen),
                            "residual_at_max": round(float(np.exp(-maxlen / tau_eff["manure"]) * 100), 1)}

# ---------------------------------------------------------------- ④ 저부하 가중 회귀
print("\n=== ④ 저부하 구간 가중 회귀 — 기저항과 생분해도 동시 추정 ===")
DD = pd.concat([pd.DataFrame(S_cols), ch4_eq.rename("y"), rel.rename("rel")], axis=1).dropna()
res_w = {}
for tag, sel in [("전체", DD.index),
                 ("저부하 하위 20%", DD.index[DD.rel <= DD.rel.quantile(.20)]),
                 ("저부하 하위 10%", DD.index[DD.rel <= DD.rel.quantile(.10)]),
                 ("저부하 하위 5%", DD.index[DD.rel <= DD.rel.quantile(.05)])]:
    d = DD.loc[sel]
    d = d[d.index.year <= 2021] if tag == "전체" else d
    if len(d) < 40:
        continue
    A_ = np.c_[d[list(CODC)].values, np.ones(len(d))]
    r_ = lsq_linear(A_, d["y"].values,
                    bounds=(np.array([0, 0, 0, 0.]), np.array([1, 1, 1, np.inf])))
    res_w[tag] = {"n": int(len(d)), "base": round(float(r_.x[3])),
                  "bd": {k: round(float(v), 3) for k, v in zip(CODC, r_.x[:3])}}
    print(f"  {tag:14s} n={len(d):4d}  기저 {r_.x[3]:6.0f}  BD " +
          " ".join(f"{NM[k]} {v:.2f}" for k, v in zip(CODC, r_.x[:3])))
OUT["weighted"] = res_w

# ---------------------------------------------------------------- ⑤ 절편의 가정 민감도
print("\n=== ⑤ 외삽 절편의 동역학 가정 민감도 ===")
sens = []
for hrt_ in [6, 12, 20, 40.5]:
    for kmul in [0.5, 1.0, 2.0]:
        Sc = {k: kern((fe * Rm[k] * CODC[k] * B_COD).ffill(limit=3), KS[k] * kmul, hrt_) for k in CODC}
        St = sum(Sc.values())
        dd = pd.concat([St.rename("S"), ch4_eq.rename("y")], axis=1).dropna()
        dd = dd[dd.S > 0]
        qq = np.percentile(dd.S, [0, 2, 5, 10, 20])
        bb = []
        for i in range(len(qq) - 1):
            sg = dd[(dd.S >= qq[i]) & (dd.S < qq[i + 1])]
            if len(sg) > 15:
                bb.append((sg.S.mean(), sg.y.mean()))
        if len(bb) >= 3:
            A_ = np.c_[np.ones(len(bb)), [x[0] for x in bb]]
            c_ = np.linalg.lstsq(A_, np.array([x[1] for x in bb]), rcond=None)[0]
            sens.append({"hrt": hrt_, "k_mult": kmul, "intercept": round(float(c_[0]))})
for x in sens:
    print(f"  HRT {x['hrt']:5.1f}일, k×{x['k_mult']:.1f} → 절편 {x['intercept']:6d} ㎥CH₄/d")
iv = [x["intercept"] for x in sens]
print(f"  절편 범위 {min(iv)}~{max(iv)} ㎥/d (중앙 {int(np.median(iv))})")
OUT["sensitivity"] = sens

# ---------------------------------------------------------------- ⑥ 종합

ests = [OUT["extrap"]["low4_intercept"], OUT["extrap"]["all_intercept"]] + iv
med = float(np.median(ests))
mean_ch4 = float(ch4_eq.mean())
OUT["summary"] = {"estimates": [int(x) for x in ests], "median": round(med),
                  "range": [int(min(ests)), int(max(ests))],
                  "share_of_mean_ch4": round(med / mean_ch4 * 100, 1),
                  "regression_estimate": res_w["전체"]["base"],
                  "regression_share": round(res_w["전체"]["base"] / mean_ch4 * 100, 1),
                  "lowload_estimate": res_w.get("저부하 하위 10%", {}).get("base"),
                  "mean_ch4": round(mean_ch4)}
print(f"\n=== ⑥ 종합 ===")
print(f"  외삽·민감도 기반 추정 범위 {min(ests)}~{max(ests)}, 중앙 {med:.0f} ㎥CH₄/d")
print(f"  평균 메탄 {mean_ch4:.0f} 대비 {med/mean_ch4*100:.0f}%")
print(f"  전체 회귀 기반 {res_w['전체']['base']} ({res_w['전체']['base']/mean_ch4*100:.0f}%) → "
      f"저부하 구간으로 좁히면 {res_w['저부하 하위 10%']['base']} ({res_w['저부하 하위 10%']['base']/mean_ch4*100:.0f}%)")

with open("outputs/baseload_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/baseload_results.json")


# ---------------------------------------------------------------- ⑦ 기저항 고정 후 BD 재추정
# 이 분석의 목적. 회귀에서 기저항과 생분해도가 서로 맞바꿔지므로, 기저항을
# 저부하 지렛대로 독립 추정해 고정하면 생분해도가 식별될 수 있다.

# 저부하 표본의 연도 편중 점검 (2018년은 고분산 국면이라 편중되면 해석 주의)
low10 = DD.index[DD.rel <= DD.rel.quantile(.10)]
yr_low = pd.Series(low10).dt.year.value_counts().sort_index().to_dict()
yr_all = pd.Series(DD.index).dt.year.value_counts().sort_index().to_dict()
OUT["lowload_year_mix"] = {"low10": {int(k): int(v) for k, v in yr_low.items()},
                           "all": {int(k): int(v) for k, v in yr_all.items()}}
print("\n=== ⑦ 저부하 표본의 연도 분포 점검 ===")
print("  저부하 하위10%:", {int(k): int(v) for k, v in yr_low.items()})
print("  전체        :", {int(k): int(v) for k, v in yr_all.items()})
sh18 = yr_low.get(2018, 0) / max(sum(yr_low.values()), 1) * 100
print(f"  2018년 비중 {sh18:.0f}% (전체 {yr_all.get(2018,0)/sum(yr_all.values())*100:.0f}%)"
      f" — {'편중 있음, 해석 주의' if sh18 > 30 else '심한 편중 없음'}")

print("\n=== ⑧ 기저항 고정 후 생분해도 재추정 (블록 부트스트랩 300회) ===")
DBD = pd.concat([pd.DataFrame(S_cols), ch4_eq.rename("y")], axis=1).dropna()
DBD = DBD[DBD.index.year <= 2021]
blocks = [g for _, g in DBD.groupby(pd.Grouper(freq="30D")) if len(g) > 5]
rng = np.random.default_rng(0)
bd_scen = {}
for tag, base_fix in [("자유 추정", None), ("기저 1,400 고정", 1400.0),
                      ("기저 900 고정", 900.0), ("기저 0 고정", 0.0)]:
    B = []
    for _ in range(300):
        ss = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if base_fix is None:
            A_ = np.c_[ss[list(CODC)].values, np.ones(len(ss))]
            lo_, hi_ = np.array([0, 0, 0, 0.]), np.array([1, 1, 1, np.inf])
            y_ = ss["y"].values
        else:
            A_ = ss[list(CODC)].values
            lo_, hi_ = np.array([0, 0, 0.]), np.array([1, 1, 1.])
            y_ = ss["y"].values - base_fix
        B.append(lsq_linear(A_, y_, bounds=(lo_, hi_)).x[:3])
    B = np.array(B)
    ent = {}
    for i, k in enumerate(CODC):
        lo, me, hi = np.percentile(B[:, i], [2.5, 50, 97.5])
        ent[k] = {"median": round(float(me), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                  "identified": bool(lo > 0.02 and hi < 0.98)}
    bd_scen[tag] = ent
    print(f"  [{tag}] " + "  ".join(
        f"{NM[k]} {ent[k]['median']:.2f} [{ent[k]['ci'][0]:.2f},{ent[k]['ci'][1]:.2f}]"
        f"{'✓' if ent[k]['identified'] else '✗'}" for k in CODC))
OUT["bd_by_base"] = bd_scen
lit = {"foodww": 0.82, "manure": 0.45, "food": 0.69}
print("  (문헌값: " + " ".join(f"{NM[k]} {v:.2f}" for k, v in lit.items()) + ")")
best_tag = min(bd_scen, key=lambda t: sum(abs(bd_scen[t][k]["median"] - lit[k]) for k in CODC))
print(f"  문헌값과 가장 가까운 시나리오: {best_tag}")
OUT["closest_to_lit"] = best_tag

with open("outputs/baseload_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
