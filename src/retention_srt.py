# -*- coding: utf-8 -*-
"""잔존율 가중 적분 + SRT 탐색 — 기질 분리 재검증.

■ 두 가지 질문에 답한다
   Q1. 잔존율 가중 적분이 반영됐는가?
       → 반영돼 있다. 커널 h(τ)=k·exp(−(1/SRT+k)τ) 의 exp(−τ/SRT) 성분이
         정확히 '오늘 투입분 100%, τ일 전 투입분 exp(−τ/SRT)' 잔존율이다.
         다만 두 변형을 구분해 비교한다.
           (A) 잔존만  : 재고 = Σ exp(−τ/SRT)·Load(t−τ),  메탄 = θ·재고
                         분해로 소비되는 물질을 재고에서 빼지 않는다.
           (B) 잔존+분해: 재고 = Σ exp(−(1/SRT+k)τ)·Load(t−τ), 메탄 = k·재고
                         분해된 만큼 재고에서 제외 (물질수지 정합)
   Q2. '기질 분리 불가'는 시점 정렬 실패 탓인가?
       → 앞선 판정은 서술이 부정확했다. 식별되지 않는 것은 BD 자체가 아니라
         **BD와 COD 농도의 곱**이다. 두 값은 항상 곱으로만 나타나므로
         (θ_i = BD_i · C_i · 0.35) 개별 분해는 원리적으로 불가능하다.
         정작 검증할 것은 **θ_i(기질 t당 실효 메탄수율)가 식별되는가**이며,
         이는 시점 정렬(SRT·지연)에 따라 달라질 수 있다. 전 구간을 훑어 확인한다.

■ 재매개화
     CH₄(t) = Σ_i θ_i · [feed(t)·r_i(t−지연) 를 잔존 커널로 통과시킨 값]
     θ_i [㎥CH₄ / t 기질]  = BD_i × C_i(kg COD/t) × 0.35(㎥CH₄/kg COD)
     θ_i 는 관측으로 결정 가능한 양이고, BD_i 는 C_i 를 실측해야 분리된다.
"""
import json
import numpy as np
import pandas as pd

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
TR = slice("2018-01-01", "2021-12-31")
VA = slice("2022-01-01", "2022-12-31")
TE = slice("2023-01-01", "2023-09-17")
KS = {"foodww": .40, "manure": .08, "food": .30}          # 1차 분해속도 [1/d]
NM = {"foodww": "음폐수", "manure": "가축분뇨", "food": "음식물"}
LIT_BD = {"foodww": .82, "manure": .45, "food": .69}
LIT_C = {"foodww": 140., "manure": 50., "food": 250.}     # kg COD/t 습중량

feed = M.feed_AB_tpd.ffill(limit=2)
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = list(KS)
I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)
y = M.biogas_AB_m3d * (M.CH4_pct.ffill(limit=3) / 100)


def mix(df, t0, tm, K=40):
    """반입 → 저류조·산발효조 통과 후 소화조 도달 조성."""
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


def retention(s, srt, k=None, K=300):
    """잔존율 가중 적분.
       k=None  : (A) 잔존만  w(τ)=exp(−τ/SRT)          — 오늘 100%, τ일 전 exp(−τ/SRT)
       k 지정  : (B) 잔존+분해 w(τ)=k·exp(−(1/SRT+k)τ)  — 분해 소비분 제외, 메탄 발생률
    """
    tau = np.arange(K + 1)
    w = np.exp(-tau / srt) if k is None else k * np.exp(-(1.0 / srt + k) * tau)
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


def design(srt, mode="B", tau0=0, tmix=3.0):
    """기질별 재고(또는 메탄발생률) 열. 계수는 θ_i [㎥CH₄/t 기질]."""
    Rm = mix(R, tau0, tmix)
    cols = {}
    for kk in KS:
        ton = (feed * Rm[kk]).ffill(limit=3)                  # 기질 i의 소화조 투입 톤
        cols[kk] = retention(ton, srt, None if mode == "A" else KS[kk])
    return pd.DataFrame(cols)


def fit_theta(X, sl):
    d = pd.concat([X, y.rename("y")], axis=1).loc[sl].dropna()
    A = d[list(KS)].values
    th, *_ = np.linalg.lstsq(A, d["y"].values, rcond=None)     # 절편 없음(2차 감사 결론)
    return {k: float(v) for k, v in zip(KS, th)}


def score(X, sl, th):
    d = pd.concat([X, y.rename("y")], axis=1).loc[sl].dropna()
    p = sum(th[k] * d[k] for k in KS).values
    t = d["y"].values
    return {"n": int(len(d)),
            "R2": round(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 3),
            "MAPE": round(float(np.abs((t - p) / t).mean() * 100), 2)}


OUT = {}

# ================================================================ 1. SRT 탐색
print("=== 1. SRT 탐색 — 잔존율 가중 적분 (절편 없음, θ 자유 추정) ===")
print("  모드 A = 잔존만 / 모드 B = 잔존+분해")
print(f"  {'SRT':>5s} {'모드':>4s} | {'θ 음폐수':>9s} {'θ 분뇨':>8s} {'θ 음식물':>9s} | "
      f"{'사고학습R²':>9s} {'평가R²':>7s} {'평가MAPE':>8s}")
scan = []
for mode in ["A", "B"]:
    for srt in [5, 8, 12, 16, 20, 25, 30, 40, 50, 60]:
        X = design(srt, mode)
        th = fit_theta(X, TR)
        sv, st = score(X, VA, th), score(X, TE, th)
        scan.append({"srt": srt, "mode": mode, "theta": {k: round(v, 3) for k, v in th.items()},
                     "valid_R2": sv["R2"], "test_R2": st["R2"], "test_MAPE": st["MAPE"]})
        print(f"  {srt:5d} {mode:>4s} | {th['foodww']:9.2f} {th['manure']:8.2f} {th['food']:9.2f} | "
              f"{sv['R2']:9.3f} {st['R2']:7.3f} {st['MAPE']:7.2f}%")
best = max(scan, key=lambda x: x["valid_R2"])
print(f"  → 사고학습 기준 최적: SRT {best['srt']}일, 모드 {best['mode']} "
      f"(평가 R² {best['test_R2']:.3f}, MAPE {best['test_MAPE']}%)")
OUT["srt_scan"] = scan
OUT["best"] = best

# ================================================================ 2. 지연·혼합 동시 탐색
print("\n=== 2. 전단 지연·혼합까지 동시 탐색 (최적 SRT 근방) ===")
grid = []
for tau0 in [0, 1, 2, 3, 4, 5]:
    for tmix in [1, 3, 5, 8]:
        X = design(best["srt"], best["mode"], tau0, tmix)
        th = fit_theta(X, TR)
        grid.append({"tau0": tau0, "tmix": tmix, "valid_R2": score(X, VA, th)["R2"],
                     "test_R2": score(X, TE, th)["R2"],
                     "theta": {k: round(v, 3) for k, v in th.items()}})
grid.sort(key=lambda x: -x["valid_R2"])
for g in grid[:5]:
    print(f"  τ₀={g['tau0']}일 τ_mix={g['tmix']}일 → 사고학습 {g['valid_R2']:+.3f} 평가 {g['test_R2']:+.3f}  "
          f"θ " + " ".join(f"{NM[k]} {g['theta'][k]:.2f}" for k in KS))
gbest = grid[0]
OUT["delay_grid"] = grid[:8]

# ================================================================ 3. θ 식별가능성
print("\n=== 3. θ 식별가능성 — 블록 부트스트랩 (30일 블록, 500회) ===")
Xb = design(best["srt"], best["mode"], gbest["tau0"], gbest["tmix"])
Db = pd.concat([Xb, y.rename("y")], axis=1).loc[TR].dropna()
blocks = [g for _, g in Db.groupby(pd.Grouper(freq="30D")) if len(g) > 5]
rng = np.random.default_rng(0)
B = []
for _ in range(500):
    ss = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
    th, *_ = np.linalg.lstsq(ss[list(KS)].values, ss["y"].values, rcond=None)
    B.append(th)
B = np.array(B)
theta_ci = {}
for i, k in enumerate(KS):
    lo, me, hi = np.percentile(B[:, i], [2.5, 50, 97.5])
    rel = (hi - lo) / abs(me) if me != 0 else np.inf
    theta_ci[k] = {"median": round(float(me), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                   "rel_width": round(float(rel), 2),
                   "identified": bool(lo > 0 and rel < 1.0)}
    print(f"  θ {NM[k]:6s} = {me:6.2f} ㎥CH₄/t  95%CI [{lo:6.2f}, {hi:6.2f}]  "
          f"상대폭 {rel:.2f}  {'식별됨' if theta_ci[k]['identified'] else '불안정'}")
OUT["theta_ci"] = theta_ci
print(f"  θ 간 상관: ", {f"{NM[a]}-{NM[b]}": round(float(np.corrcoef(B[:, i], B[:, j])[0, 1]), 2)
                        for i, a in enumerate(KS) for j, b in enumerate(KS) if i < j})

# ================================================================ 4. θ → BD 환산
print("\n=== 4. θ 를 문헌 COD 농도로 나눠 생분해도 환산 ===")
print("  θ_i = BD_i × C_i × 0.35 이므로  BD_i = θ_i / (C_i × 0.35)")
bd_from_theta = {}
for k in KS:
    bd = theta_ci[k]["median"] / (LIT_C[k] * 0.35)
    lo = theta_ci[k]["ci"][0] / (LIT_C[k] * 0.35)
    hi = theta_ci[k]["ci"][1] / (LIT_C[k] * 0.35)
    bd_from_theta[k] = {"bd": round(float(bd), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                        "lit": LIT_BD[k], "C_assumed": LIT_C[k],
                        "physical": bool(0 <= bd <= 1)}
    flag = "물리적 범위" if 0 <= bd <= 1 else "★범위 이탈"
    print(f"  {NM[k]:6s} θ {theta_ci[k]['median']:6.2f} / (C {LIT_C[k]:.0f} × 0.35) = "
          f"BD {bd:5.2f} [{lo:.2f}, {hi:.2f}]  (문헌 {LIT_BD[k]:.2f})  [{flag}]")
OUT["bd_from_theta"] = bd_from_theta

# ================================================================ 5. 총량 정합
Xf = design(best["srt"], best["mode"], gbest["tau0"], gbest["tmix"])
thf = fit_theta(Xf, TR)
contrib = {k: float((thf[k] * Xf[k]).loc[TR].mean()) for k in KS}
tot = sum(contrib.values())
print("\n=== 5. 기질별 메탄 기여 (학습구간 평균) ===")
for k in KS:
    print(f"  {NM[k]:6s} {contrib[k]:7.0f} ㎥/d  ({contrib[k]/tot*100:5.1f}%)")
print(f"  합계 {tot:.0f} ㎥/d  vs 실측 평균 {y.loc[TR].mean():.0f} ㎥/d "
      f"({tot/y.loc[TR].mean()*100:.0f}%)")
OUT["contribution"] = {k: {"m3d": round(contrib[k]), "pct": round(contrib[k] / tot * 100, 1)} for k in KS}
OUT["total_check"] = {"model": round(tot), "observed": round(float(y.loc[TR].mean())),
                      "pct": round(tot / float(y.loc[TR].mean()) * 100, 1)}
OUT["final_score"] = {"train": score(Xf, TR, thf), "valid": score(Xf, VA, thf), "test": score(Xf, TE, thf)}
print(f"  성능: 학습 R² {OUT['final_score']['train']['R2']:+.3f} / "
      f"사고학습 {OUT['final_score']['valid']['R2']:+.3f} / 평가 {OUT['final_score']['test']['R2']:+.3f} "
      f"(MAPE {OUT['final_score']['test']['MAPE']}%)")

idx = M.index
def ser(s, r=0):
    s = s.reindex(idx)
    return [None if pd.isna(v) else round(float(v), r) for v in s.values]


OUT["series"] = {"obs": ser(y), "pred": ser(sum(thf[k] * Xf[k] for k in KS)),
                 **{f"sub_{k}": ser(thf[k] * Xf[k]) for k in KS}}
OUT["params"] = {"srt": best["srt"], "mode": best["mode"], "tau0": gbest["tau0"],
                 "tmix": gbest["tmix"], "theta": {k: round(thf[k], 3) for k in KS},
                 "k": KS}
with open("outputs/retention_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/retention_results.json")


# ================================================================ 6. 조성 정보 플라시보 검정
# "기질 분리 불가"가 시점 정렬 실패 탓인지 확인하는 결정적 검정.
# 조성 시계열을 블록 단위로 섞어도 성능이 유지되면 조성 정보는 허상이다.
print("\n=== 6. 조성 정보 플라시보 검정 (30일 블록 셔플 200회) ===")
Xr = design(best["srt"], best["mode"], gbest["tau0"], gbest["tmix"])
th_r = fit_theta(Xr, TR)
real = score(Xr, TE, th_r)["R2"]

# 대조 A: 조성을 전 기간 평균 상수로 고정
Rc = pd.DataFrame({k: pd.Series(R[k].mean(), index=R.index) for k in KS})
Rm_c = mix(Rc, gbest["tau0"], gbest["tmix"])
Xc = pd.DataFrame({k: retention((feed * Rm_c[k]).ffill(limit=3), best["srt"], KS[k]) for k in KS})
const_r2 = score(Xc, TE, fit_theta(Xc, TR))["R2"]

# 대조 B: 총 투입량 단일 커널
X1 = pd.DataFrame({"tot": retention(feed.ffill(limit=3), best["srt"], 0.35)})
D1 = pd.concat([X1, y.rename("y")], axis=1).dropna()
t1, *_ = np.linalg.lstsq(D1.loc[TR][["tot"]].values, D1.loc[TR]["y"].values, rcond=None)
d1 = D1.loc[TE]; p1 = d1[["tot"]].values @ t1; tt = d1["y"].values
tot_r2 = float(1 - ((tt - p1) ** 2).sum() / ((tt - tt.mean()) ** 2).sum())

rng2 = np.random.default_rng(0)
nb = len(R.index) // 30
sh = []
for _ in range(200):
    order = rng2.permutation(nb)
    arr = np.vstack([R.iloc[i * 30:(i + 1) * 30].values for i in order] +
                    [R.iloc[nb * 30:].values])[: len(R.index)]
    Rs = pd.DataFrame(arr, index=R.index, columns=R.columns)
    Rm_s = mix(Rs, gbest["tau0"], gbest["tmix"])
    Xs = pd.DataFrame({k: retention((feed * Rm_s[k]).ffill(limit=3), best["srt"], KS[k]) for k in KS})
    try:
        sh.append(score(Xs, TE, fit_theta(Xs, TR))["R2"])
    except Exception:
        pass
sh = np.array(sh)
pval = float((sh >= real).mean())
OUT["placebo"] = {"real_R2": round(real, 3), "shuffle_median": round(float(np.median(sh)), 3),
                  "shuffle_p95": round(float(np.percentile(sh, 95)), 3),
                  "shuffle_max": round(float(sh.max()), 3), "n_shuffle": int(len(sh)),
                  "p_value": round(pval, 4), "const_composition_R2": round(const_r2, 3),
                  "total_only_R2": round(tot_r2, 3),
                  "gain_over_total": round(real - tot_r2, 3)}
print(f"  실제 조성 평가 R² {real:+.3f}")
print(f"  셔플 200회: 중앙 {np.median(sh):+.3f}, 95백분위 {np.percentile(sh, 95):+.3f}, 최대 {sh.max():+.3f}")
print(f"  셔플이 실제를 넘는 비율 {pval*100:.1f}%  → {'조성 정보 유의' if pval < 0.05 else '조성 정보 무의미'}")
print(f"  대조: 조성 상수화 {const_r2:+.3f} / 총 투입량만 {tot_r2:+.3f}")
print(f"  → 조성 정보의 순 기여 +{real - tot_r2:.3f}")

# ================================================================ 7. 물리 제약 하의 해
print("\n=== 7. 물리 제약(BD≤1)과 문헌 BD 고정 시나리오 ===")
from scipy.optimize import lsq_linear as _lsq
trX = pd.concat([Xr, y.rename("y")], axis=1).loc[TR].dropna()
cons = {}
th_free = np.array([th_r[k] for k in KS])
ub = np.array([1.0 * LIT_C[k] * 0.35 for k in KS])
th_c = _lsq(trX[list(KS)].values, trX["y"].values, bounds=(np.zeros(3), ub)).x
th_l = np.array([LIT_BD[k] * LIT_C[k] * 0.35 for k in KS])
for tag, th_ in [("제약 없음", th_free), ("BD ≤ 1", th_c), ("문헌 BD 고정", th_l)]:
    thd = {k: float(v) for k, v in zip(KS, th_)}
    sv, st = score(Xr, VA, thd), score(Xr, TE, thd)
    bd = {k: round(float(thd[k] / (LIT_C[k] * 0.35)), 2) for k in KS}
    tot = float(sum(thd[k] * Xr[k].loc[TR].mean() for k in KS))
    cons[tag] = {"theta": {k: round(thd[k], 1) for k in KS}, "bd": bd,
                 "valid_R2": sv["R2"], "test_R2": st["R2"],
                 "total_m3d": round(tot), "total_pct": round(tot / float(y.loc[TR].mean()) * 100)}
    print(f"  [{tag:11s}] θ " + " ".join(f"{thd[k]:6.1f}" for k in KS) +
          f" → BD " + " ".join(f"{NM[k]} {bd[k]:5.2f}" for k in KS) +
          f" | 사고학습 {sv['R2']:+.3f} 평가 {st['R2']:+.3f} | 총량 {tot/float(y.loc[TR].mean())*100:.0f}%")
scale = float(y.loc[TR].mean() / sum(th_l[i] * Xr[k].loc[TR].mean() for i, k in enumerate(KS)))
cons["scale_needed"] = round(scale, 2)
cons["cod_needed"] = {k: round(LIT_C[k] * scale) for k in KS}
print(f"  문헌 BD로 총량을 맞추려면 COD 농도가 {scale:.2f}배 필요: " +
      " ".join(f"{NM[k]} {round(LIT_C[k]*scale)}" for k in KS) + " kg COD/t")
print("  (문헌 범위: 음폐수 100~180, 돈분슬러리 40~80, 음식물 200~300)")
OUT["constrained"] = cons

with open("outputs/retention_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
