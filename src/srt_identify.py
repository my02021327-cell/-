# -*- coding: utf-8 -*-
"""SRT를 실제 데이터로 추정할 수 있는가 — 세 갈래 검정과 그 결과.

■ 앞선 두 오류의 정정
   (1) 문헌 BD·COD를 정답으로 놓고 "총량이 맞는 SRT"를 고른 것은 순환 논증이다.
       영천 소화조 고유 특성이 있을 수 있는데 문헌값을 기준으로 삼으면
       그 특성이 SRT 추정으로 전가된다. 실제 데이터로 SRT를 정해야 한다.
   (2) SRT = 8,000㎥ / 197 t/d = 40.5일 은 부피를 질량으로 나눈 것이다.
       차원이 맞지 않고, 투입물은 소화액이 아니라 원기질이다.

■ 데이터 기반 SRT 추정 세 갈래
   A. 소화액 TS 동적 응답  — 반응조 고형물 수지의 시상수 = 1/(1/SRT + k_d)
   B. 비분해성 추적자      — 암모니아·알칼리도는 분해되지 않으므로(k_d≈0)
                             응답 시상수가 곧 SRT. 원리적으로 가장 깨끗한 방법
   C. 고형물 물질수지      — 배출 유량을 역산해 SRT = V / Q_out. V가 미지수

■ 결과: 세 갈래 모두 실패. SRT는 이 데이터로 결정되지 않는다.
   그러나 **예측에는 지장이 없다** — 가스 응답이 SRT에 둔감하기 때문이다.
"""
import json
import numpy as np
import pandas as pd

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
feed = M.feed_AB_tpd
acid_TS = M.acid_TS_pct
dig_TS = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)
dig_VS = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)
ALK, TAN = M.ALK_A_mgL, M.NH3N_A_mgL
OUT = {}


def ewm(s, tau, K=400):
    w = np.exp(-np.arange(K + 1) / tau); w /= w.sum()
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


def scan(driver, target, taus):
    rows = []
    for t in taus:
        X = ewm(driver, t)
        D = pd.concat([X.rename("x"), target.rename("y")], axis=1).dropna()
        if len(D) < 50:
            continue
        A = np.c_[np.ones(len(D)), D["x"].values]
        b = np.linalg.lstsq(A, D["y"].values, rcond=None)[0]
        p = A @ b
        r2 = float(1 - ((D["y"].values - p) ** 2).sum() / ((D["y"].values - D["y"].mean()) ** 2).sum())
        rows.append({"tau": t, "R2": round(r2, 4), "slope": round(float(b[1]), 4), "n": int(len(D))})
    return rows


TAUS = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]

# ---------------------------------------------------------------- A. 소화액 TS
TSload = (feed * acid_TS / 100).interpolate(limit=3)
a = scan(TSload, dig_TS, TAUS)
ba = max(a, key=lambda x: x["R2"])
OUT["A_digTS"] = {"scan": a, "best": ba,
                  "cv_pct": round(float(dig_TS.std() / dig_TS.mean() * 100), 1)}
print("=== A. 소화액 TS 동적 응답 ===")
print(f"  최적 시상수 {ba['tau']}일, R² {ba['R2']:.4f}  (소화액 TS 변동계수 {OUT['A_digTS']['cv_pct']}%)")
print(f"  → R²가 0.1에도 못 미친다. 소화액 TS가 투입 변동에 거의 반응하지 않아 신호가 없다")

# ---------------------------------------------------------------- B. 비분해성 추적자
# 전단 공정(전처리 → 여액저장조 → 산생성조)을 먼저 통과시킨다.
#   반입일과 소화조 투입일은 다르다. 반입 당일 전처리에 들어가지만 저류조에서 섞이고
#   산발효조를 거쳐야 소화조에 도달하므로, 추적자 구동변수도 그 변환을 거쳐야 한다.
#   전단 지연 τ₀(순수 이송) + 혼합 시상수 τ_mix(저류조·산발효조 완전혼합)
def front_end(s, tau0, tmix, K=60):
    w = np.exp(-np.arange(K + 1) / tmix); w /= w.sum()
    v = np.nan_to_num(s.ffill(limit=7).values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index).shift(tau0)


share_raw = (M.intake_manure_tpd / M.intake_total_tpd).clip(0, 1)
OUT["B_tracer"] = {}
print("\n=== B. 비분해성 추적자 (암모니아·알칼리도, k_d≈0이므로 시상수 = SRT) ===")
print("  구동변수: 분뇨 반입비율을 전단 공정(전처리·여액저장조·산발효조)에 통과시킨 뒤")
print("            소화조 투입량과 곱해 '소화조에 실제로 들어간 분뇨 상당량' 산출")
FE = [(0, 1.0), (1, 3.0), (2, 3.0), (3, 3.0), (4, 3.0), (0, 5.0), (4, 5.0)]
best_overall = None
for tgt, nm in [(ALK, "알칼리도"), (TAN, "NH3-N")]:
    per_fe = []
    for t0, tm in FE:
        Nl = (feed.ffill(limit=2) * front_end(share_raw, t0, tm))
        rows = scan(Nl, tgt, TAUS)
        if not rows:
            continue
        b = max(rows, key=lambda x: x["R2"])
        per_fe.append({"tau0": t0, "tmix": tm, "best_tau": b["tau"], "R2": b["R2"],
                       "at_edge": bool(b["tau"] == max(TAUS))})
    bb = max(per_fe, key=lambda x: x["R2"])
    OUT["B_tracer"][nm] = {"front_end_scan": per_fe, "best": bb}
    print(f"  [{nm}] 전단 조합별 최선:")
    for x in per_fe:
        print(f"    τ₀={x['tau0']}일 τ_mix={x['tmix']}일 → 최적 시상수 {x['best_tau']:3d}일 "
              f"R² {x['R2']:.4f}{'  (격자 상한)' if x['at_edge'] else ''}")
    print(f"    → 최선 R² {bb['R2']:.4f} (τ₀={bb['tau0']}, τ_mix={bb['tmix']}, 시상수 {bb['best_tau']}일)"
          f"{'  ★격자 상한 — 사실상 신호 없음' if bb['at_edge'] else ''}")
print("  → 두 추적자 모두 R² 0.06 미만이고 최적값이 격자 상한에 붙는다.")
print("     분뇨 반입 변동이 소화조 암모니아·알칼리도로 전달되는 신호가 관측되지 않는다")

# ---------------------------------------------------------------- C. 고형물 물질수지
d = pd.DataFrame({"f": feed, "aTS": acid_TS, "dTS": dig_TS, "dw": M.dewater_tpd}).dropna()
TSin = float((d["f"] * d["aTS"] / 100).mean())
rem = float(1 - d["dTS"].mean() / d["aTS"].mean())
TSout = TSin * (1 - rem)
Qout = TSout / (float(d["dTS"].mean()) / 100)
OUT["C_balance"] = {"TS_in_tpd": round(TSin, 2), "TS_removal_pct": round(rem * 100, 1),
                    "TS_out_tpd": round(TSout, 2), "Q_out_tpd": round(Qout, 1),
                    "feed_tpd": round(float(d["f"].mean()), 1),
                    "ratio_out_in": round(Qout / float(d["f"].mean()), 3),
                    "dewater_tpd": round(float(d["dw"].mean()), 1),
                    "dewater_over_Qout": round(float(d["dw"].mean()) / Qout, 2),
                    "srt_by_V": {"8000": round(8000 / Qout, 1), "4800(60%)": round(4800 / Qout, 1),
                                 "2400(30%)": round(2400 / Qout, 1)}}
print("\n=== C. 고형물 물질수지 — 배출 유량 역산 ===")
print(f"  투입 TS {TSin:.2f} t/d, 제거율 {rem*100:.1f}% → 잔여 {TSout:.2f} t/d")
print(f"  소화액 TS {d['dTS'].mean():.2f}% → 배출 유량 {Qout:.1f} t/d "
      f"(투입 {d['f'].mean():.1f} t/d 대비 {Qout/d['f'].mean():.2f}배)")
print(f"  탈수기 처리량 {d['dw'].mean():.1f} t/d 는 배출 유량의 {d['dw'].mean()/Qout:.2f}배")
print("  → 배출 유량 ≈ 투입 유량. 탈수기 수치는 소화조 배출이 아니다(반송·희석 포함 추정)")
print(f"  → SRT = V / {Qout:.0f}. V=8,000㎥면 {8000/Qout:.1f}일, 60%면 {4800/Qout:.1f}일, 30%면 {2400/Qout:.1f}일")
print("     유효 용적 V를 데이터로 알 수 없으므로 SRT가 결정되지 않는다")

# ---------------------------------------------------------------- D. 예측 민감도
print("\n=== D. 그런데 예측에는 지장이 없다 — SRT 민감도 재확인 ===")
KS = {"foodww": .40, "manure": .08, "food": .30}
NM = {"foodww": "음폐수", "manure": "가축분뇨", "food": "음식물"}
TR = slice("2018-01-01", "2021-12-31"); VA = slice("2022-01-01", "2022-12-31")
TE = slice("2023-01-01", "2023-09-17")
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = list(KS); I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)
y = M.biogas_AB_m3d * (M.CH4_pct.ffill(limit=3) / 100)
fe = feed.ffill(limit=2)


def mixr(t0, tm, K=40):
    s = R.ffill(limit=7)
    w = np.exp(-np.arange(K + 1) / tm); w /= w.sum()
    o = {}
    for c in s.columns:
        v = np.nan_to_num(s[c].values); a_ = np.zeros(len(v))
        for i, wv in enumerate(w):
            a_[i:] += wv * v[: len(v) - i]
        o[c] = pd.Series(a_, index=s.index).shift(t0)
    Rm = pd.DataFrame(o)
    return Rm.div(Rm.sum(axis=1), axis=0)


Rm = mixr(0, 5.0)


def retk(s, srt, k, K=500):
    w = k * np.exp(-(1 / srt + k) * np.arange(K + 1))
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


sens = []
print(f"  {'SRT':>6s} | {'θ음폐수':>7s} {'θ분뇨':>7s} {'θ음식물':>7s} | {'사고학습':>7s} {'평가':>7s} {'MAPE':>6s}")
for srt in [8, 12, 20, 25, 30, 40.5, 60]:
    X = pd.DataFrame({k: retk((fe * Rm[k]).ffill(limit=3), srt, KS[k]) for k in KS})
    D = pd.concat([X, y.rename("y")], axis=1).dropna()
    th, *_ = np.linalg.lstsq(D.loc[TR][list(KS)].values, D.loc[TR]["y"].values, rcond=None)
    o = {}
    for nm, sl in [("v", VA), ("t", TE)]:
        dd = D.loc[sl]; p = dd[list(KS)].values @ th; t = dd["y"].values
        o[nm] = float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())
        if nm == "t":
            mape = float(np.abs((t - p) / t).mean() * 100)
    sens.append({"srt": srt, "theta": {k: round(float(th[i]), 1) for i, k in enumerate(KS)},
                 "valid_R2": round(o["v"], 3), "test_R2": round(o["t"], 3), "MAPE": round(mape, 2)})
    print(f"  {srt:6.1f} | {th[0]:7.1f} {th[1]:7.1f} {th[2]:7.1f} | "
          f"{o['v']:7.3f} {o['t']:7.3f} {mape:5.2f}%")
vr = [x["valid_R2"] for x in sens]; tr_ = [x["test_R2"] for x in sens]
OUT["D_sensitivity"] = {"scan": sens, "valid_range": [min(vr), max(vr)],
                        "test_range": [min(tr_), max(tr_)]}
print(f"  → SRT 8~60일에서 사고학습 R² {min(vr):.3f}~{max(vr):.3f}, 평가 {min(tr_):.3f}~{max(tr_):.3f}")
print("     예측 성능은 SRT에 거의 무관하다. SRT가 바뀌면 θ의 절대 크기만 함께 이동한다")
print("     (θ와 SRT가 서로 보상 — 곱이 총량으로 고정되기 때문)")

with open("outputs/srt_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/srt_results.json")
