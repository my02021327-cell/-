# -*- coding: utf-8 -*-
"""기질별 화학양론 + 공정 지연 기반 메탄생성 예측 모델.

■ 전제 (사용자 확인 사항)
   - 마스터의 date는 **반입일**이다. 소화조 투입일이 아니다.
     반입은 당일 09~18시에 이뤄지고 곧바로 유기물 전처리로 들어간다.
   - 반입량 전량이 아니라 `feed_AB_tpd`에 명시된 톤만 소화조로 투입된다.
   - 소화조 유효 HRT는 이상적 조건에서 12일 수준으로 본다
     (유기산화조에서 이미 VFA 형태로 넘어오므로 메탄발효 단계만 남음).

■ 공정 흐름과 지연 구조
     반입(분뇨·음폐수·음식물)  →  전처리(당일)  →  여액저장조(혼합·완충)
       →  유기산화조(산발효)  →  혐기소화조(메탄발효, HRT 12일)  →  가스

   ① 반입 → 소화조 투입까지의 지연 τ₀ 와 혼합 시상수 τ_mix
        문헌 사전값: 저류조 유효용량 3일 이상 + 산발효조 1~2일 → τ₀ ≈ 4~5일
        (국립환경과학원 「가축분뇨 병합처리 바이오가스화 설계·운전 기술지침」,
         2상 소화 산발효조 HRT 1~2일 운전 사례)
        본 스크립트는 τ₀·τ_mix를 학습구간에서 격자탐색해 문헌 범위와 대조한다.
   ② 소화조 내부: 완전혼합(CSTR) 체류시간 분포  g(τ) = (1/HRT)·exp(−τ/HRT)

■ 기질별 메탄 수율 — 화학양론(Buswell) 기반
     CcHhOoNn + (c − h/4 − o/2 + 3n/4)H₂O
         → (c/2 + h/8 − o/4 − 3n/8)CH₄ + (c/2 − h/8 + o/4 + 3n/8)CO₂ + nNH₃
     B_th [m³CH₄/kg VS] = 22.414 × (c/2 + h/8 − o/4 − 3n/8) / M
   생화학 조성(탄수화물·단백질·지질)별 이론 수율을 각 기질의 조성비로 가중해 산출하고,
   문헌 생분해도(BD)를 곱해 실효 수율 B_i 를 얻는다.

■ 모델식
     CH₄(t) = η · Σ_i B_i · Σ_τ g(τ)·VS_total(t−τ)·φ_i(t−τ−τ₀)
     φ_i : 반입 조성에서 유래한 기질 i 의 VS 기여 분율(저류조 혼합 반영)
     η   : 학습구간에서 추정하는 단일 스케일 계수 (계통 손실·미포집분 흡수)

■ 검증: 학습 2018-2021 / 사고학습 2022 / 평가 2023(최종 1회). 무작위 분할 금지.
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

# ================================================================ ① 화학양론 수율
# 생화학 조성별 이론 메탄수율 — Buswell 식을 각 대표 분자에 적용
COMP = {
    "carb":  {"formula": "C6H10O5",   "c": 6,  "h": 10,  "o": 5, "n": 0},   # 탄수화물(글루칸)
    "prot":  {"formula": "C5H7O2N",   "c": 5,  "h": 7,   "o": 2, "n": 1},   # 단백질
    "lipid": {"formula": "C57H104O6", "c": 57, "h": 104, "o": 6, "n": 0},   # 지질(트리글리세리드)
}


def buswell(c, h, o, n):
    """B_th [m³CH₄/kg VS] = 22.414 × (c/2 + h/8 − o/4 − 3n/8) / 분자량"""
    ch4_mol = c / 2 + h / 8 - o / 4 - 3 * n / 8
    Mw = 12.011 * c + 1.008 * h + 15.999 * o + 14.007 * n
    return 22.414 * ch4_mol / Mw, ch4_mol, Mw


for k, v in COMP.items():
    b, mol, mw = buswell(v["c"], v["h"], v["o"], v["n"])
    v.update({"B_th": round(b, 4), "ch4_mol": round(mol, 3), "Mw": round(mw, 2)})

# 기질별 생화학 조성비(VS 기준)와 생분해도 — 문헌값
#   음식물류: 이론 0.52 / 실측 0.36 Sm³CH₄/kgVS (국내 음식물류 BMP 연구) → BD≈0.69
#   음폐수  : 음식물류 재활용시설 침출수. 가용성 분획이 커 BD 높음
#             (BMP 10일 358 / 28일 478 mL CH₄/gVS)
#   가축분뇨: 장내 소화를 이미 거쳐 난분해성 리그노셀룰로스 비중이 큼 → BD 낮음
SUB = {
    "foodww": {"name": "음폐수", "carb": 0.62, "prot": 0.16, "lipid": 0.22, "BD": 0.82,
               "TSfrac": 0.055, "VSTS": 0.88,
               "src": "음식물류 재활용시설 침출수 BMP(28일 478 mLCH₄/gVS), 가용성 분획 우세"},
    "manure": {"name": "가축분뇨", "carb": 0.52, "prot": 0.24, "lipid": 0.07, "BD": 0.45,
               "TSfrac": 0.045, "VSTS": 0.75,
               "src": "돈분·우분 슬러리. 장내 소화 후 난분해성 섬유질 잔존, BD 0.4~0.5"},
    "food":   {"name": "음식물", "carb": 0.58, "prot": 0.18, "lipid": 0.24, "BD": 0.69,
               "TSfrac": 0.180, "VSTS": 0.90,
               "src": "국내 음식물류 BMP 실측 0.36 / 이론 0.52 Sm³CH₄/kgVS → BD 0.69"},
}
for k, s in SUB.items():
    s["B_th"] = round(sum(COMP[c]["B_th"] * s[c] for c in ["carb", "prot", "lipid"]), 4)
    s["B_eff"] = round(s["B_th"] * s["BD"], 4)
    s["VSfrac"] = round(s["TSfrac"] * s["VSTS"], 4)      # 습중량 대비 VS 분율

print("=== 화학양론 기반 기질별 메탄수율 ===")
for k, v in COMP.items():
    print(f"  {v['formula']:10s} 분자량 {v['Mw']:6.2f}  CH₄ {v['ch4_mol']:5.2f} mol  "
          f"B_th = {v['B_th']:.4f} m³CH₄/kg VS")
for k, s in SUB.items():
    print(f"  {s['name']:6s} 조성 탄수 {s['carb']:.2f}/단백 {s['prot']:.2f}/지질 {s['lipid']:.2f}"
          f" → 이론 {s['B_th']:.3f} × BD {s['BD']:.2f} = 실효 {s['B_eff']:.3f} m³CH₄/kg VS")

# ================================================================ ② 반입 조성 → 투입 조성
intake = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].copy()
intake.columns = ["foodww", "manure", "food"]
intake = intake.clip(lower=0)
tot = intake.sum(axis=1)
intake = intake.where(tot > 0)                       # 반입 0인 날(일요일 등)은 결측 처리
# VS 기여 기준 분율: 반입 톤 × 각 기질의 VS 함량
vs_contrib = pd.DataFrame({k: intake[k] * SUB[k]["VSfrac"] for k in SUB})
phi_raw = vs_contrib.div(vs_contrib.sum(axis=1), axis=0)      # 반입일 기준 VS 기여 분율


def storage_mix(df, tau0, tau_mix, K=40):
    """여액저장조·산발효조를 통과하며 섞이는 과정.
    반입 후 τ₀일 지연 + 시상수 τ_mix의 지수 혼합(CSTR)으로 모사한다.
    반입이 없는 날은 직전 조성이 유지되도록 보간 후 가중합한다."""
    src = df.ffill(limit=7)
    w = np.exp(-np.arange(K + 1) / tau_mix)
    w /= w.sum()
    out = {}
    for c in src.columns:
        v = src[c].values
        acc = np.full(len(v), np.nan)
        base = np.nan_to_num(v)
        o = np.zeros(len(v))
        for i, wv in enumerate(w):
            o[i:] += wv * base[: len(v) - i]
        acc = o
        out[c] = pd.Series(acc, index=src.index).shift(tau0)
    R = pd.DataFrame(out)
    return R.div(R.sum(axis=1), axis=0)


# ================================================================ ③ 소화조 체류시간 분포
def cstr_rtd(s, hrt, K=120):
    """완전혼합 반응조 체류시간 분포 g(τ)=(1/HRT)exp(−τ/HRT)로 가중합."""
    w = np.exp(-np.arange(K + 1) / hrt)
    w /= w.sum()
    v = np.nan_to_num(s.values)
    o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


# 실측 총 VS 부하 (소화조 투입 시점 기준)
feed = M.feed_AB_tpd.ffill(limit=2)
VS_total = (feed * M.acid_VS_pct.interpolate(limit=3) / 100)     # t VS/d
TS_total = (feed * M.acid_TS_pct.interpolate(limit=3) / 100)
CH4_obs = M.CH4_m3d

# VS 함량 정규화 —
#   반입물은 전처리에서 협잡물 제거·농축을 거치므로(반입 210 t/d → 투입 197 t/d)
#   투입물의 고형물 농도가 반입물 문헌값보다 높다. 문헌 VS 함량을 그대로 쓰면
#   총 VS가 과소추정되고 그 오차가 스케일 계수 η로 넘어가, η×BD 가 이론 상한을
#   넘어서는 비물리적 결과가 나온다. 따라서 학습구간 평균 총 VS가 실측과 일치하도록
#   VS 함량 전체를 한 개의 상수로 맞춘다. 기질 간 상대비는 문헌값 그대로 보존된다.
_r_all = intake.div(intake.sum(axis=1), axis=0)
_vs_lit = sum(feed * _r_all[k] * SUB[k]["VSfrac"] for k in SUB)
_d = pd.concat([_vs_lit.rename("lit"), VS_total.rename("obs")], axis=1).loc[TR].dropna()
K_VS = float(_d["obs"].mean() / _d["lit"].mean())
for k in SUB:
    SUB[k]["VSfrac_raw"] = SUB[k]["VSfrac"]
    SUB[k]["VSfrac"] = round(SUB[k]["VSfrac"] * K_VS, 5)
print(f"\nVS 함량 정규화 계수 K_VS = {K_VS:.3f} "
      f"(문헌 기반 총 VS {_d['lit'].mean():.2f} → 실측 {_d['obs'].mean():.2f} t VS/d)")


def predict(tau0, tau_mix, hrt, mode="obsVS"):
    """CH₄(t) = η·Σ_i B_i·Σ_τ g(τ)·VS_i(t−τ)

    mode="tonnage" : 사용자 지정 방식 — 소화조 투입 톤을 반입 비율로 배분한 뒤
                     각 기질의 VS 함량(문헌)을 곱해 기질별 VS 부하를 독립 산출한다.
                       feed_i(t) = feed_AB(t)·r_i(t−τ₀),  VS_i = feed_i·VSfrac_i
    mode="obsVS"   : 관측 제약 방식 — 실측 총 VS 부하를 기질별 VS 기여 분율로 배분.
                     총 VS가 실측과 일치하도록 구속되므로 조건수가 좋다.
    """
    phi = storage_mix(phi_raw, tau0, tau_mix)          # VS 기여 분율(혼합·지연 반영)
    if mode == "tonnage":
        rr = storage_mix(intake.div(intake.sum(axis=1), axis=0), tau0, tau_mix)  # 질량 비율
        pot = sum(SUB[k]["B_eff"] * (feed * rr[k] * SUB[k]["VSfrac"]) * 1000 for k in SUB)
    else:
        pot = sum(SUB[k]["B_eff"] * (VS_total * phi[k]) * 1000 for k in SUB)
    return cstr_rtd(pot.ffill(limit=3), hrt)                     # 소화조 RTD 통과


def design(tau0, tau_mix, hrt, mode="tonnage"):
    """기질별 RTD 통과 잠재 메탄 [m³/d] 3열 설계행렬.
    열 i = B_th,i · VS_i 를 체류시간 분포로 가중합한 값. 계수는 생분해도 BD_i.
    """
    rr = storage_mix(intake.div(intake.sum(axis=1), axis=0), tau0, tau_mix)
    phi = storage_mix(phi_raw, tau0, tau_mix)
    cols = {}
    for k in SUB:
        if mode == "tonnage":
            vs_k = feed * rr[k] * SUB[k]["VSfrac"]          # t VS/d
        else:
            vs_k = VS_total * phi[k]
        cols[k] = cstr_rtd((SUB[k]["B_th"] * vs_k * 1000).ffill(limit=3), hrt)
    return pd.DataFrame(cols)


def fit_bd(X, obs, sl, with_base=True):
    """생분해도 BD_i 를 0≤BD≤1 제약 하에서 추정한다.
    자유 스케일 계수 η를 쓰면 η·BD 가 이론 수율을 넘어서는 비물리적 해가 나오므로
    (실측: 음폐수 함의 BD 1.20), 계수 자체를 물리적 상한에 묶는다.
    with_base=True 이면 난분해성 분획의 정상상태 기여를 비음수 절편으로 함께 추정한다.
    """
    d = pd.concat([X, obs.rename("y")], axis=1).loc[sl].dropna()
    if len(d) < 30:
        return None
    A = d[list(SUB)].values
    lo, hi = [0.0] * 3, [1.0] * 3
    if with_base:
        A = np.c_[A, np.ones(len(d))]
        lo, hi = lo + [0.0], hi + [np.inf]
    r = lsq_linear(A, d["y"].values, bounds=(np.array(lo), np.array(hi)), max_iter=500)
    bd = {k: float(v) for k, v in zip(SUB, r.x[:3])}
    return {"bd": bd, "base": float(r.x[3]) if with_base else 0.0}


def apply_bd(X, par):
    return sum(par["bd"][k] * X[k] for k in SUB) + par["base"]


def score_bd(X, obs, sl, par):
    d = pd.concat([apply_bd(X, par).rename("p"), obs.rename("y")], axis=1).loc[sl].dropna()
    t, yh = d["y"].values, d["p"].values
    ss = ((t - yh) ** 2).sum()
    return {"n": int(len(d)),
            "R2": round(float(1 - ss / ((t - t.mean()) ** 2).sum()), 3),
            "RMSE": round(float(np.sqrt(ss / len(d)))),
            "MAPE": round(float(np.abs((t - yh) / t).mean() * 100), 2)}


def fit_eta(pred, obs, sl):
    d = pd.concat([pred.rename("p"), obs.rename("y")], axis=1).loc[sl].dropna()
    if len(d) < 30:
        return np.nan, np.nan
    A = np.c_[np.ones(len(d)), d["p"].values]
    b = np.linalg.lstsq(A, d["y"].values, rcond=None)[0]
    return float(b[1]), float(b[0])


def score(pred, obs, sl, eta, c0):
    d = pd.concat([pred.rename("p"), obs.rename("y")], axis=1).loc[sl].dropna()
    yh = c0 + eta * d["p"].values
    t = d["y"].values
    ss = ((t - yh) ** 2).sum()
    return {"n": int(len(d)),
            "R2": round(float(1 - ss / ((t - t.mean()) ** 2).sum()), 3),
            "RMSE": round(float(np.sqrt(ss / len(d)))),
            "MAPE": round(float(np.abs((t - yh) / t).mean() * 100), 2)}


# ================================================================ ④ 지연 파라미터 격자탐색
FOLDS = [(slice("2018-01-01", "2019-12-31"), slice("2020-01-01", "2020-12-31")),
         (slice("2018-01-01", "2020-12-31"), slice("2021-01-01", "2021-12-31")),
         (slice("2018-01-01", "2021-12-31"), slice("2022-01-01", "2022-12-31"))]
HRT_USER = 12.0
MODE = "tonnage"          # 사용자 지정 방식을 주 모델로 둔다


def cv_of(tau0, tau_mix, hrt, mode=MODE):
    X = design(tau0, tau_mix, hrt, mode)
    cvs = []
    for a, b in FOLDS:
        par = fit_bd(X, CH4_obs, a)
        if par:
            cvs.append(score_bd(X, CH4_obs, b, par)["R2"])
    return (float(np.mean(cvs)) if cvs else -9e9), X


# (가) 사용자 지정 HRT 12일 고정 — τ₀·τ_mix 탐색
#   τ_mix 하한을 3일로 둔다. 설계지침상 저류조 유효용량이 3일 이상이므로 그보다 빠른
#   혼합은 물리적으로 불가능하다. 제약 없이 탐색하면 τ_mix=0.5일이 근소하게 선택되는데,
#   그 경우 소화조 도달 조성이 하루 만에 1%↔97%로 진동해 실제 저류조 거동과 모순된다.
MIX_GRID = [3, 5, 7, 10, 14]
grid = []
for tau0 in range(0, 21):
    for tau_mix in MIX_GRID:
        cv, _ = cv_of(tau0, tau_mix, HRT_USER)
        grid.append({"tau0": tau0, "tau_mix": tau_mix, "cv": round(cv, 4)})
grid.sort(key=lambda x: -x["cv"])
best = grid[0]
print(f"\n=== (가) 지연 격자탐색 — HRT {HRT_USER:.0f}일 고정, 학습구간 롤링 3폴드 ===")
for g in grid[:6]:
    print(f"  τ₀={g['tau0']:2d}일  τ_mix={g['tau_mix']:4.1f}일  CV R² {g['cv']:+.4f}")
print(f"  → 최적 τ₀={best['tau0']}일, τ_mix={best['tau_mix']}일")

# (나) HRT까지 자유 탐색 — 데이터가 선호하는 유효 체류시간
free = []
for hrt in [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40.5]:
    bb = max(((cv_of(t0, tm, hrt)[0], t0, tm) for t0 in range(0, 21)
              for tm in MIX_GRID), key=lambda x: x[0])
    free.append({"hrt": hrt, "cv": round(bb[0], 4), "tau0": bb[1], "tau_mix": bb[2]})
print("\n=== (나) 소화조 유효 HRT 자유 탐색 (각 HRT에서 τ₀·τ_mix 재최적화) ===")
for h in free:
    mark = "  ← 사용자 지정" if h["hrt"] == HRT_USER else ""
    print(f"  HRT {h['hrt']:5.1f}일  최적 τ₀={h['tau0']:2d}  τ_mix={h['tau_mix']:4.1f}  "
          f"CV R² {h['cv']:+.4f}{mark}")
best_free = max(free, key=lambda x: x["cv"])
hrt_scan = free

# (다) 문헌 사전값 고정 시나리오 — τ₀ 4일(저류 3일 + 산발효 1일)
LIT = {"tau0": 4, "tau_mix": 3.0, "hrt": HRT_USER}
LIT["cv"] = round(cv_of(LIT["tau0"], LIT["tau_mix"], LIT["hrt"])[0], 4)
print(f"\n=== (다) 문헌 사전값 고정 시나리오 ===")
print(f"  τ₀=4일(저류 3일+산발효 1일), τ_mix=3일, HRT 12일 → CV R² {LIT['cv']:+.4f}")

# ================================================================ ⑤ 최종 모델
X = design(best["tau0"], best["tau_mix"], HRT_USER, MODE)
PAR = fit_bd(X, CH4_obs, TR)
P = apply_bd(X, PAR)
FIN = {"train": score_bd(X, CH4_obs, TR, PAR),
       "valid": score_bd(X, CH4_obs, VA, PAR),
       "test": score_bd(X, CH4_obs, TE, PAR)}
print(f"\n=== 최종 모델 — 생분해도 제약 적합 (난분해성 기저 {PAR['base']:.0f} ㎥/d) ===")
for k, v in FIN.items():
    print(f"  {k:6s} n={v['n']:4d}  R²={v['R2']:+.3f}  RMSE={v['RMSE']:5d}  MAPE={v['MAPE']:.2f}%")

# --- 식별가능성 진단 -------------------------------------------------------
_dX = pd.concat([X, CH4_obs.rename("y")], axis=1).loc[TR].dropna()
_A = _dX[list(SUB)].values
_An = _A / _A.std(axis=0)
_cond = float(np.linalg.cond(_An))
_corr = pd.DataFrame(_A, columns=list(SUB)).corr()
IDENT = {"cond": round(_cond, 1),
         "corr": {f"{a}-{b}": round(float(_corr.loc[a, b]), 3)
                  for i, a in enumerate(SUB) for b in list(SUB)[i + 1:]},
         "ratio_cv": {k: round(float(_r_all[k].std() / _r_all[k].mean() * 100), 1) for k in SUB},
         "at_bound": [k for k in SUB if PAR["bd"][k] >= 0.999 or PAR["bd"][k] <= 0.001]}
print("\n=== 식별가능성 진단 ===")
print(f"  설계행렬 조건수 {IDENT['cond']:.0f}  (열 간 상관 " +
      ", ".join(f"{k} {v:+.3f}" for k, v in IDENT["corr"].items()) + ")")
print("  반입 비율 변동계수: " + ", ".join(f"{SUB[k]['name']} {v}%" for k, v in IDENT["ratio_cv"].items()))
print(f"  경계에 붙은 계수: {IDENT['at_bound']} → 기질별 BD는 이 데이터로 분리 불가")

print("\n=== 기질별 생분해도 — 문헌 가정 vs 데이터 추정 ===")
for k, sb in SUB.items():
    bd_f = PAR["bd"][k]
    sb["BD_fit"] = round(bd_f, 3)
    sb["B_fit"] = round(sb["B_th"] * bd_f, 4)
    print(f"  {sb['name']:6s} 이론수율 {sb['B_th']:.3f}  |  문헌 BD {sb['BD']:.2f} → 실효 {sb['B_eff']:.3f}"
          f"  |  추정 BD {bd_f:.2f} → 실효 {sb['B_fit']:.3f} m³CH₄/kg VS")

# 주 모델(권고): 문헌 BD 고정 — 기저항만 추정. 식별 불가한 BD를 데이터로 밀지 않는다.
PAR_LIT = {"bd": {k: SUB[k]["BD"] for k in SUB}, "base": 0.0}
_dl = pd.concat([apply_bd(X, PAR_LIT).rename("p"), CH4_obs.rename("y")], axis=1).loc[TR].dropna()
PAR_LIT["base"] = float(max(0.0, (_dl["y"] - _dl["p"]).mean()))
FIN_LIT = {k: score_bd(X, CH4_obs, sl, PAR_LIT) for k, sl in
           [("train", TR), ("valid", VA), ("test", TE)]}
print(f"\n=== 주 모델 — 문헌 BD 고정 (기저항 {PAR_LIT['base']:.0f} ㎥/d) ===")
for k, v in FIN_LIT.items():
    print(f"  {k:6s} n={v['n']:4d}  R²={v['R2']:+.3f}  RMSE={v['RMSE']:5d}  MAPE={v['MAPE']:.2f}%")

# 대조군 A: 기질 구분 없음 (총 투입 톤만, 같은 지연·RTD)
_rr = storage_mix(intake.div(intake.sum(axis=1), axis=0), best["tau0"], best["tau_mix"])
_Bmix = sum(SUB[k]["B_th"] * SUB[k]["VSfrac"] for k in SUB) / 3
Xf = pd.DataFrame({"flat": cstr_rtd((feed * _Bmix * 1000).ffill(limit=3), HRT_USER)})
_d = pd.concat([Xf, CH4_obs.rename("y")], axis=1).loc[TR].dropna()
_b = np.linalg.lstsq(np.c_[np.ones(len(_d)), _d["flat"].values], _d["y"].values, rcond=None)[0]
def _sc(sl):
    dd = pd.concat([Xf, CH4_obs.rename("y")], axis=1).loc[sl].dropna()
    yh = _b[0] + _b[1] * dd["flat"].values; t = dd["y"].values
    ss = ((t - yh) ** 2).sum()
    return {"n": int(len(dd)), "R2": round(float(1 - ss / ((t - t.mean()) ** 2).sum()), 3),
            "RMSE": round(float(np.sqrt(ss / len(dd)))),
            "MAPE": round(float(np.abs((t - yh) / t).mean() * 100), 2)}
CTRL_FLAT = {k: _sc(sl) for k, sl in [("train", TR), ("valid", VA), ("test", TE)]}

# 대조군 B: 기질 구분 + 지연/RTD 없음
X0 = design(0, 0.5, 0.5, MODE)
PAR0 = fit_bd(X0, CH4_obs, TR)
CTRL_ND = {k: score_bd(X0, CH4_obs, sl, PAR0) for k, sl in
           [("train", TR), ("valid", VA), ("test", TE)]}

# 변형: 실측 총 VS 배분 방식
Xo = design(best["tau0"], best["tau_mix"], HRT_USER, "obsVS")
PARo = fit_bd(Xo, CH4_obs, TR)
ALT_OBS = {k: score_bd(Xo, CH4_obs, sl, PARo) for k, sl in
           [("train", TR), ("valid", VA), ("test", TE)]}

print(f"\n=== 대조군 ===")
print(f"  기질 구분 없음(투입 톤만)  평가 R²={CTRL_FLAT['test']['R2']:+.3f} MAPE={CTRL_FLAT['test']['MAPE']}%")
print(f"  기질 구분 + 지연 없음      평가 R²={CTRL_ND['test']['R2']:+.3f} MAPE={CTRL_ND['test']['MAPE']}%")
print(f"  변형: 실측 총 VS 배분      평가 R²={ALT_OBS['test']['R2']:+.3f} MAPE={ALT_OBS['test']['MAPE']}%")
eta, c0 = 1.0, PAR["base"]

# ================================================================ ⑥ 총 소요시간
phi_best = storage_mix(intake.div(intake.sum(axis=1), axis=0), best["tau0"], best["tau_mix"])
tau_grid = np.arange(0, 121)
g = np.exp(-tau_grid / HRT_USER); g /= g.sum()
mean_dig = float((tau_grid * g).sum())
cum = np.cumsum(g)
t50 = int(np.argmax(cum >= 0.5)); t90 = int(np.argmax(cum >= 0.9))
mix_mean = best["tau_mix"]
TOTAL = {
    "tau0_delay_d": best["tau0"],
    "tau_mix_d": best["tau_mix"],
    "hrt_d": HRT_USER,
    "mean_digester_d": round(mean_dig, 1),
    "mean_total_d": round(best["tau0"] + mix_mean + mean_dig, 1),
    "t50_total_d": round(best["tau0"] + mix_mean + t50, 1),
    "t90_total_d": round(best["tau0"] + mix_mean + t90, 1),
    "lit_prior": "저류조 ≥3일 + 산발효조 1~2일 → τ₀ 4~5일 (국립환경과학원 병합처리 기술지침)",
}
print(f"\n=== 반입 → 메탄생성 총 소요시간 ===")
print(f"  전처리·저류·산발효 지연 τ₀ = {best['tau0']}일 (혼합 시상수 {best['tau_mix']}일)")
print(f"  소화조 평균 체류 {mean_dig:.1f}일 (중앙 {t50}일, 90% {t90}일)")
print(f"  → 평균 총 소요 {TOTAL['mean_total_d']}일 / 50% {TOTAL['t50_total_d']}일 / 90% {TOTAL['t90_total_d']}일")
print(f"  문헌 사전값: {TOTAL['lit_prior']}")

# ================================================================ ⑦ 기질별 기여도
contrib = {}
for k in SUB:
    pot_k = PAR_LIT["bd"][k] * X[k]
    contrib[k] = pot_k
CB = pd.DataFrame(contrib)
share = (CB.div(CB.sum(axis=1), axis=0) * 100)
by_year = {}
for yr, gg in share.groupby(share.index.year):
    by_year[int(yr)] = {k: round(float(gg[k].mean()), 1) for k in SUB}
intake_share = {}
for yr, gg in intake.groupby(intake.index.year):
    s = gg.sum()
    intake_share[int(yr)] = {k: round(float(s[k] / s.sum() * 100), 1) for k in SUB}
print("\n=== 연도별 반입 비율(%) vs 메탄 기여율(%) ===")
for yr in by_year:
    a = intake_share.get(yr, {})
    print(f"  {yr}  반입 " + " ".join(f"{SUB[k]['name']} {a.get(k,0):4.1f}%" for k in SUB) +
          "   |  메탄기여 " + " ".join(f"{SUB[k]['name']} {by_year[yr][k]:4.1f}%" for k in SUB))

idx = M.index
def ser(s, r=0):
    s = s.reindex(idx)
    return [None if pd.isna(v) else round(float(v), r) for v in s.values]


OUT = {
    "split": {"train": "2018-01-01~2021-12-31", "valid": "2022-01-01~2022-12-31",
              "test": "2023-01-01~2023-09-17"},
    "comp": COMP, "sub": SUB, "K_VS": round(K_VS, 4),
    "grid": grid[:12], "best": best, "hrt_scan": hrt_scan,
    "best_free": best_free, "lit_scenario": LIT, "mode": MODE, "alt_obsVS": ALT_OBS,
    "bd_fit": {k: round(v, 4) for k, v in PAR["bd"].items()},
    "base_m3d": round(PAR["base"], 1),
    "ident": IDENT, "final_lit": FIN_LIT,
    "bd_lit": {k: SUB[k]["BD"] for k in SUB}, "base_lit": round(PAR_LIT["base"], 1),
    "final": FIN, "ctrl_flat": CTRL_FLAT, "ctrl_nodelay": CTRL_ND,
    "total_time": TOTAL,
    "share_by_year": by_year, "intake_share_by_year": intake_share,
    "series": {
        "obs": ser(CH4_obs), "pred": ser(apply_bd(X, PAR_LIT)),
        "pred_fitbd": ser(P),
        "sub_foodww": ser(CB["foodww"]), "sub_manure": ser(CB["manure"]),
        "sub_food": ser(CB["food"]),
        "phi_foodww": ser(phi_best["foodww"] * 100, 1),
        "phi_manure": ser(phi_best["manure"] * 100, 1),
        "phi_food": ser(phi_best["food"] * 100, 1),
    },
}
with open("outputs/substrate_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/substrate_results.json")
