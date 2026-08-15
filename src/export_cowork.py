# -*- coding: utf-8 -*-
"""Cowork 활용용 데이터셋 내보내기 — 소화조 내부 일별 가중치 + SRT 추정 입력.

산출물 (outputs/cowork/)
  영천BGP_소화조_가중치_SRT_데이터셋.xlsx   … 7개 시트 통합본
  01_일별_소화조상태.csv                    … 일별 재고·가중치·예측 (주력)
  02_잔존율커널.csv                         … τ일 전 투입분의 잔존율·메탄발생 가중치
  03_SRT추정_입력.csv                       … SRT 재추정에 필요한 원자료
  04_SRT_민감도.csv                         … SRT를 흔들었을 때의 성능·θ 변화
  05_물질수지_요약.csv                      … 배출 유량 역산과 SRT 허용 구간

■ 모델 정의 (src/retention_srt.py 와 동일)
    소화조 유입 조성  r_i(t) = 전단커널 ⊛ 반입조성   (τ₀ 지연 + τ_mix 혼합, Σw=1)
    기질 i 투입 톤     L_i(t) = feed(t) · r_i(t)
    잔존 재고          N_i(t) = Σ_τ exp(−τ/SRT) · L_i(t−τ)            [모드 A, 톤]
    메탄발생 가중재고  G_i(t) = Σ_τ k_i·exp(−(1/SRT+k_i)τ) · L_i(t−τ) [모드 B, 톤/d]
    예측 메탄          CH₄(t) = Σ_i θ_i · G_i(t)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/cowork")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 모델 상수
SRT = 12.0                                                # 가정값 (§2.8.4). 추정치 아님
TAU0, TMIX = 0, 3.0                                       # 전단 지연·혼합 (롤링 검증 선택)
KS = {"foodww": 0.40, "manure": 0.08, "food": 0.30}       # 1차 분해속도 [1/d]
NM = {"foodww": "음폐수", "manure": "가축분뇨", "food": "음식물"}
SPLIT = [("학습", "2018-01-01", "2021-12-31"),
         ("사고학습", "2022-01-01", "2022-12-31"),
         ("평가", "2023-01-01", "2023-09-17")]

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]

feed = M.feed_AB_tpd.ffill(limit=2)
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = list(KS)
I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0)
y = M.biogas_AB_m3d * (M.CH4_pct.ffill(limit=3) / 100)


def mix(df, t0, tm, K=40):
    """반입 조성 → 전처리·여액저장조·산생성조 통과 후 소화조 유입 조성."""
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


def conv(s, w):
    """가중치 벡터 w(τ)를 시계열 s에 적용한 누적 합성곱."""
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, wv in enumerate(w):
        o[i:] += wv * v[: len(v) - i]
    return pd.Series(o, index=s.index)


K = 300
tau = np.arange(K + 1)
W_A = np.exp(-tau / SRT)                                   # 잔존율
W_B = {k: KS[k] * np.exp(-(1.0 / SRT + KS[k]) * tau) for k in KS}   # 메탄발생 가중치

Rm = mix(R, TAU0, TMIX)
TON = {k: (feed * Rm[k]).ffill(limit=3) for k in KS}       # 기질별 소화조 투입 톤
INV = {k: conv(TON[k], W_A) for k in KS}                   # 잔존 재고 [t]
GEN = {k: conv(TON[k], W_B[k]) for k in KS}                # 메탄발생 가중재고 [t/d]

# θ — 학습구간 OLS(절편 없음), src/retention_srt.py 와 동일 절차
X = pd.DataFrame(GEN)
D = pd.concat([X, y.rename("y")], axis=1).dropna()
tr = D.loc["2018-01-01":"2021-12-31"]
theta_v, *_ = np.linalg.lstsq(tr[list(KS)].values, tr["y"].values, rcond=None)
TH = {k: float(v) for k, v in zip(KS, theta_v)}

# 재고 가중 평균 체류일 — 지금 소화조 안에 있는 물질이 평균 며칠 전에 들어왔는가
tot_ton = sum(TON[k] for k in KS)
inv_tot = conv(tot_ton, W_A)
age_num = conv(tot_ton, W_A * tau)
mean_age = (age_num / inv_tot).replace([np.inf, -np.inf], np.nan)

# ================================================================ 1. 일별 소화조 상태
d1 = pd.DataFrame(index=M.index)
d1["구분"] = ""
for nm, a, b in SPLIT:
    d1.loc[a:b, "구분"] = nm
d1["소화조_투입량_tpd"] = feed.round(2)
d1["반입_총량_tpd"] = M.intake_total_tpd.round(2)
for k in KS:
    d1[f"반입_{NM[k]}_tpd"] = I[k].round(2)
for k in KS:
    d1[f"반입비율_{NM[k]}"] = R[k].round(4)
for k in KS:
    d1[f"유입비율_{NM[k]}"] = Rm[k].round(4)               # 전단 통과 후 소화조 유입 조성
for k in KS:
    d1[f"투입톤_{NM[k]}_t"] = TON[k].round(3)
for k in KS:
    d1[f"잔존재고_{NM[k]}_t"] = INV[k].round(2)
d1["잔존재고_합계_t"] = inv_tot.round(2)
d1["재고_평균체류일"] = mean_age.round(2)
for k in KS:
    d1[f"발생가중재고_{NM[k]}_t"] = GEN[k].round(3)
for k in KS:
    d1[f"예측CH4_{NM[k]}_m3d"] = (TH[k] * GEN[k]).round(1)
pred = sum(TH[k] * GEN[k] for k in KS)
d1["예측CH4_합계_m3d"] = pred.round(1)
d1["실측CH4_m3d"] = y.round(1)
d1["잔차_m3d"] = (y - pred).round(1)
d1["잔차율_pct"] = ((y - pred) / y * 100).round(2)
# 소화조 내부 상태 (진단용 동반 컬럼)
d1["소화액_TS_pct"] = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1).round(3)
d1["소화액_VS_pct"] = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1).round(3)
d1["소화액_pH"] = M[["dig_pH_A", "dig_pH_B"]].mean(axis=1).round(2)
d1["소화조_온도_C"] = M[["dig_T_A_C", "dig_T_B_C"]].mean(axis=1).round(2)
d1["VFA_A_mgL"] = M.VFA_A_mgL
d1["ALK_A_mgL"] = M.ALK_A_mgL
d1["VFA_ALK_A"] = M.VFA_ALK_A.round(4)
d1["NH3N_A_mgL"] = M.NH3N_A_mgL
d1["FAN_A_mgL"] = M.FAN_A_mgL.round(1)
d1["OLR_kgVS_m3d"] = M.OLR_kgVS_m3d.round(3)
d1 = d1.reset_index().rename(columns={"date": "날짜"})

# ================================================================ 2. 잔존율 커널
KK = 90
t2 = np.arange(KK + 1)
d2 = pd.DataFrame({"경과일_tau": t2})
for s_ in [8, 12, 20, 41]:
    d2[f"잔존율_SRT{s_}일"] = np.round(np.exp(-t2 / s_), 5)
d2["잔존율_채택_SRT12일"] = np.round(np.exp(-t2 / SRT), 5)
d2["누적잔존_비중_SRT12일"] = np.round(np.cumsum(np.exp(-t2 / SRT)) / np.exp(-t2 / SRT).sum(), 5)
for k in KS:
    w = KS[k] * np.exp(-(1.0 / SRT + KS[k]) * t2)
    d2[f"메탄발생가중_{NM[k]}"] = np.round(w, 6)
    d2[f"누적메탄_비중_{NM[k]}"] = np.round(np.cumsum(w) / (KS[k] * np.exp(-(1.0 / SRT + KS[k]) * tau)).sum(), 5)

# ================================================================ 3. SRT 추정 입력
d3 = pd.DataFrame(index=M.index)
d3["소화조_투입량_tpd"] = feed.round(2)
d3["산발효조_TS_pct"] = M.acid_TS_pct
d3["유입_TS부하_tpd"] = (feed * M.acid_TS_pct / 100).round(3)
d3["소화액_TS_A_pct"] = M.dig_TS_A_pct
d3["소화액_TS_B_pct"] = M.dig_TS_B_pct
d3["소화액_TS_평균_pct"] = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1).round(3)
d3["소화액_VS_평균_pct"] = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1).round(3)
d3["TS제거율_pct"] = (100 * (1 - d3["소화액_TS_평균_pct"] / d3["산발효조_TS_pct"])).round(2)
d3["잔여고형물_tpd"] = (d3["유입_TS부하_tpd"] * (1 - d3["TS제거율_pct"] / 100)).round(3)
d3["역산_배출유량_tpd"] = (d3["잔여고형물_tpd"] / (d3["소화액_TS_평균_pct"] / 100)).round(1)
d3["탈수기_처리량_tpd"] = M.dewater_tpd
d3["추적자_알칼리도_A_mgL"] = M.ALK_A_mgL             # 비분해성 (k_d≈0)
d3["추적자_NH3N_A_mgL"] = M.NH3N_A_mgL               # 비분해성 (k_d≈0)
d3["분뇨_반입비율"] = (M.intake_manure_tpd / M.intake_total_tpd).clip(0, 1).round(4)
d3["분뇨_투입상당톤_t"] = TON["manure"].round(3)
d3["실측CH4_m3d"] = y.round(1)
d3 = d3.reset_index().rename(columns={"date": "날짜"})

# ================================================================ 4~5. 요약표
SR = json.load(open("outputs/srt_results.json", encoding="utf-8"))
d4 = pd.DataFrame([{"가정_SRT_일": x["srt"],
                    "θ_음폐수_m3CH4_per_t": x["theta"]["foodww"],
                    "θ_가축분뇨_m3CH4_per_t": x["theta"]["manure"],
                    "θ_음식물_m3CH4_per_t": x["theta"]["food"],
                    "사고학습_R2": x["valid_R2"], "평가_R2": x["test_R2"], "평가_MAPE_pct": x["MAPE"]}
                   for x in SR["D_sensitivity"]["scan"]])

C = SR["C_balance"]
A_ = SR["A_digTS"]
d5 = pd.DataFrame([
    ["유입 TS", C["TS_in_tpd"], "t/d", "기질별 투입 톤 × 산발효조 TS 함량"],
    ["TS 제거율", C["TS_removal_pct"], "%", "유입 TS 대비 소화 중 분해·가스화"],
    ["잔여 고형물", C["TS_out_tpd"], "t/d", "유입 TS × (1 − 제거율)"],
    ["소화조 배출 유량 (역산)", C["Q_out_tpd"], "t/d", "잔여 고형물 ÷ 소화액 TS 농도"],
    ["소화조 투입량 (비교)", C["feed_tpd"], "t/d", f"배출/유입 = {C['ratio_out_in']} — 수지가 닫힘"],
    ["탈수기 처리량 (비교)", C["dewater_tpd"], "t/d",
     f"배출 유량의 {C['dewater_over_Qout']}배 — 반송·희석수 포함, SRT 분모로 쓰면 안 됨"],
    ["SRT (유효 용적 8,000㎥)", C["srt_by_V"]["8000"], "일", "설계 용적 100% 가정"],
    ["SRT (유효 용적 4,800㎥)", C["srt_by_V"]["4800(60%)"], "일", "설계 용적 60% 가정"],
    ["SRT (유효 용적 2,400㎥)", C["srt_by_V"]["2400(30%)"], "일", "설계 용적 30% 가정"],
    ["소화액 TS 변동계수", A_["cv_pct"], "%", "이 값이 작아 동적 응답으로 SRT를 읽을 수 없음"],
    ["A. 소화액 TS 응답 R²", A_["best"]["R2"], "-", f"최적 시상수 {A_['best']['tau']}일 (격자 상한) — 신호 없음"],
    ["B. 알칼리도 추적자 R²", SR["B_tracer"]["알칼리도"]["best"]["R2"], "-", "격자 상한 — 신호 없음"],
    ["B′. NH3-N 추적자 R²", SR["B_tracer"]["NH3-N"]["best"]["R2"], "-", "격자 상한 — 신호 없음"],
], columns=["항목", "값", "단위", "비고"])

# ================================================================ 6. 모델 계수
sc = {}
for nm, a, b in SPLIT:
    dd = D.loc[a:b]
    p = dd[list(KS)].values @ theta_v
    t_ = dd["y"].values
    sc[nm] = (int(len(dd)),
              round(float(1 - ((t_ - p) ** 2).sum() / ((t_ - t_.mean()) ** 2).sum()), 3),
              round(float(np.abs((t_ - p) / t_).mean() * 100), 2))
d6 = pd.DataFrame([
    ["SRT", SRT, "일", "가정값 — 물질수지 허용 구간 12.2~40.8일의 하단. 추정치가 아님(§2.8.4)"],
    ["전단 지연 τ₀", TAU0, "일", "반입 → 소화조 유입 순수 이송 지연"],
    ["전단 혼합 τ_mix", TMIX, "일", "여액저장조·산생성조 완전혼합 시상수 (Σw=1, 총량 보존)"],
    ["k 음폐수", KS["foodww"], "1/d", "1차 분해속도"],
    ["k 가축분뇨", KS["manure"], "1/d", "1차 분해속도 — 가장 느려 SRT에 민감"],
    ["k 음식물", KS["food"], "1/d", "1차 분해속도"],
    ["θ 음폐수", round(TH["foodww"], 3), "㎥CH₄/t", "학습구간 OLS(절편 없음). θ = BD × COD농도 × 0.35"],
    ["θ 가축분뇨", round(TH["manure"], 3), "㎥CH₄/t", "동일"],
    ["θ 음식물", round(TH["food"], 3), "㎥CH₄/t", "동일 — 95% CI가 0을 포함해 불안정"],
] + [[f"{nm} R²", sc[nm][1], "-", f"n={sc[nm][0]}, MAPE {sc[nm][2]}%"] for nm, _, _ in SPLIT],
    columns=["항목", "값", "단위", "비고"])

# ================================================================ 0. 읽어주세요
README = [
    ["■ 이 파일은", "영천 통합바이오가스화시설 소화조의 '일별 내부 재고 가중치'와 'SRT 추정 입력자료'를 담고 있다."],
    ["", ""],
    ["■ 핵심 개념 — 잔존율 가중 적분", ""],
    ["", "오늘 소화조 안에 있는 물질은 오늘 들어온 것만이 아니다."],
    ["", "τ일 전 투입분은 exp(−τ/SRT)만큼 남아 있다. 오늘분 100%, 1일 전 92.0%, 7일 전 55.8%, 14일 전 31.1% (SRT 12일)."],
    ["", "메탄은 남아 있는 물질이 분해되며 나오므로, 가중치에 분해속도까지 곱한 h(τ)=k·exp(−(1/SRT+k)τ)를 쓴다."],
    ["", "시트 02가 이 가중치 표이고, 시트 01은 그 가중치를 실제 일별 투입량에 적용한 결과다."],
    ["", ""],
    ["■ 시트 안내", ""],
    ["01_일별_소화조상태", "일별 투입 조성·기질별 잔존 재고·메탄발생 가중재고·예측/실측 메탄·내부 이화학. 주력 시트."],
    ["02_잔존율커널", "τ일 전 투입분의 잔존율과 메탄발생 가중치. SRT 8/12/20/41일 비교 포함."],
    ["03_SRT추정_입력", "SRT를 다시 추정하려 할 때 필요한 원자료 일별 계열."],
    ["04_SRT_민감도", "SRT를 8~60일로 흔들었을 때 θ와 예측 성능이 어떻게 변하는가."],
    ["05_물질수지_요약", "고형물 물질수지로 역산한 배출 유량과 SRT 허용 구간."],
    ["06_모델계수", "모델에 실제로 들어간 상수와 구간별 성능."],
    ["", ""],
    ["■ 반드시 알아둘 것 — 반입일 ≠ 투입일", ""],
    ["", "원본 날짜는 유기물이 반입된 날이지 소화조에 투입된 날이 아니다."],
    ["", "반입물은 전처리 → 여액저장조 → 산생성조를 거쳐 소화조에 도달한다."],
    ["", "따라서 '반입비율_*' 과 '유입비율_*' 은 다른 값이다. 모델이 쓰는 것은 유입비율 쪽이다."],
    ["", "변환은 순수 지연 τ₀와 혼합 시상수 τ_mix의 정규화 지수커널(Σw=1)이며 총량은 보존된다."],
    ["", ""],
    ["■ SRT 12일은 추정치가 아니라 가정이다", ""],
    ["", "운전 데이터만으로 SRT를 추정하는 세 방향(소화액 TS 동적 응답 / 비분해성 추적자 / 고형물 물질수지)을"],
    ["", "모두 시도했으나 전부 실패했다. 소화액 TS 변동계수가 8.2%뿐이라 응답 진폭이 없기 때문이다."],
    ["", "물질수지가 좁혀준 구간은 12.2~40.8일이고, 그 안에서 결정되지 않는다."],
    ["", "다만 SRT를 8~60일로 흔들어도 평가 R² 변화가 0.013뿐이다(시트 04)."],
    ["", "θ와 SRT가 서로 보상하기 때문이며, SRT 불확실성은 예측이 아니라 θ→생분해도 해석에서만 문제가 된다."],
    ["", "SRT를 실제로 확정하려면 추적자 시험(리튬·플루오레세인 펄스 주입 후 배출 농도 추적)이 필요하다."],
    ["", ""],
    ["■ 데이터 품질 주의", ""],
    ["", "· 2023년 자료는 09-17에서 끝난다."],
    ["", "· dig_T_A/B 에 센서 고장 구간이 있다 (원본 flag_TA_sensor_fault / flag_TB_sensor_fault)."],
    ["", "· 2023년 ALK_B 는 VFA_B 값으로 오염돼 있어 A계열만 사용했다."],
    ["", "· VS/TS 비가 2020년에 0.80 → 0.70 으로 계단 변화한다 (측정법 변경 추정)."],
    ["", "· θ 음식물은 95% 신뢰구간이 0을 포함한다. 투입 비중이 3% 미만이라 개별 계수가 불안정하다."],
]
d0 = pd.DataFrame(README, columns=["구분", "내용"])

# ================================================================ 저장
SHEETS = [("00_읽어주세요", d0), ("01_일별_소화조상태", d1), ("02_잔존율커널", d2),
          ("03_SRT추정_입력", d3), ("04_SRT_민감도", d4), ("05_물질수지_요약", d5),
          ("06_모델계수", d6)]

xlsx = OUT / "영천BGP_소화조_가중치_SRT_데이터셋.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    for nm, df in SHEETS:
        df.to_excel(xw, sheet_name=nm, index=False)
    from openpyxl.styles import Alignment, Font, PatternFill
    hdr = PatternFill("solid", fgColor="1F3A5F")
    for nm, df in SHEETS:
        ws = xw.sheets[nm]
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF", size=10)
            c.fill = hdr
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.freeze_panes = "B2" if nm.startswith(("01", "03")) else "A2"
        for i, col in enumerate(df.columns, 1):
            wide = max(len(str(col)) * 1.6, 10)
            if nm == "00_읽어주세요":
                wide = 26 if i == 1 else 110
            elif nm in ("05_물질수지_요약", "06_모델계수"):
                wide = [30, 12, 8, 76][i - 1]
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(wide, 110)
        if nm == "00_읽어주세요":
            for row in ws.iter_rows(min_row=2):
                row[0].font = Font(bold=True, size=10)
                row[1].alignment = Alignment(wrap_text=True, vertical="top")

for nm, df in SHEETS[1:]:
    df.to_csv(OUT / f"{nm}.csv", index=False, encoding="utf-8-sig")

print(f"저장: {xlsx}")
for nm, df in SHEETS:
    print(f"  {nm:22s} {len(df):5d}행 × {len(df.columns):2d}열")
print(f"\nθ = " + ", ".join(f"{NM[k]} {TH[k]:.2f}" for k in KS) + " ㎥CH₄/t")
print("구간 성능: " + " / ".join(f"{nm} R² {sc[nm][1]:+.3f} MAPE {sc[nm][2]:.2f}%" for nm, _, _ in SPLIT))
print(f"잔존재고 합계 평균 {inv_tot.mean():.0f} t, 재고 평균체류일 {mean_age.mean():.1f}일")
