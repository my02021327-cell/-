"""
운전이상 감지 · 신호등 관제 (BioGuard-AI Operational Guard)
================================================================================
혐기성 소화조의 운전이상을 4단계 신호등(🟢정상 / 🟡주의 / 🟠점검 / 🔴위험)으로
판정하고, 단계별 **공정제어 지시**를 운전자에게 제시한다.
실행: `python -m src.anomaly_monitor`.

────────────────────────────────────────────────────────────────────────────────
설계 원칙 : 왜 교과서 임계값을 그대로 쓰면 안 되는가
────────────────────────────────────────────────────────────────────────────────
영천 BGP 실측 분포를 보면 이 시설은 **전형적인 '산성화형'이 아니라 '암모니아 저해형'**
이다. 따라서 산성화 중심의 표준 임계는 사실상 발화하지 않는다.

  · 소화조 pH 중앙 **7.97** (p5 7.70) → 표준 산성화 임계 pH<6.8 발생률 **0.0%**
  · VFA/ALK 중앙 **0.17** (Ripley 안정기준 <0.3) → 완충 매우 양호
  · 총알칼리도 중앙 **16,935 mg/L** (통상 2,000~5,000의 3~4배) → ALK<3,000 발생률 0.3%
  · 반면 **NH4-N 중앙 3,710 mg/L**, **유리암모니아(FAN) 중앙 389 mg/L**
    → FAN 저해임계(100~150)의 **2.6배**. 측정일의 **98%가 100 mg/L 초과**

가축분뇨(유입량의 34%)의 높은 질소가 암모니아성 완충을 만들어 pH·알칼리도를 끌어올리고,
그 결과 **산성화 지표는 안전해 보이지만 실제 저해요인은 유리암모니아**인 구조다.
→ 본 모듈은 ① **문헌 절대임계**(안전 하한선)와 ② **본 시설 통계밴드**(평상운전 대비
이탈)를 **동시에** 적용하고, 지표별로 실제 발화율을 함께 보고한다.

────────────────────────────────────────────────────────────────────────────────
감시 지표 (사용자 지정 4종 + 추가 제안 8종)
────────────────────────────────────────────────────────────────────────────────
[사용자 지정]
  1. pH                     메탄생성균 활성 (본 시설은 고pH 방향 위험)
  2. 온도                    중온소화 유지
  3. VFA/알칼리도 (Ripley)    산성화·완충능 소진
  4. 암모니아                 → **총암모니아(TAN)와 유리암모니아(FAN)로 분리**.
                            저해를 일으키는 실체는 FAN 이며 pH·온도에 따라 크게 달라진다
                            (Anthonisen: FAN = TAN /(1+10^(pKa−pH)), pKa=0.09018+2729.92/T)

[추가 제안 — 근거와 함께]
  5. 알칼리도 절대값          비(比)가 정상이어도 완충 총량이 낮으면 급변에 취약
  6. VFA 절대농도            중간대사물 축적(아세트산 등가) — 비와 함께 보면 오진 방지
  7. **OLR(유기물부하율)**    kg VS/㎥·d. **혐기성소화 실패의 최다 원인은 과부하**이며
                            운전자가 직접 통제 가능한 1차 조작변수
  8. HRT(수리학적 체류시간)   과수리부하 시 미생물 세척(washout) 위험
  9. 온도 변화율 |dT/dt|      메탄생성균은 절대온도보다 **급변에 더 취약**(열충격)
 10. 메탄함량 CH4%           메탄생성균 스트레스의 조기 신호(가스 품질 저하)
 11. 비메탄수율               Nm³CH4/kgVS. 제안서 기준(0.7 주의 / 0.6 점검 / 0.5 위험)
 12. **모델 예측잔차**         본 프로젝트 앙상블 예측 대비 실측 이탈(z-score).
                            위 화학지표로 설명되지 않는 **미지 원인 이상**을 잡아내는
                            보완 지표(센서 고장·급이 오류·혼합 불량 등)

[제외 — 데이터 부재]
  · 개별 VFA(아세트산·프로피온산 등) : 원본에 컬럼은 있으나 **전 기간 값이 0%**.
    프로피온산/아세트산 비(>1.4 조기경보)는 **계측을 시작해야** 사용 가능 → 개선 권고.
  · H2S : 가용 29.5% 이나 변동이 거의 없어(σ≈0) 판별력 없음 → 참고용.

────────────────────────────────────────────────────────────────────────────────
핵심 설계 : '만성 상태' 와 '급성 이탈' 의 분리 (경보 피로 방지)
────────────────────────────────────────────────────────────────────────────────
1차 설계에서 문헌 절대임계를 모든 지표에 적용했더니 **TAN 99.6% · VFA 99.4% ·
FAN 99.1% 가 상시 발화** 했다. 매일 울리는 경보는 운전자가 무시하게 되어(alarm fatigue)
관제 기능을 잃는다. 실제로 단계별 수율 변별력도 사라졌다(정상 판정일 n=1).

원인은 **만성적 운전 특성을 급성 경보로 취급**한 것이다. 이 시설에서 높은 암모니아·VFA는
'오늘 발생한 이상'이 아니라 **기질 구성에서 비롯된 상시 조건**이다. 따라서 지표를 두 종류로
나눈다.

  ■ **급성(acute)** — 문헌 **절대임계**. 오늘 조치가 필요한 안전 이탈.
      pH · 온도 · VFA/ALK · 온도변화 · OLR · HRT · CH4함량 · 수율 · 예측이탈
  ■ **만성(chronic)** — **본 시설 자체 기준선 대비 이탈**(후행 365일 중앙값·MAD 기준
      robust z-score, 미래정보 미사용)로 판정. 절대수준은 별도의 **상시 리스크 프로파일**
      로 1회 보고하고 전략적 대응(기질 배합·희석·순응)을 권고한다.
      FAN · TAN · VFA(절대) · 알칼리도

이렇게 하면 급성 경보는 드물게·의미 있게 울리고, 만성 리스크는 놓치지 않고 관리된다.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.bio_lag import V_DIGESTER
from src.final_ensemble import (HOLDOUT_YEAR, SARIMAX_EXOG, SARIMAX_ORDER, OUT,
                                TARGET, prepare)

warnings.filterwarnings("ignore")

LEVELS = {0: "정상", 1: "주의", 2: "점검", 3: "위험"}
# matplotlib 한글 폰트가 없는 환경이므로 그래프 라벨은 영문으로 표기
LEVELS_EN = {0: "Normal", 1: "Watch", 2: "Inspect", 3: "Critical"}
NAME_EN = {"pH": "pH", "온도": "Temperature", "VFA_ALK": "VFA/Alk", "FAN": "Free NH3",
           "TAN": "Total NH3", "알칼리도": "Alkalinity", "VFA": "VFA (abs)",
           "OLR": "OLR", "HRT": "HRT", "온도변화": "dT/dt",
           "CH4함량": "CH4 content", "수율": "CH4 yield", "예측이탈": "Model deviation"}
LEVEL_COLOR = {0: "#2E9B4F", 1: "#E8C33C", 2: "#E2872B", 3: "#C0392B"}

# ──────────────────────────────────────────────────────────────────────────────
# 지표 정의 : (하위밴드, 상위밴드) 로 단계 판정
#   band_hi : 값이 클수록 위험 → [주의, 점검, 위험] 시작점
#   band_lo : 값이 작을수록 위험 → [주의, 점검, 위험] 시작점(내림차순)
# ──────────────────────────────────────────────────────────────────────────────
INDICATORS = {
    "pH": dict(col="소화조_pH", unit="-", group="사용자지정",
               band_hi=[8.10, 8.30, 8.50], band_lo=[7.60, 7.30, 7.00],
               why="메탄생성균 활성 pH. 본 시설은 암모니아성 완충으로 고pH 방향이 위험",
               act={1: "유입 C/N 확인, 가축분뇨 비율 소폭 하향",
                    2: "질소 부하 저감(가축분뇨↓·음폐수↑), 희석수 검토",
                    3: "즉시 질소성 기질 투입 중단, 희석·반송 개시, 정밀분석 의뢰"}),
    "온도": dict(col="소화조_온도", unit="℃", group="사용자지정",
               band_hi=[40.0, 41.5, 43.0], band_lo=[36.0, 34.0, 32.0],
               why="중온소화(38±2℃) 유지. 이탈 시 메탄생성균 활성 급감",
               act={1: "열교환기·순환펌프 상태 점검",
                    2: "가온 설비 정비, 투입온도 조정",
                    3: "가온계통 긴급 점검, 투입량 감량으로 부하 완화"}),
    "VFA_ALK": dict(col="VFA_ALK", unit="-", group="사용자지정",
               band_hi=[0.30, 0.40, 0.80], band_lo=None,
               why="Ripley 산성화 지표. 완충능 대비 중간대사물 축적 정도",
               act={1: "투입량 유지·모니터링 강화(일 2회 분석)",
                    2: "OLR 10~20% 감량, 알칼리도 보충 검토",
                    3: "투입 일시 중단, 알칼리제(중조) 투입, 반송·희석"}),
    "FAN": dict(mode="chronic", col="FAN", unit="mg/L", group="사용자지정(세분)",
               band_hi=[100.0, 200.0, 400.0], band_lo=None,
               why="유리암모니아(Anthonisen). 실제 저해를 일으키는 화학종 — pH·온도 의존",
               act={1: "질소 부하 추적, C/N 비 개선(고탄소 기질 병합)",
                    2: "가축분뇨 비율 하향·희석, pH 상승 억제",
                    3: "질소성 기질 투입 중단, 희석·부분 배출, 순응 미생물 보충 검토"}),
    "TAN": dict(mode="chronic", col="NH4N", unit="mg/L", group="사용자지정(세분)",
               band_hi=[1500.0, 3000.0, 4000.0], band_lo=None,
               why="총암모니아. FAN 의 모집단이며 장기 축적 추세 관리 지표",
               act={1: "유입 질소 모니터링", 2: "고질소 기질 비율 조정",
                    3: "희석·부분 배출로 총암모니아 저감"}),
    "알칼리도": dict(mode="chronic", col="소화조_TAlk", unit="mg/L", group="추가제안",
               band_hi=None, band_lo=[4000.0, 3000.0, 2000.0],
               why="완충 총량. 비가 정상이어도 총량이 낮으면 급변에 취약",
               act={1: "알칼리도 추이 관찰", 2: "중조 보충 검토",
                    3: "알칼리제 즉시 투입, 투입량 감량"}),
    "VFA": dict(mode="chronic", col="소화조_VFA", unit="mg/L", group="추가제안",
               band_hi=[2000.0, 3000.0, 4000.0], band_lo=None,
               why="중간대사물 절대 축적량. 비(比)와 함께 보아 오진 방지",
               act={1: "분석 주기 단축", 2: "OLR 감량 검토",
                    3: "투입 중단·원인(과부하/저해) 규명"}),
    "OLR": dict(col="OLR", unit="kgVS/㎥·d", group="추가제안",
               band_hi=[2.5, 3.5, 4.5], band_lo=None,
               why="유기물부하율. **소화조 실패의 최다 원인이 과부하**이며 1차 조작변수",
               act={1: "투입 계획 재확인", 2: "투입량 10~20% 감량",
                    3: "투입량 30% 이상 감량 또는 일시 중단"}),
    "HRT": dict(col="HRT", unit="d", group="추가제안",
               band_hi=None, band_lo=[25.0, 20.0, 15.0],
               why="체류시간. 과수리부하 시 미생물 세척(washout) 위험",
               act={1: "수리부하 점검", 2: "투입 유량 감량",
                    3: "유량 대폭 감량, 미생물 유실 여부 확인"}),
    "온도변화": dict(col="dT", unit="℃/d", group="추가제안",
               band_hi=[0.5, 1.0, 2.0], band_lo=None,
               why="메탄생성균은 절대온도보다 **급변(열충격)** 에 더 취약",
               act={1: "온도 제어 안정화", 2: "가온 제어 튜닝, 급변 원인 제거",
                    3: "가온계통 긴급 점검, 투입 일시 조정"}),
    "CH4함량": dict(col="CH4_content", unit="%", group="추가제안",
               band_hi=None, band_lo=[60.0, 55.0, 50.0],
               why="가스 품질. 메탄생성균 스트레스의 조기 신호",
               act={1: "가스 조성 추이 관찰", 2: "부하·저해 인자 점검",
                    3: "정밀 진단(저해·과부하·누설) 및 부하 저감"}),
    "수율": dict(col="yield", unit="Nm³CH4/kgVS", group="추가제안",
               band_hi=None, band_lo=[0.70, 0.60, 0.50],
               why="비메탄수율(제안서 기준 0.7/0.6/0.5). 전환효율 종합 지표",
               act={1: "기질 품질·전처리 점검", 2: "부하·저해 인자 종합 점검",
                    3: "운전조건 전면 재검토, 원인 규명 후 부하 재설정"}),
    "예측이탈": dict(col="resid_z", unit="σ", group="추가제안",
               band_hi=[2.0, 3.0, 4.0], band_lo=None,
               why="앙상블 예측 대비 실측 이탈. 화학지표로 설명 안 되는 **미지 원인** 감지",
               act={1: "계측기·급이 기록 교차확인",
                    2: "센서 검교정, 혼합·급이 설비 점검",
                    3: "계측 신뢰성 및 공정 이상 정밀 진단"}),
}


# ──────────────────────────────────────────────────────────────────────────────
def build_monitor_frame() -> pd.DataFrame:
    """감시 지표 산출 (파생 + 보조 데이터 결합 + 모델 예측잔차)."""
    df = prepare()
    extra = pd.read_excel("data/monitor_extra.xlsx").set_index("date")
    df = df.join(extra)

    # 물리적으로 불가능한 값 제거
    df.loc[(df.CH4_content < 40) | (df.CH4_content > 75), "CH4_content"] = np.nan

    df["VFA_ALK"] = df["소화조_VFA"] / df["소화조_TAlk"].replace(0, np.nan)
    df["OLR"] = df["VS_in"] / V_DIGESTER
    df["HRT"] = V_DIGESTER / df["투입량합계"].replace(0, np.nan)
    df["dT"] = df["소화조_온도"].diff().abs()
    df["yield"] = df[TARGET] / df["VS_in"].replace(0, np.nan)

    # 유리암모니아 (Anthonisen)
    T = df["소화조_온도"] + 273.15
    pKa = 0.09018 + 2729.92 / T
    df["FAN"] = df["NH4N"] / (1 + 10 ** (pKa - df["소화조_pH"]))

    # 모델 예측잔차 z-score (학습기 잔차 σ 기준)
    endog = df[TARGET].astype(float)
    exog = df[SARIMAX_EXOG].ffill().bfill()
    cut = int((df.index.year < HOLDOUT_YEAR).sum())
    res = SARIMAX(endog.iloc[:cut], exog=exog.iloc[:cut], order=SARIMAX_ORDER,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, maxiter=300)
    pred = res.apply(endog, exog=exog).get_prediction(
        start=endog.index[1], dynamic=False).predicted_mean
    resid = endog - pred.reindex(endog.index)
    sigma = resid[resid.index.year < HOLDOUT_YEAR].std()
    df["pred"] = pred.reindex(df.index)
    df["resid_z"] = (resid / sigma).abs()
    return df


def level_of(value: float, spec: dict) -> float:
    """급성 지표 : 문헌 절대임계로 단계(0~3) 판정. 결측이면 NaN."""
    if pd.isna(value):
        return np.nan
    lv = 0
    if spec.get("band_hi"):
        for i, t in enumerate(spec["band_hi"], start=1):
            if value >= t:
                lv = max(lv, i)
    if spec.get("band_lo"):
        for i, t in enumerate(spec["band_lo"], start=1):
            if value <= t:
                lv = max(lv, i)
    return float(lv)


# 만성 지표 : 본 시설 기준선 대비 robust z-score 밴드(주의/점검/위험)
CHRONIC_Z = [2.0, 3.0, 4.0]
BASE_WIN = 365      # 후행 기준선 창(일)
BASE_MIN = 120      # 기준선 최소 표본


def relative_level(s: pd.Series, spec: dict) -> pd.Series:
    """후행(causal) 중앙값·MAD 기준 robust z-score → 단계.
    미래 정보를 쓰지 않도록 shift(1) 후 rolling 으로 기준선을 만든다."""
    prev = s.shift(1)
    med = prev.rolling(BASE_WIN, min_periods=BASE_MIN).median()
    mad = (prev - med).abs().rolling(BASE_WIN, min_periods=BASE_MIN).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    # 위험 방향 : band_hi 가 있으면 상방, band_lo 면 하방
    if spec.get("band_hi"):
        zz = z
    else:
        zz = -z
    lv = pd.Series(np.nan, index=s.index)
    ok = zz.notna() & s.notna()
    lv[ok] = 0.0
    for i, t in enumerate(CHRONIC_Z, start=1):
        lv[ok & (zz >= t)] = float(i)

    # ── 절대안전 상한 : 상대 이탈만으로 과도한 등급이 나오지 않도록 제한 ──
    # 알칼리도 중앙 16,935 mg/L 처럼 절대수준이 충분히 안전하면, 통계적 이탈이 커도
    # 공정 위험이 아니다. 따라서 최종 등급 = min(상대등급, 절대등급 + 1) 로 캡을 건다.
    # (상대 이탈은 '조기경보' 로 최대 한 단계만 격상시킬 수 있다)
    abs_lv = s.apply(lambda v: level_of(v, spec))
    lv[ok] = np.minimum(lv[ok], abs_lv[ok] + 1)
    return lv


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, spec in INDICATORS.items():
        c = spec["col"]
        if c not in df:
            out[name] = np.nan
            continue
        if spec.get("mode") == "chronic":
            out[name] = relative_level(df[c], spec)        # 시설 기준선 대비 이탈
        else:
            out[name] = df[c].apply(lambda v: level_of(v, spec))
    # 종합 = 관측된 지표 중 최악 단계 (보수적 관제)
    out["종합"] = out[list(INDICATORS)].max(axis=1, skipna=True)
    out["관측지표수"] = out[list(INDICATORS)].notna().sum(axis=1)
    return out


def chronic_profile(df: pd.DataFrame) -> dict:
    """만성 리스크 상시 프로파일 — 절대수준 1회 평가(일일 경보가 아님)."""
    prof = {}
    for name in [n for n, s in INDICATORS.items() if s.get("mode") == "chronic"]:
        spec = INDICATORS[name]
        s = df[spec["col"]].dropna()
        if s.empty:
            continue
        med = float(s.median())
        lv = level_of(med, spec)
        prof[name] = {"중앙값": round(med, 1), "단위": spec["unit"],
                      "절대수준_판정": LEVELS[int(lv)],
                      "가용률_%": round(float(df[spec["col"]].notna().mean() * 100), 1),
                      "전략적_대응": spec["act"].get(int(lv), "현 수준 유지")}
    return prof


def advise(df: pd.DataFrame, lv: pd.DataFrame, day) -> dict:
    """특정일의 신호등 판정 + 원인지표 + 공정제어 지시."""
    row, lrow = df.loc[day], lv.loc[day]
    items = []
    for name, spec in INDICATORS.items():
        l = lrow.get(name)
        if pd.isna(l) or l == 0:
            continue
        items.append({"지표": name, "단계": LEVELS[int(l)],
                      "값": round(float(row[spec["col"]]), 3), "단위": spec["unit"],
                      "지시": spec["act"][int(l)], "근거": spec["why"]})
    items.sort(key=lambda x: -list(LEVELS.values()).index(x["단계"]))
    overall = lrow["종합"]
    return {"date": str(pd.Timestamp(day).date()),
            "종합": LEVELS[int(overall)] if not pd.isna(overall) else "판정불가",
            "원인지표": items}


# ──────────────────────────────────────────────────────────────────────────────
# 검증 : 경보가 실제 효율 저하와 연결되는가
# ──────────────────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame, lv: pd.DataFrame) -> dict:
    """경보가 실제 성능저하와 연결되는지 검증.

    ⚠ 수율(=메탄/VS_in)은 **분모 효과** 때문에 검증지표로 부적합하다. 저투입일에는
    분모가 작아 수율이 기계적으로 높아진다(VS_in 저사분위 수율 1.00 vs 고사분위 0.64).
    실제로 저투입일에 몰린 온도·알칼리도 경보가 '고수율'로 나타나 해석이 뒤집힌다.
    → 분모 효과가 없는 **CH4 함량(%)** 과 **모델 예측잔차** 로 검증한다.
    """
    def by_level(series, name, fwd=False):
        s = series.shift(-1).rolling(7, min_periods=3).mean().shift(-6) if fwd else series
        ok = lv["종합"].notna() & s.notna()
        g = s[ok].groupby(lv.loc[ok, "종합"])
        return {LEVELS[int(k)]: {"n": int(v.size), name: round(float(v.median()), 3)}
                for k, v in g}

    resid = df[TARGET] - df["pred"]
    fire = {n: round(float((lv[n] >= 1).sum() / lv[n].notna().sum() * 100), 1)
            for n in INDICATORS if lv[n].notna().sum() > 0}
    return {
        "CH4함량_by_단계": by_level(df["CH4_content"], "CH4_%"),
        "향후7일_CH4함량_by_단계": by_level(df["CH4_content"], "CH4_%", fwd=True),
        "예측잔차_by_단계": by_level(resid, "잔차_Nm3"),
        "수율_by_단계_참고": by_level(df["yield"], "수율"),
        "수율_검증_부적합_사유": "분모(VS_in) 효과 — 저투입일 수율이 기계적으로 상승",
        "지표별_주의이상_발화율_%": fire}


# ──────────────────────────────────────────────────────────────────────────────
def plot_dashboard(df, lv, val, path):
    names = list(INDICATORS)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=.45, wspace=.22)
    fig.suptitle("Operational Guard — Anomaly Traffic-Light Monitoring",
                 fontsize=15, fontweight="bold", y=.975)
    cmap = ListedColormap([LEVEL_COLOR[i] for i in range(4)])

    # A. 지표별 단계 타임라인
    ax = fig.add_subplot(gs[0, :])
    M = lv[names].T.values.astype(float)
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-.5, vmax=3.5,
              interpolation="nearest",
              extent=[0, len(lv), len(names), 0])
    ax.set_yticks(np.arange(len(names)) + .5)
    ax.set_yticklabels([NAME_EN.get(n, n) for n in names], fontsize=8)
    step = max(len(lv) // 12, 1)
    ax.set_xticks(np.arange(0, len(lv), step))
    ax.set_xticklabels([str(d.date())[:7] for d in lv.index[::step]], fontsize=7, rotation=45)
    ax.set_title("A. Indicator status timeline (green=OK, yellow=watch, orange=inspect, red=critical; white=no data)",
                 fontsize=10, fontweight="bold")

    # B. 종합 단계 분포
    ax = fig.add_subplot(gs[1, 0])
    cnt = lv["종합"].value_counts().reindex([0, 1, 2, 3]).fillna(0)
    ax.bar([LEVELS_EN[i] for i in range(4)], cnt.values,
           color=[LEVEL_COLOR[i] for i in range(4)])
    tot = cnt.sum()
    for i, v in enumerate(cnt.values):
        ax.text(i, v, f"{int(v)}\n({v/tot*100:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("days"); ax.set_title("B. Overall status distribution", fontsize=10, fontweight="bold")

    # C. 지표별 발화율
    ax = fig.add_subplot(gs[1, 1])
    fr = pd.Series(val["지표별_주의이상_발화율_%"]).sort_values()
    ax.barh(range(len(fr)), fr.values, color="#4F81BD")
    ax.set_yticks(range(len(fr))); ax.set_yticklabels([NAME_EN.get(i,i) for i in fr.index], fontsize=8)
    ax.set_xlabel("% of measured days at ≥ watch")
    ax.set_title("C. Alarm firing rate by indicator", fontsize=10, fontweight="bold")

    # D. FAN 시계열(핵심 위험)
    ax = fig.add_subplot(gs[2, 0])
    s = df["FAN"].dropna()
    ax.scatter(s.index, s.values, s=12, color="#C0392B", alpha=.7)
    for t, c, l in [(100, "#E8C33C", "watch 100"), (200, "#E2872B", "inspect 200"),
                    (400, "#C0392B", "critical 400")]:
        ax.axhline(t, color=c, ls="--", lw=1, label=l)
    ax.set_ylabel("Free ammonia (mg/L)"); ax.legend(fontsize=7)
    ax.set_title("D. Free ammonia — chronic inhibition risk", fontsize=10, fontweight="bold")

    # E. 단계별 수율
    ax = fig.add_subplot(gs[2, 1])
    lab = [k for k in ["정상", "주의", "점검", "위험"] if k in val["CH4함량_by_단계"]]
    yv = [val["CH4함량_by_단계"][k]["CH4_%"] for k in lab]
    nn = [val["CH4함량_by_단계"][k]["n"] for k in lab]
    ax.bar([LEVELS_EN[list(LEVELS.values()).index(k)] for k in lab], yv,
           color=[LEVEL_COLOR[list(LEVELS.values()).index(k)] for k in lab])
    for i, (v, n) in enumerate(zip(yv, nn)):
        ax.text(i, v, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(yv) - 2, max(yv) + 2)
    ax.set_ylabel("median CH4 content (%)")
    ax.set_title("E. CH4 content vs alarm level (validation)", fontsize=10, fontweight="bold")

    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    df = build_monitor_frame()
    lv = evaluate(df)
    val = validate(df, lv)

    print("===== 운전이상 신호등 관제 =====")
    print(f"  감시 지표 {len(INDICATORS)}종 (사용자 지정 4 + 추가 제안 8)")
    cnt = lv["종합"].value_counts().reindex([0, 1, 2, 3]).fillna(0)
    tot = cnt.sum()
    print(f"\n  [종합 판정 분포] 판정가능 {int(tot)}일")
    for i in range(4):
        print(f"    {LEVELS[i]:4s} {int(cnt[i]):5d}일 ({cnt[i]/tot*100:5.1f}%)")

    prof = chronic_profile(df)
    print("\n  [만성 리스크 상시 프로파일] — 일일 경보가 아닌 전략적 관리 대상")
    for k, v in prof.items():
        print(f"    {k:6s} 중앙 {v['중앙값']:>8}{v['단위']:<6} 절대수준={v['절대수준_판정']:4s} "
              f"(가용 {v['가용률_%']}%)")
        print(f"           → {v['전략적_대응']}")

    print("\n  [지표별 '주의 이상' 발화율] (만성지표는 시설 기준선 대비 이탈 기준)")
    for k, v in sorted(val["지표별_주의이상_발화율_%"].items(), key=lambda x: -x[1]):
        tag = "만성" if INDICATORS[k].get("mode") == "chronic" else "급성"
        print(f"    {k:8s} {v:5.1f}%   [{tag}] ({INDICATORS[k]['group']})")

    print("\n  [검증] 경보 단계별 성능 (분모효과 없는 지표 사용)")
    print(f"    {'단계':4s} {'n':>5s} {'CH4함량%':>9s} {'향후7일CH4%':>11s} {'예측잔차':>10s} {'(참고)수율':>9s}")
    for k in ["정상", "주의", "점검", "위험"]:
        a = val["CH4함량_by_단계"].get(k); b = val["향후7일_CH4함량_by_단계"].get(k)
        c = val["예측잔차_by_단계"].get(k); d = val["수율_by_단계_참고"].get(k)
        if not a:
            continue
        print(f"    {k:4s} {a['n']:5d} {a['CH4_%']:9.2f} {(b['CH4_%'] if b else float('nan')):11.2f} "
              f"{(c['잔차_Nm3'] if c else float('nan')):10.1f} {(d['수율'] if d else float('nan')):9.3f}")
    print(f"    ※ 수율은 분모(VS_in) 효과로 검증 부적합 — 참고용만 표기")

    # 최근 위험/점검 사례
    worst = lv[lv["종합"] >= 2].index
    print(f"\n  [점검·위험 발생일 {len(worst)}일] 최근 사례:")
    for day in list(worst)[-2:]:
        a = advise(df, lv, day)
        print(f"    ── {a['date']} : 종합 {a['종합']}")
        for it in a["원인지표"][:4]:
            print(f"       [{it['단계']}] {it['지표']}={it['값']}{it['단위']} → {it['지시']}")

    plot_dashboard(df, lv, val, f"{OUT}/anomaly_dashboard.png")
    lv.assign(**{c: df[c] for c in ["소화조_pH", "소화조_온도", "VFA_ALK", "FAN",
                                    "OLR", "yield"]}).to_csv(f"{OUT}/anomaly_levels.csv")
    with open(f"{OUT}/anomaly_report.json", "w", encoding="utf-8") as f:
        json.dump({"indicators": {k: {"group": v["group"], "unit": v["unit"],
                                      "band_hi": v["band_hi"], "band_lo": v["band_lo"],
                                      "why": v["why"], "actions": v["act"]}
                                  for k, v in INDICATORS.items()},
                   "status_distribution": {LEVELS[i]: int(cnt[i]) for i in range(4)},
                   "chronic_profile": prof,
                   "design": {"acute": "문헌 절대임계(오늘 조치 필요)",
                              "chronic": f"시설 기준선(후행 {BASE_WIN}일 중앙값·MAD) 대비 "
                                         f"robust z {CHRONIC_Z} — 경보피로 방지"},
                   "validation": val,
                   "site_characterization": {
                       "failure_mode": "암모니아 저해형 (산성화형 아님)",
                       "pH_median": 7.97, "VFA_ALK_median": 0.17,
                       "TAlk_median": 16935, "FAN_median": 389,
                       "note": "pH<6.8 발생률 0.0% — 표준 산성화 임계는 본 시설에서 미발화"},
                   "excluded": {"개별VFA(프로피온산/아세트산)": "원본 데이터 0% — 계측 시작 권고",
                                "H2S": "변동 거의 없어 판별력 없음(참고용)"}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  산출: {OUT}/anomaly_dashboard.png, anomaly_levels.csv, anomaly_report.json")


if __name__ == "__main__":
    main()
