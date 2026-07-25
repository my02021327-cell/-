"""
운전이상 경보 시스템 (Dashboard Alarm Engine)
================================================================================
**예측 모델과 무관한 독립 시스템.** 대시보드에 입력되는 계측값이 **설정값(setpoint)**
을 이탈하면 운전자에게 4단계 경보(🟢정상 / 🟡주의 / 🟠점검 / 🔴위험)와 **공정제어 지시**
를 제공한다. 실행: `python -m src.alarm_system`.

    입력값(계측) → 설정값 비교 → 경보 단계 → 운전자 조치 지시

────────────────────────────────────────────────────────────────────────────────
경보 변수 선정 (판단 근거)
────────────────────────────────────────────────────────────────────────────────
[Tier 1 · 공정 안정성 직결 — 즉시 조치]
  pH · VFA/알칼리도 · 온도 · **유리암모니아(FAN)**
    · FAN 은 총암모니아(TAN)와 달리 **실제 저해를 일으키는 화학종**이며 같은 TAN 이라도
      pH·온도에 따라 수배 달라진다 → Anthonisen 식으로 산출해 감시한다.
      FAN = TAN / (1 + 10^(pKa − pH)),  pKa = 0.09018 + 2729.92/T(K)

[Tier 2 · 부하 — 운전자가 직접 조작하는 1차 변수]
  OLR(유기물부하율) · HRT(체류시간) · 온도변화율(|dT/dt|)
    · **혐기성소화 실패의 최다 원인은 과부하(OLR)** 이며, 메탄생성균은 절대온도보다
      **급변(열충격)** 에 더 취약하다.

[Tier 3 · 보조 — 원인 규명·추세 관리]
  총암모니아(TAN) · VFA 절대농도 · 알칼리도 절대값 · CH4 함량

[경보에서 제외 — 판단 근거]
  · **비메탄수율** : 분모(VS_in) 효과로 **저투입일에 수율이 기계적으로 상승**한다
    (VS_in 저사분위 1.00 vs 고사분위 0.64). 실제로 발화율 38%로 최다였고 경보-성능
    검증에서 해석이 뒤집혀 **오경보원**으로 확인 → 경보 제외, 성능 참고지표로만 사용.
  · **모델 예측잔차** : 본 시스템은 계측값-설정값 비교 전용이므로 모델 의존 지표는 제외.
  · **개별 VFA(프로피온산/아세트산 비)** : 원본에 컬럼만 있고 값이 전 기간 0% → 계측 개시 권고.
  · **H2S** : 변동이 거의 없어(σ≈0) 판별력 없음.

────────────────────────────────────────────────────────────────────────────────
설정값을 '문헌값 그대로' 쓰지 않은 이유
────────────────────────────────────────────────────────────────────────────────
본 시설은 **산성화형이 아니라 암모니아 저해형**이다.
  · pH 중앙 7.97 → 문헌 산성화 임계(pH<6.8) 발생률 **0.0%** (영구 미발화)
  · 총알칼리도 중앙 16,935 mg/L (통상 2,000~5,000의 3~4배), VFA/ALK 중앙 0.17
  · 그러나 **FAN 중앙 389 mg/L** — 문헌 저해임계(100~150)를 **측정일의 98%가 초과**

문헌 절대임계를 그대로 적용하면 FAN·TAN·VFA 가 **매일 발화(99%)** 하여 운전자가 경보를
무시하게 된다(alarm fatigue). 따라서 설정값은 다음 원칙으로 정했다.

  ■ 문헌 안전한계가 유효한 변수(pH·온도·VFA/ALK·OLR·HRT·dT/dt·CH4%)
      → **문헌·설계 기준** 을 설정값으로 사용
  ■ 만성적으로 높은 변수(FAN·TAN·VFA·알칼리도)
      → **본 시설 실측 분포(p75/p90/p97.5)** 를 설정값으로 사용해 '평소 대비 이탈'을 잡고,
        절대수준의 위험은 **상시 만성 리스크**로 대시보드에 항상 표시한다.
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

from src.bio_lag import V_DIGESTER
from src.final_ensemble import OUT, TARGET, prepare

warnings.filterwarnings("ignore")

LEVELS = {0: "정상", 1: "주의", 2: "점검", 3: "위험"}
LEVELS_EN = {0: "Normal", 1: "Watch", 2: "Inspect", 3: "Critical"}
LEVEL_COLOR = {0: "#2E9B4F", 1: "#E8C33C", 2: "#E2872B", 3: "#C0392B"}

# ──────────────────────────────────────────────────────────────────────────────
# 설정값(SETPOINT) 표
#   normal : 정상 운전 범위
#   hi / lo: [주의, 점검, 위험] 이탈 시작점 (hi=상방 위험, lo=하방 위험)
#   basis  : 설정 근거 (문헌 / 본 시설 실측 분위수)
# ──────────────────────────────────────────────────────────────────────────────
SETPOINTS = {
    # ── Tier 1 : 공정 안정성 직결 ─────────────────────────────────────────────
    "pH": dict(tier=1, col="소화조_pH", unit="", normal="7.7 ~ 8.1",
               hi=[8.10, 8.30, 8.40], lo=[7.70, 7.50, 7.20],
               basis="본 시설 안정운전 p5~p95(7.71~8.13) + 메탄생성균 활성역",
               act={1: "유입 C/N 확인, 가축분뇨 비율 소폭 조정",
                    2: "질소 부하 저감(가축분뇨↓·음폐수↑), 희석수 투입 검토",
                    3: "질소성 기질 투입 중단, 희석·반송 개시, 정밀분석 의뢰"}),
    "VFA/Alk": dict(tier=1, col="VFA_ALK", unit="", normal="< 0.30",
               hi=[0.30, 0.40, 0.60], lo=None,
               basis="Ripley 문헌기준(<0.3 안정). 본 시설 p97.5=0.33",
               act={1: "분석 주기 단축(일 2회), 투입량 현 수준 유지",
                    2: "OLR 10~20% 감량, 알칼리도 보충 검토",
                    3: "투입 일시 중단, 알칼리제(중조) 투입, 반송·희석"}),
    "온도": dict(tier=1, col="소화조_온도", unit="℃", normal="36.0 ~ 40.0",
               hi=[40.0, 41.0, 42.5], lo=[36.0, 34.0, 32.0],
               basis="중온소화 38±2℃ (본 시설 안정운전 중앙 38.0)",
               act={1: "열교환기·순환펌프 상태 점검",
                    2: "가온 설비 정비, 투입 온도 조정",
                    3: "가온계통 긴급 점검, 투입량 감량으로 부하 완화"}),
    "FAN": dict(tier=1, col="FAN", unit="mg/L", normal="< 530",
               hi=[530.0, 650.0, 740.0], lo=None, chronic=True,
               basis="본 시설 실측 p75/p90/p97.5 (문헌 100~150 은 상시 초과로 사용 불가)",
               act={1: "질소 부하 추적, C/N 비 개선(고탄소 기질 병합)",
                    2: "가축분뇨 비율 하향·희석, pH 상승 억제",
                    3: "질소성 기질 투입 중단, 희석·부분 배출, 순응 미생물 보충"}),
    # ── Tier 2 : 부하(운전자 직접 조작) ───────────────────────────────────────
    "OLR": dict(tier=2, col="OLR", unit="kgVS/㎥·d", normal="< 1.8",
               hi=[1.8, 2.5, 3.5], lo=None,
               basis="본 시설 p97.5=1.81 + 중온 CSTR 설계 상한(2.5~3.5)",
               act={1: "투입 계획 재확인, 기질 농도 점검",
                    2: "투입량 10~20% 감량",
                    3: "투입량 30% 이상 감량 또는 일시 중단"}),
    "HRT": dict(tier=2, col="HRT", unit="d", normal="> 30", hi=None,
               lo=[30.0, 25.0, 20.0],
               basis="본 시설 p2.5=30.3 (설계 HRT 40.6일)",
               act={1: "수리부하 점검, 유입 유량 확인",
                    2: "투입 유량 감량",
                    3: "유량 대폭 감량, 미생물 유실(washout) 여부 확인"}),
    "온도변화": dict(tier=2, col="dT", unit="℃/d", normal="< 0.5",
               hi=[0.5, 1.0, 2.0], lo=None,
               basis="열충격 방지(문헌 <1℃/일). 본 시설 p97.5=0.35",
               act={1: "온도 제어 안정화 확인",
                    2: "가온 제어 튜닝, 급변 원인 제거",
                    3: "가온계통 긴급 점검, 투입 일시 조정"}),
    # ── Tier 3 : 보조(원인 규명·추세) ─────────────────────────────────────────
    "TAN": dict(tier=3, col="NH4N", unit="mg/L", normal="< 4,500",
               hi=[4500.0, 5300.0, 5750.0], lo=None, chronic=True,
               basis="본 시설 실측 p75/p90/p97.5 (FAN 의 모집단·추세관리용)",
               act={1: "유입 질소 모니터링 강화",
                    2: "고질소 기질(가축분뇨) 비율 조정",
                    3: "희석·부분 배출로 총암모니아 저감"}),
    "VFA": dict(tier=3, col="소화조_VFA", unit="mg/L", normal="< 3,100",
               hi=[3100.0, 3350.0, 3730.0], lo=None, chronic=True,
               basis="본 시설 실측 p75/p90/p97.5 (절대 축적량 추세)",
               act={1: "분석 주기 단축", 2: "OLR 감량 검토",
                    3: "투입 중단, 원인(과부하/저해) 규명"}),
    "알칼리도": dict(tier=3, col="소화조_TAlk", unit="mg/L", normal="> 10,800",
               hi=None, lo=[10800.0, 8000.0, 5000.0], chronic=True,
               basis="본 시설 실측 p10=10,832 + 완충능 확보 하한",
               act={1: "알칼리도 추이 관찰",
                    2: "중조 보충 검토, 투입량 조정",
                    3: "알칼리제 즉시 투입, 투입량 감량"}),
    "CH4함량": dict(tier=3, col="CH4_content", unit="%", normal="> 61",
               hi=None, lo=[61.0, 58.0, 55.0],
               basis="본 시설 p2.5=60.7 (가스 품질 저하 = 메탄생성균 스트레스)",
               act={1: "가스 조성 추이 관찰",
                    2: "부하·저해 인자 점검",
                    3: "정밀 진단(저해·과부하·누설) 및 부하 저감"}),
}

TIER_NAME = {1: "공정안정성", 2: "부하", 3: "보조"}
NAME_EN = {"pH": "pH", "VFA/Alk": "VFA/Alk", "온도": "Temperature", "FAN": "Free NH3",
           "OLR": "OLR", "HRT": "HRT", "온도변화": "dT/dt", "TAN": "Total NH3",
           "VFA": "VFA (abs)", "알칼리도": "Alkalinity", "CH4함량": "CH4 content"}


# ──────────────────────────────────────────────────────────────────────────────
# 경보 엔진 (대시보드 실시간 호출용)
# ──────────────────────────────────────────────────────────────────────────────
def check_value(name: str, value: float) -> int | None:
    """계측값 1건을 설정값과 비교해 경보 단계(0~3) 반환. 값이 없으면 None."""
    sp = SETPOINTS[name]
    if value is None or pd.isna(value):
        return None
    lv = 0
    if sp["hi"]:
        for i, t in enumerate(sp["hi"], start=1):
            if value >= t:
                lv = max(lv, i)
    if sp["lo"]:
        for i, t in enumerate(sp["lo"], start=1):
            if value <= t:
                lv = max(lv, i)
    return lv


def check_all(readings: dict) -> dict:
    """**대시보드 진입점** — 현재 계측값 dict → 종합 경보 + 변수별 상태 + 조치 지시.

    readings 예) {"pH":8.35, "온도":33.1, "VFA/Alk":0.28, "OLR":1.2, ...}
    """
    items, worst = [], -1
    for name, sp in SETPOINTS.items():
        v = readings.get(name)
        lv = check_value(name, v)
        if lv is None:
            continue
        worst = max(worst, lv)
        items.append({"변수": name, "현재값": round(float(v), 3), "단위": sp["unit"],
                      "설정값(정상)": sp["normal"], "단계": LEVELS[lv], "_lv": lv,
                      "구분": TIER_NAME[sp["tier"]],
                      "조치": sp["act"][lv] if lv > 0 else "-"})
    items.sort(key=lambda x: (-x["_lv"], SETPOINTS[x["변수"]]["tier"]))
    return {"종합경보": LEVELS[worst] if worst >= 0 else "판정불가",
            "이탈변수수": sum(1 for i in items if i["_lv"] > 0),
            "변수상태": items}


def format_alarm(res: dict) -> str:
    """운전자용 텍스트 출력."""
    mark = {"정상": "🟢", "주의": "🟡", "점검": "🟠", "위험": "🔴"}
    out = [f"{mark.get(res['종합경보'],'⚪')} 종합경보: {res['종합경보']}"
           f"  (이탈 {res['이탈변수수']}건)"]
    for it in res["변수상태"]:
        if it["_lv"] == 0:
            continue
        out.append(f"  {mark[it['단계']]} [{it['단계']}] {it['변수']} = "
                   f"{it['현재값']}{it['단위']}  (설정 {it['설정값(정상)']}) [{it['구분']}]")
        out.append(f"        └ 조치: {it['조치']}")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# 이력 평가 · 검증
# ──────────────────────────────────────────────────────────────────────────────
def build_frame() -> pd.DataFrame:
    """계측 이력 + 파생지표(FAN·OLR·HRT·dT)."""
    df = prepare()
    df = df.join(pd.read_excel("data/monitor_extra.xlsx").set_index("date"))
    df.loc[(df.CH4_content < 40) | (df.CH4_content > 75), "CH4_content"] = np.nan
    df["VFA_ALK"] = df["소화조_VFA"] / df["소화조_TAlk"].replace(0, np.nan)
    df["OLR"] = df["VS_in"] / V_DIGESTER
    df["HRT"] = V_DIGESTER / df["투입량합계"].replace(0, np.nan)
    df["dT"] = df["소화조_온도"].diff().abs()
    T = df["소화조_온도"] + 273.15
    df["FAN"] = df["NH4N"] / (1 + 10 ** ((0.09018 + 2729.92 / T) - df["소화조_pH"]))
    df["yield"] = df[TARGET] / df["VS_in"].replace(0, np.nan)
    return df


def evaluate_history(df: pd.DataFrame) -> pd.DataFrame:
    """이력 전체에 설정값 적용.

    종합경보 = **Tier 1·2 가 주도**하고, Tier 3(보조)은 최대 '주의'까지만 기여한다.
    보조지표(TAN·VFA절대·알칼리도·CH4함량)는 원인 규명용이라 단독으로 '점검·위험'을
    띄우면 상위 경보가 희석된다.
    """
    lv = pd.DataFrame(index=df.index)
    for name, sp in SETPOINTS.items():
        lv[name] = df[sp["col"]].apply(lambda v: check_value(name, v)) if sp["col"] in df else np.nan
    t12 = [n for n, s in SETPOINTS.items() if s["tier"] <= 2]
    t3 = [n for n, s in SETPOINTS.items() if s["tier"] == 3]
    lv["종합"] = pd.concat([lv[t12].max(axis=1, skipna=True),
                          lv[t3].max(axis=1, skipna=True).clip(upper=1)],
                         axis=1).max(axis=1, skipna=True)
    return lv


def validate(df: pd.DataFrame, lv: pd.DataFrame) -> dict:
    """설정값 타당성 점검.

    ⚠ **본 경보를 '성능저하 예측력'으로 검증하는 것은 이 데이터로는 불가능하다.**
      ① CH4 함량은 **부하와 교란** — 저부하일수록 상승(r(OLR,CH4%)=−0.13,
         OLR 저사분위 68.6% vs 고사분위 65.3%). 온도·부하 경보 검증에 쓸 수 없다.
      ② 비메탄수율은 **분모(VS_in) 교란** — 저투입일 기계적 상승.
      ③ 안정성 경보(pH·VFA/Alk·FAN)의 점검·위험 표본이 n=28·13 으로 과소.
      ④ 무엇보다 과거 데이터에는 **운전자 개입이 이미 반영**되어 있다. 이탈이 생기면
         운전자가 조치했으므로 '경보 후 악화'가 관측되지 않는 것이 정상이다.
    → 따라서 본 시스템은 **성능 예측기가 아니라 상태 감시기(condition monitor)** 이며,
      타당성은 ⓐ 설정값의 공정공학적 근거 ⓑ 합리적 발화율(경보피로 없음)
      ⓒ 실제 off-normal 상태의 정확한 포착 으로 판단한다.
    """
    fire = {n: round(float((lv[n] >= 1).sum() / max(lv[n].notna().sum(), 1) * 100), 1)
            for n in SETPOINTS if lv[n].notna().sum() > 0}
    d = df[["OLR", "CH4_content"]].dropna()
    return {
        "지표별_발화율_%": fire,
        "종합_분포_%": {LEVELS[i]: round(float((lv["종합"] == i).sum()
                                            / lv["종합"].notna().sum() * 100), 1)
                     for i in range(4)},
        "성능검증_불가_사유": {
            "CH4함량": f"부하와 교란 r(OLR,CH4%)={d.OLR.corr(d.CH4_content):+.3f} — 저부하일수록 상승",
            "비메탄수율": "분모(VS_in) 교란 — 저투입일 기계적 상승",
            "표본": "안정성 경보의 점검·위험 표본 n=28·13 로 과소",
            "운전자개입": "과거 데이터에 조치가 이미 반영 — 경보 후 악화가 관측되지 않음"},
        "시스템_성격": "성능 예측기가 아닌 상태 감시기(condition monitor)"}


def chronic_note(df: pd.DataFrame) -> dict:
    """만성 리스크 상시 표시 — 설정값은 시설 기준이나 절대수준은 문헌 대비 높음."""
    fan = df["FAN"].dropna()
    return {"유리암모니아": {
        "중앙값_mg_L": round(float(fan.median()), 0) if len(fan) else None,
        "문헌_저해임계_mg_L": "100 ~ 150",
        "초과율_%": round(float((fan > 150).mean() * 100), 1) if len(fan) else None,
        "추세": "2018 ~250 → 2021 ~700 mg/L 상승",
        "상시경고": "본 시설은 만성 암모니아 저해 상태. 설정값은 '평소 대비 이탈' 기준이며, "
                 "절대수준 자체가 높으므로 C/N 개선·희석·순응 미생물 관리가 상시 필요",
        "계측_권고": "NH4-N 가용률 10.5% — FAN 감시를 위해 정기 계측 확대 필요"}}


# ──────────────────────────────────────────────────────────────────────────────
def plot(df, lv, val, path):
    names = list(SETPOINTS)
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=.5, wspace=.22)
    fig.suptitle("Digester Alarm System — Setpoint Deviation Monitoring",
                 fontsize=15, fontweight="bold", y=.975)
    cmap = ListedColormap([LEVEL_COLOR[i] for i in range(4)])

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(np.ma.masked_invalid(lv[names].T.values.astype(float)), aspect="auto",
              cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest",
              extent=[0, len(lv), len(names), 0])
    ax.set_yticks(np.arange(len(names)) + .5)
    ax.set_yticklabels([f"{NAME_EN[n]} (T{SETPOINTS[n]['tier']})" for n in names], fontsize=8)
    st = max(len(lv) // 12, 1)
    ax.set_xticks(np.arange(0, len(lv), st))
    ax.set_xticklabels([str(d.date())[:7] for d in lv.index[::st]], fontsize=7, rotation=45)
    ax.set_title("A. Setpoint deviation timeline (white = no measurement)",
                 fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    cnt = lv["종합"].value_counts().reindex([0, 1, 2, 3]).fillna(0)
    tot = cnt.sum()
    ax.bar([LEVELS_EN[i] for i in range(4)], cnt.values,
           color=[LEVEL_COLOR[i] for i in range(4)])
    for i, v in enumerate(cnt.values):
        ax.text(i, v, f"{int(v)}\n({v/tot*100:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("days"); ax.set_title("B. Overall alarm distribution", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    fr = pd.Series(val["지표별_발화율_%"]).sort_values()
    ax.barh(range(len(fr)), fr.values,
            color=[LEVEL_COLOR[1] if SETPOINTS[i].get("chronic") else "#4F81BD" for i in fr.index])
    ax.set_yticks(range(len(fr))); ax.set_yticklabels([NAME_EN[i] for i in fr.index], fontsize=8)
    ax.set_xlabel("% of measured days deviating from setpoint")
    ax.set_title("C. Deviation rate by variable", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[2, 0])
    s = df["FAN"].dropna()
    ax.scatter(s.index, s.values, s=12, color="#C0392B", alpha=.7)
    for t, c, l in [(530, "#E8C33C", "watch 530"), (650, "#E2872B", "inspect 650"),
                    (740, "#C0392B", "critical 740"), (150, "#666", "literature 150")]:
        ax.axhline(t, color=c, ls="--", lw=1, label=l)
    ax.set_ylabel("Free ammonia (mg/L)"); ax.legend(fontsize=7)
    ax.set_title("D. Free ammonia — setpoints vs literature limit", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[2, 1])
    tiers = [1, 2, 3]
    bot = np.zeros(3)
    for i in range(1, 4):
        vals = []
        for t in tiers:
            cols = [n for n, sp in SETPOINTS.items() if sp["tier"] == t]
            vals.append(float((lv[cols].max(axis=1, skipna=True) == i).sum()))
        ax.bar([f"Tier {t}\n{['stability','load','support'][t-1]}" for t in tiers],
               vals, bottom=bot, color=LEVEL_COLOR[i], label=LEVELS_EN[i])
        bot += np.array(vals)
    ax.set_ylabel("days with deviation"); ax.legend(fontsize=8)
    ax.set_title("E. Deviation days by tier and severity", fontsize=10, fontweight="bold")

    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    df = build_frame()
    lv = evaluate_history(df)
    val = validate(df, lv)

    print("===== 운전이상 경보 시스템 (설정값 이탈 감시) =====")
    print(f"\n[설정값 표]  경보변수 {len(SETPOINTS)}종")
    print(f"  {'변수':9s}{'구분':10s}{'설정값(정상)':>14s}  {'주의':>9s}{'점검':>9s}{'위험':>9s}  근거")
    for n, sp in SETPOINTS.items():
        b = sp["hi"] or sp["lo"]
        arrow = "≥" if sp["hi"] else "≤"
        print(f"  {n:9s}{TIER_NAME[sp['tier']]:10s}{sp['normal']:>14s}  "
              f"{arrow+str(b[0]):>9s}{arrow+str(b[1]):>9s}{arrow+str(b[2]):>9s}  {sp['basis'][:34]}")

    cnt = lv["종합"].value_counts().reindex([0, 1, 2, 3]).fillna(0); tot = cnt.sum()
    print(f"\n[이력 적용 결과] 판정가능 {int(tot)}일")
    for i in range(4):
        print(f"  {LEVELS[i]:4s} {int(cnt[i]):5d}일 ({cnt[i]/tot*100:5.1f}%)")

    print("\n[변수별 설정값 이탈률]")
    for k, v in sorted(val["지표별_발화율_%"].items(), key=lambda x: -x[1]):
        tag = "만성(시설기준)" if SETPOINTS[k].get("chronic") else "문헌기준"
        print(f"  {k:9s} {v:5.1f}%   [{tag}]")

    print("\n[검증 관점] 본 시스템은 성능 예측기가 아니라 '상태 감시기'")
    for k, v in val["성능검증_불가_사유"].items():
        print(f"  · {k:8s} {v}")
    print("  → 타당성은 ⓐ설정값의 공정공학적 근거 ⓑ합리적 발화율 ⓒoff-normal 포착 으로 판단")

    cn = chronic_note(df)["유리암모니아"]
    print(f"\n[상시 만성 리스크] 유리암모니아 중앙 {cn['중앙값_mg_L']:.0f} mg/L "
          f"(문헌 저해 {cn['문헌_저해임계_mg_L']}, 초과율 {cn['초과율_%']}%)")
    print(f"  {cn['상시경고']}")
    print(f"  ※ {cn['계측_권고']}")

    # 대시보드 호출 예시
    print("\n[대시보드 호출 예시] check_all(현재 계측값)")
    latest = lv[lv["종합"] >= 2].index[-1]
    readings = {n: df.loc[latest, sp["col"]] for n, sp in SETPOINTS.items()}
    print(f"  ── {str(latest.date())} 계측값 입력 시")
    print(format_alarm(check_all(readings)))

    plot(df, lv, val, f"{OUT}/alarm_dashboard.png")
    lv.to_csv(f"{OUT}/alarm_levels.csv")
    with open(f"{OUT}/alarm_setpoints.json", "w", encoding="utf-8") as f:
        json.dump({"setpoints": {n: {k: v for k, v in sp.items() if k != "col"}
                                 for n, sp in SETPOINTS.items()},
                   "excluded": {"비메탄수율": "분모(VS_in) 효과로 저투입일 기계적 상승 — 오경보원",
                                "모델예측잔차": "본 시스템은 계측값-설정값 비교 전용",
                                "개별VFA": "원본 데이터 0% — 계측 개시 권고",
                                "H2S": "변동 없음(σ≈0)"},
                   "distribution": {LEVELS[i]: int(cnt[i]) for i in range(4)},
                   "validation": val, "chronic_risk": chronic_note(df)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n산출: {OUT}/alarm_dashboard.png, alarm_levels.csv, alarm_setpoints.json")


if __name__ == "__main__":
    main()
