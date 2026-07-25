"""
BioGuard-AI 통합 관제 대시보드 — 데이터 생성
================================================================================
세 기능을 하나의 운전자 화면으로 통합한다. 실행: `python -m src.dashboard`.

  ① **메탄생성량 다단계 예측**  현재 이화학 상태 + 투입 계획 → 향후 14일 예측(신뢰구간)
  ② **소화조 상태이상 수준**    계측값의 설정값 이탈 → 4단계 신호등
  ③ **공정제어 지시**          이탈 지표별 운전자 조치 방법

────────────────────────────────────────────────────────────────────────────────
예측 가능 기간 (본 데이터 실측, 2023 홀드아웃 롤링-오리진 154 origin)
────────────────────────────────────────────────────────────────────────────────
| horizon | 1일 | 3일 | 5일 | 7일 | 10일 | 14일 | 21일 | 28일 |
|---------|-----|-----|-----|-----|------|------|------|------|
| 투입계획 반영 R² | 0.951 | 0.863 | 0.769 | 0.753 | 0.719 | 0.690 | 0.674 | 0.647 |
| 투입 미지 R²     | 0.946 | 0.827 | 0.732 | 0.654 | 0.543 | 0.501 | 0.382 | 0.186 |
| persistence R²  | 0.940 | 0.820 | 0.660 | 0.570 | 0.463 | 0.378 | 0.111 | −0.311 |

  · **1~3일** : R² 0.86~0.95. 소화조 자체 동특성(AR)이 지배 — 매우 정확.
  · **4~7일** : R² 0.75~0.80. **일상 운전 판단의 실용 구간**.
  · **8~14일** : R² 0.69~0.75. 투입 계획을 반영해야 유지됨(미지 시 0.50까지 하락)
                → **기질 수급·정비 계획 수립용**.
  · **15~28일** : R² 0.65 (투입계획 반영 시). 투입을 모르면 0.19로 붕괴.
                → 절대값 예측이 아니라 **'이 투입계획이면 이 수준' 시나리오 평가용**.

  ⇒ **대시보드 기본 표시는 14일**. 그 이상은 시나리오 도구로만 제공한다.
  ⇒ persistence 는 7일 이후 급락(0.57→−0.31)하는 반면 모델은 완만히 유지되어,
     **horizon 이 길수록 모델의 상대 우위가 커진다**(28일에서 R² 격차 0.96).

참고 : Choi et al.(2025, Water Research) 의 Transformer 다단계 예측 연구도 1·4·7·10·14일
horizon 을 평가하며, 미래 투입부하를 제공하면 성능이 개선됨을 보고한다. 본 시스템은
투입이 **운전자가 계획하는 제어입력**이라는 점을 활용해 동일 전략을 취한다.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.alarm_system import (LEVELS, SETPOINTS, TIER_NAME, build_frame,
                              check_all, chronic_note)
from src.final_ensemble import OUT, SARIMAX_EXOG, SARIMAX_ORDER, TARGET

warnings.filterwarnings("ignore")

HORIZON = 14          # 대시보드 기본 예측 기간
HISTORY = 45          # 화면에 표시할 과거 일수
HOLDOUT_YEAR = 2023

# 실측된 horizon 별 예측 신뢰도 (src/dashboard.py 상단 표 참조)
HORIZON_SKILL = [
    {"h": 1, "known": 0.951, "unknown": 0.946, "persist": 0.940},
    {"h": 3, "known": 0.863, "unknown": 0.827, "persist": 0.820},
    {"h": 5, "known": 0.769, "unknown": 0.732, "persist": 0.660},
    {"h": 7, "known": 0.753, "unknown": 0.654, "persist": 0.570},
    {"h": 10, "known": 0.719, "unknown": 0.543, "persist": 0.463},
    {"h": 14, "known": 0.690, "unknown": 0.501, "persist": 0.378},
    {"h": 21, "known": 0.674, "unknown": 0.382, "persist": 0.111},
    {"h": 28, "known": 0.647, "unknown": 0.186, "persist": -0.311},
]


def project_pools(df: pd.DataFrame, i: int, plan_feed: float, steps: int) -> pd.DataFrame:
    """투입 계획(plan_feed, ㎥/일)을 미래로 전개해 기질가용성 2-pool 을 재귀 산출.

    S_j(t) = (1−α_j)·S_j(t−1) + α_j·feed(t),  α_j = 1 − e^(−1/τ_j)
    (1차반응 EWMA 의 정확한 재귀식 — 운전자가 계획한 투입에 대한 시나리오 평가)
    """
    from src.bio_lag import TAU_FAST, TAU_SLOW
    a_f, a_s = 1 - np.exp(-1 / TAU_FAST), 1 - np.exp(-1 / TAU_SLOW)
    sf, ss = float(df["S_fast"].iloc[i]), float(df["S_slow"].iloc[i])
    rows = []
    for _ in range(steps):
        sf = (1 - a_f) * sf + a_f * plan_feed
        ss = (1 - a_s) * ss + a_s * plan_feed
        rows.append({"S_fast": sf, "S_slow": ss})
    return pd.DataFrame(rows, index=pd.date_range(df.index[i] + pd.Timedelta(days=1),
                                                  periods=steps))


def build_payload(as_of: str | None = None, plan_feed: float | None = None) -> dict:
    """대시보드 페이로드 생성.

    plan_feed : 향후 투입 계획(㎥/일). 미지정 시 최근 7일 평균 투입량을 계획으로 간주.
      ※ 원자료는 2023-09-18 부터 투입 기록이 0(자료 종료)이라 그대로 쓰면 예측이
        붕괴한다. 실제 운영에서 미래 투입은 **운전자가 계획하는 제어입력**이므로
        계획값을 입력받아 시나리오로 예측하는 것이 올바르다.
    """
    df = build_frame()
    endog = df[TARGET].astype(float)
    exog = df[SARIMAX_EXOG].ffill().bfill()
    cut = int((df.index.year < HOLDOUT_YEAR).sum())

    # 기준일 = 지정일 또는 관측이 있는 마지막 날
    idx = df.index[endog.notna()]
    ref = pd.Timestamp(as_of) if as_of else idx[-1]
    i = df.index.get_loc(ref)

    # 투입 계획 : 미지정이면 최근 7일 평균(정상 운전 지속 시나리오)
    if plan_feed is None:
        recent = df["투입량합계"].iloc[max(i - 6, 0):i + 1].replace(0, np.nan).dropna()
        plan_feed = float(recent.mean()) if len(recent) else float(df["투입량합계"].median())

    # ── ① 다단계 예측 ────────────────────────────────────────────────────────
    res = SARIMAX(endog.iloc[:cut], exog=exog.iloc[:cut], order=SARIMAX_ORDER,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, maxiter=300)
    r = res.apply(endog.iloc[:i + 1], exog=exog.iloc[:i + 1])
    steps = HORIZON
    fut = project_pools(df, i, plan_feed, steps)
    fc = r.get_forecast(steps=steps, exog=fut[SARIMAX_EXOG])
    mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.20)   # 80% 예측구간

    hist = endog.iloc[max(i - HISTORY, 0):i + 1]
    payload = {
        "as_of": str(ref.date()),
        "history": [{"date": str(d.date()), "value": (None if pd.isna(v) else round(float(v), 1))}
                    for d, v in hist.items()],
        # endog 인덱스에 freq 가 없어 예측 인덱스가 정수로 나오므로 날짜를 직접 생성
        "forecast": [{"date": str(d.date()), "mean": round(float(m), 1),
                      "lo": round(float(lo), 1), "hi": round(float(hi), 1),
                      "h": k + 1}
                     for k, (d, m, lo, hi) in enumerate(
                         zip(pd.date_range(ref + pd.Timedelta(days=1), periods=steps),
                             mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values))],
        "horizon_skill": HORIZON_SKILL,
        "horizon_guide": {
            "1-3": "R² 0.86~0.95 — 소화조 자체 동특성 지배, 매우 정확",
            "4-7": "R² 0.75~0.80 — 일상 운전 판단의 실용 구간",
            "8-14": "R² 0.69~0.75 — 투입계획 반영 필요(미지 시 0.50), 기질수급·정비 계획용",
            "15-28": "R² 0.65(투입계획 반영 시) — 시나리오 평가용, 투입 미지 시 0.19로 붕괴",
        },
    }

    # ── ② 상태이상 수준 + ③ 제어 지시 ────────────────────────────────────────
    readings = {}
    for name, sp in SETPOINTS.items():
        v = df[sp["col"]].iloc[i] if sp["col"] in df else np.nan
        if pd.isna(v):                       # 결측이면 직전 관측으로 대체(운전 화면 관행)
            s = df[sp["col"]].iloc[:i + 1].dropna()
            v = s.iloc[-1] if len(s) else np.nan
        readings[name] = None if pd.isna(v) else float(v)
    alarm = check_all(readings)

    payload["alarm"] = {
        "overall": alarm["종합경보"],
        "deviating": alarm["이탈변수수"],
        "items": [{"name": it["변수"], "value": it["현재값"], "unit": it["단위"],
                   "setpoint": it["설정값(정상)"], "level": it["단계"],
                   "lv": it["_lv"], "tier": it["구분"],
                   "action": it["조치"],
                   "basis": SETPOINTS[it["변수"]]["basis"]}
                  for it in alarm["변수상태"]],
    }
    payload["actions"] = [
        {"priority": i2 + 1, "name": it["name"], "level": it["level"],
         "action": it["action"], "tier": it["tier"]}
        for i2, it in enumerate([x for x in payload["alarm"]["items"] if x["lv"] > 0])
    ]
    payload["chronic"] = chronic_note(df)["유리암모니아"]

    # 최근 30일 종합경보 추이 (신호등 히스토리)
    from src.alarm_system import evaluate_history
    lv = evaluate_history(df)
    recent = lv["종합"].iloc[max(i - 29, 0):i + 1]
    payload["alarm_history"] = [
        {"date": str(d.date()), "level": (None if pd.isna(v) else LEVELS[int(v)])}
        for d, v in recent.items()]

    payload["plan_feed"] = round(float(plan_feed), 1)

    # ── 시나리오 계수 : 예측은 미래 투입에 대해 '정확히 선형' 이므로
    #    ŷ(h) = η(h) + β_fast·S_fast(h) + β_slow·S_slow(h) 로 브라우저에서 정확 계산된다.
    #    (검증: 실제 예측 − 선형 재구성 오차 = 0.000000000)
    from src.bio_lag import TAU_FAST, TAU_SLOW, V_DIGESTER
    b_f, b_s = float(res.params["S_fast"]), float(res.params["S_slow"])
    zero = project_pools(df, i, 0.0, steps)
    f0 = r.get_forecast(steps=steps, exog=zero[SARIMAX_EXOG]).predicted_mean.values
    eta = f0 - (b_f * zero["S_fast"].values + b_s * zero["S_slow"].values)
    half = (ci.iloc[:, 1].values - ci.iloc[:, 0].values) / 2      # 투입과 무관
    vs_ratio = float((df["VS_in"] / df["투입량합계"].replace(0, np.nan)).median())
    payload["scenario"] = {
        "eta": [round(float(v), 2) for v in eta],
        "beta_fast": round(b_f, 4), "beta_slow": round(b_s, 4),
        "half": [round(float(v), 1) for v in half],
        "s_fast0": round(float(df["S_fast"].iloc[i]), 4),
        "s_slow0": round(float(df["S_slow"].iloc[i]), 4),
        "alpha_fast": round(float(1 - np.exp(-1 / TAU_FAST)), 6),
        "alpha_slow": round(float(1 - np.exp(-1 / TAU_SLOW)), 6),
        "feed_max": 260, "v_digester": V_DIGESTER,
        "vs_per_m3": round(vs_ratio, 2),      # VS_in ≈ feed × 이 값 → OLR 산출
        "olr_bands": [1.8, 2.5, 3.5], "hrt_bands": [30, 25, 20],
    }
    payload["model"] = {
        "name": "Stacking(SARIMAX + XGBoost + RandomForest), 비음수 Ridge 메타",
        "holdout_R2": 0.9024, "holdout_RMSE": 341.7, "holdout_MAE": 244.5,
        "cv_R2": "0.882 ± 0.027 (롤링-오리진 8폴드)",
        "baseline_R2": 0.877,
        "bio": "CSTR 1차반응 2-pool 분포지연 (τ_fast=1d, τ_slow=8d, k_h=0.100/d, HRT=40.6d)",
    }
    return payload


def main():
    os.makedirs(OUT, exist_ok=True)
    p = build_payload()
    with open(f"{OUT}/dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

    print("===== 통합 대시보드 데이터 생성 =====")
    print(f"  기준일 : {p['as_of']}")
    print(f"  투입계획: {p['plan_feed']} ㎥/일 (최근 7일 평균)")
    print(f"  예측    : 향후 {len(p['forecast'])}일 "
          f"(첫날 {p['forecast'][0]['mean']:.0f} → 마지막 {p['forecast'][-1]['mean']:.0f} Nm³/d)")
    print(f"  종합경보: {p['alarm']['overall']} (이탈 {p['alarm']['deviating']}건)")
    for a in p["actions"]:
        print(f"    {a['priority']}. [{a['level']}] {a['name']} → {a['action']}")
    print(f"\n  예측 신뢰도(R²): " + "  ".join(
        f"{s['h']}일={s['known']:.2f}" for s in p["horizon_skill"]))
    with open(f"{OUT}/dashboard.html", "w", encoding="utf-8") as f:
        f.write(render_html(p))
    print(f"\n  산출: {OUT}/dashboard_data.json, {OUT}/dashboard.html")



# ──────────────────────────────────────────────────────────────────────────────
# 대시보드 HTML 렌더링 (자기완결형 — 외부 리소스 없음)
# ──────────────────────────────────────────────────────────────────────────────
def fmt(v: float) -> str:
    """계측값 표시 포맷 — 자리수를 값의 크기에 맞춘다(계기 판독 관행)."""
    a = abs(v)
    if a >= 1000: return f"{v:,.0f}"
    if a >= 100:  return f"{v:,.1f}"
    if a >= 10:   return f"{v:,.1f}"
    if a >= 1:    return f"{v:,.2f}"
    return f"{v:,.3f}"


SEV = {"정상": ("ok", "#3E9E5B"), "주의": ("watch", "#E0B93C"),
       "점검": ("inspect", "#E07B39"), "위험": ("crit", "#CC3D33"),
       "판정불가": ("na", "#7E9296")}




# ──────────────────────────────────────────────────────────────────────────────
# 대시보드 HTML 렌더링
#   · 디자인 : 단일 컬럼 앱형 + 토스 계열 토큰(면으로 구분, 숫자가 주인공, 액센트 1색)
#   · 시나리오 : 예측이 미래 투입에 대해 정확히 선형이므로 브라우저에서 정확 계산
#                ŷ(h) = η(h) + β_fast·S_fast(h) + β_slow·S_slow(h)
# ──────────────────────────────────────────────────────────────────────────────
SEV_KEY = {"정상": "ok", "주의": "watch", "점검": "inspect", "위험": "crit", "판정불가": "na"}


def _josa(word: str, batchim: str, no_batchim: str) -> str:
    """한국어 조사 선택. 한글은 받침 유무로, 영문 약어는 실제 발음 기준."""
    LATIN = {"pH": False, "FAN": True, "TAN": True, "OLR": True,
             "HRT": False, "VFA": False, "VFA/Alk": False}   # True = 받침 있음
    if word in LATIN:
        return batchim if LATIN[word] else no_batchim
    ch = word[-1]
    if "가" <= ch <= "힣":
        return batchim if (ord(ch) - 0xAC00) % 28 else no_batchim
    return no_batchim


def _headline(p: dict) -> tuple[str, str]:
    """상태를 한 문장으로. (시스템 용어 대신 사람 말로)"""
    devs = [i for i in p["alarm"]["items"] if i["lv"] > 0]
    if not devs:
        return "모든 지표가 정상 범위입니다", "특별한 조치 없이 현재 운전을 유지하세요."
    top = devs[0]
    n, lv = top["name"], top["level"]
    subj = f"{n} 외 {len(devs)-1}건이" if len(devs) > 1 else f"{n}{_josa(n, '이', '가')}"
    tail = {"주의": "기준에 가까워졌습니다", "점검": "기준을 벗어났습니다",
            "위험": "위험 수준입니다"}[lv]
    return f"{subj} {tail}", top["action"]


def render_html(p: dict) -> str:
    sc = p["scenario"]
    head, sub = _headline(p)
    ovk = SEV_KEY[p["alarm"]["overall"]]

    rows = "".join(
        f'''<div class="row">
      <div class="rowL"><span class="rn">{it['name']}</span>
        <span class="rs">기준 {it['setpoint']}</span></div>
      <div class="rowR"><span class="rv">{fmt(it['value'])}<em>{it['unit']}</em></span>
        <span class="tag t-{SEV_KEY[it['level']]}">{it['level']}</span></div>
    </div>''' for it in p["alarm"]["items"])

    acts = "".join(
        f'''<li class="act">
      <span class="tag t-{SEV_KEY[a['level']]}">{a['level']}</span>
      <div class="actB"><b>{a['name']}</b><p>{a['action']}</p></div>
    </li>''' for a in p["actions"]) or \
        '<li class="act"><span class="tag t-ok">정상</span><div class="actB">' \
        '<b>조치 사항 없음</b><p>모든 계측값이 기준 이내입니다.</p></div></li>'

    skill = "".join(
        f'''<div class="srow"><span class="sh">{s['h']}일 뒤</span>
      <div class="sbar"><i style="width:{max(s['known'],0)*100:.0f}%"></i></div>
      <span class="sv">{s['known']:.2f}</span></div>''' for s in p["horizon_skill"])

    strip = "".join(
        f'<i class="hb h-{SEV_KEY.get(h["level"], "na")}"></i>' for h in p["alarm_history"])

    m, ch = p["model"], p["chronic"]
    data = json.dumps({
        "history": [h for h in p["history"]],
        "asOf": p["as_of"], "plan0": p["plan_feed"], **sc,
    }, ensure_ascii=False)

    return TEMPLATE.replace("__DATA__", data) \
        .replace("__HEAD__", head).replace("__SUB__", sub).replace("__OVK__", ovk) \
        .replace("__OVERALL__", p["alarm"]["overall"]) \
        .replace("__ASOF__", p["as_of"]) \
        .replace("__DEV__", str(p["alarm"]["deviating"])) \
        .replace("__TOT__", str(len(p["alarm"]["items"]))) \
        .replace("__ROWS__", rows).replace("__ACTS__", acts) \
        .replace("__SKILL__", skill).replace("__STRIP__", strip) \
        .replace("__FANMED__", f"{ch['중앙값_mg_L']:,.0f}") \
        .replace("__FANLIM__", ch["문헌_저해임계_mg_L"]) \
        .replace("__FANEX__", str(ch["초과율_%"])) \
        .replace("__FANTREND__", ch["추세"]) \
        .replace("__FANNOTE__", ch["상시경고"]) \
        .replace("__FANREC__", ch["계측_권고"]) \
        .replace("__MODEL__", m["name"]) \
        .replace("__R2__", str(m["holdout_R2"])).replace("__RMSE__", str(m["holdout_RMSE"])) \
        .replace("__CV__", m["cv_R2"]).replace("__BIO__", m["bio"])


TEMPLATE = r"""<title>BioGuard-AI 소화조 관제</title>
<style>
:root{
  --bg:#FFFFFF; --surface:#F2F4F6; --surface2:#E8EBEE; --line:#EDF0F2;
  --t1:#191F28; --t2:#6B7684; --t3:#AEB5BD;
  --blue:#3182F6; --blueSoft:#E8F1FE;
  --ok:#00B26B; --watch:#F5A623; --inspect:#FF7A00; --crit:#F04452;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#17171C; --surface:#202127; --surface2:#2A2C33; --line:#2A2C33;
  --t1:#EDEFF2; --t2:#9AA3AD; --t3:#6B7280;
  --blue:#5A9CF8; --blueSoft:#1E2A3D;
  --ok:#26C281; --watch:#F7B955; --inspect:#FF8F2E; --crit:#FF6B6B;
}}
:root[data-theme="dark"]{
  --bg:#17171C; --surface:#202127; --surface2:#2A2C33; --line:#2A2C33;
  --t1:#EDEFF2; --t2:#9AA3AD; --t3:#6B7280;
  --blue:#5A9CF8; --blueSoft:#1E2A3D;
  --ok:#26C281; --watch:#F7B955; --inspect:#FF8F2E; --crit:#FF6B6B;
}
:root[data-theme="light"]{
  --bg:#FFFFFF; --surface:#F2F4F6; --surface2:#E8EBEE; --line:#EDF0F2;
  --t1:#191F28; --t2:#6B7684; --t3:#AEB5BD;
  --blue:#3182F6; --blueSoft:#E8F1FE;
  --ok:#00B26B; --watch:#F5A623; --inspect:#FF7A00; --crit:#F04452;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--t1);-webkit-font-smoothing:antialiased;
  font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",
    "Malgun Gothic",system-ui,sans-serif;font-size:15px;line-height:1.5}
.app{max-width:720px;margin:0 auto;padding:32px 20px 64px;display:flex;flex-direction:column;gap:12px}
.num{font-variant-numeric:tabular-nums;letter-spacing:-.02em}

/* 헤더 */
.top{padding:4px 4px 12px}
.top .date{font-size:13px;color:var(--t2);font-variant-numeric:tabular-nums}
.top h1{font-size:15px;font-weight:600;color:var(--t2);margin:2px 0 0}

/* 히어로 */
.hero{background:var(--surface);border-radius:20px;padding:24px}
.dotline{display:flex;align-items:center;gap:8px;margin-bottom:14px}
.dot{width:8px;height:8px;border-radius:50%}
.d-ok{background:var(--ok)} .d-watch{background:var(--watch)}
.d-inspect{background:var(--inspect)} .d-crit{background:var(--crit)} .d-na{background:var(--t3)}
.dotline span{font-size:13px;font-weight:600}
.s-ok{color:var(--ok)} .s-watch{color:var(--watch)}
.s-inspect{color:var(--inspect)} .s-crit{color:var(--crit)} .s-na{color:var(--t3)}
.hero h2{font-size:22px;font-weight:700;margin:0;letter-spacing:-.02em;text-wrap:balance;line-height:1.35}
.hero p{margin:8px 0 0;font-size:14px;color:var(--t2)}
.heroFoot{display:flex;gap:24px;margin-top:20px;padding-top:18px;border-top:1px solid var(--line)}
.mini span{display:block;font-size:12px;color:var(--t2);margin-bottom:3px}
.mini b{font-size:17px;font-weight:700}

/* 카드 */
.card{background:var(--surface);border-radius:20px;padding:22px}
.card>h3{font-size:17px;font-weight:700;margin:0 0 4px;letter-spacing:-.01em}
.card>.cap{font-size:13px;color:var(--t2);margin:0 0 18px}

/* 예측 */
.big{display:flex;align-items:baseline;gap:8px;margin:2px 0 16px}
.big b{font-size:38px;font-weight:700;letter-spacing:-.03em}
.big em{font-style:normal;font-size:15px;color:var(--t2);font-weight:500}
.delta{font-size:14px;font-weight:600;padding:3px 9px;border-radius:20px;
  background:var(--blueSoft);color:var(--blue)}
.delta.down{background:rgba(240,68,82,.12);color:var(--crit)}
.chartBox{position:relative}
svg.ch{width:100%;height:auto;display:block;overflow:visible}
.gl{stroke:var(--line);stroke-width:1}
.bandP{fill:var(--blue);opacity:.10}
.lineH{fill:none;stroke:var(--t3);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.lineF{fill:none;stroke:var(--blue);stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round}
.tick{fill:var(--t3);font-size:11px;font-variant-numeric:tabular-nums}
.tickR{text-anchor:end} .tickC{text-anchor:middle}
.nowl{stroke:var(--t3);stroke-width:1;stroke-dasharray:2 4}
.endD{fill:var(--blue)}
.endHalo{fill:var(--blue);opacity:.18}

/* 슬라이더 */
.sliderCard{margin-top:18px;padding-top:18px;border-top:1px solid var(--line)}
.slHead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px}
.slHead span{font-size:14px;color:var(--t2);font-weight:500}
.slHead b{font-size:20px;font-weight:700}
.slHead b em{font-style:normal;font-size:13px;color:var(--t2);font-weight:500;margin-left:3px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:34px;background:none;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:99px;background:var(--surface2)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:26px;height:26px;border-radius:50%;
  background:#fff;border:none;box-shadow:0 1px 4px rgba(0,0,0,.22),0 0 0 1px rgba(0,0,0,.04);margin-top:-10px}
input[type=range]::-moz-range-track{height:6px;border-radius:99px;background:var(--surface2)}
input[type=range]::-moz-range-thumb{width:26px;height:26px;border-radius:50%;background:#fff;border:none;
  box-shadow:0 1px 4px rgba(0,0,0,.22)}
input[type=range]:focus-visible{outline:2px solid var(--blue);outline-offset:4px;border-radius:8px}
.slScale{display:flex;justify-content:space-between;font-size:12px;color:var(--t3);
  font-variant-numeric:tabular-nums;margin-top:-4px}
.chips{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.chip{border:none;background:var(--surface2);color:var(--t1);font:inherit;font-size:13px;font-weight:600;
  padding:8px 14px;border-radius:99px;cursor:pointer}
.chip:hover{background:var(--blueSoft);color:var(--blue)}
.chip[aria-pressed="true"]{background:var(--blue);color:#fff}
.safety{display:flex;gap:10px;margin-top:16px}
.sfx{flex:1;background:var(--bg);border-radius:14px;padding:13px 15px}
.sfx span{display:block;font-size:12px;color:var(--t2);margin-bottom:4px}
.sfx b{font-size:17px;font-weight:700}
.sfx small{display:block;font-size:12px;margin-top:3px;font-weight:600}

/* 리스트 */
.row{display:flex;justify-content:space-between;align-items:center;gap:14px;
  padding:13px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:none}
.rowL{display:flex;flex-direction:column;gap:2px;min-width:0}
.rn{font-size:15px;font-weight:600}
.rs{font-size:12px;color:var(--t3);font-variant-numeric:tabular-nums}
.rowR{display:flex;align-items:center;gap:10px;white-space:nowrap}
.rv{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums}
.rv em{font-style:normal;font-size:12px;color:var(--t2);font-weight:500;margin-left:2px}
.tag{font-size:12px;font-weight:700;padding:3px 9px;border-radius:8px;min-width:42px;text-align:center}
.t-ok{background:rgba(0,178,107,.12);color:var(--ok)}
.t-watch{background:rgba(245,166,35,.14);color:var(--watch)}
.t-inspect{background:rgba(255,122,0,.14);color:var(--inspect)}
.t-crit{background:rgba(240,68,82,.13);color:var(--crit)}
.t-na{background:var(--surface2);color:var(--t3)}

/* 신호 스트립 */
.strip{display:flex;gap:3px;margin-top:6px}
.hb{flex:1;height:26px;border-radius:4px;background:var(--surface2)}
.h-ok{background:var(--ok)} .h-watch{background:var(--watch)}
.h-inspect{background:var(--inspect)} .h-crit{background:var(--crit)}
.h-na{background:var(--surface2)}

/* 조치 */
ol.acts{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px}
.act{display:flex;gap:12px;align-items:flex-start;padding:14px 0;border-bottom:1px solid var(--line)}
.act:last-child{border-bottom:none}
.actB b{font-size:15px;font-weight:600;display:block}
.actB p{margin:3px 0 0;font-size:14px;color:var(--t2)}

/* 신뢰도 */
.srow{display:flex;align-items:center;gap:12px;padding:7px 0}
.sh{font-size:13px;color:var(--t2);width:56px;flex-shrink:0;font-variant-numeric:tabular-nums}
.sbar{flex:1;height:8px;border-radius:99px;background:var(--surface2);overflow:hidden}
.sbar i{display:block;height:100%;border-radius:99px;background:var(--blue)}
.sv{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;width:38px;text-align:right}
.legendTxt{font-size:13px;color:var(--t2);margin:16px 0 0;line-height:1.65}
.legendTxt b{color:var(--t1)}

/* 만성 */
.warn{background:rgba(240,68,82,.07);border-radius:16px;padding:16px 18px}
.warn b{font-size:15px;display:block;margin-bottom:6px}
.warn p{margin:6px 0 0;font-size:13.5px;color:var(--t2);line-height:1.6}
.foot{font-size:12px;color:var(--t3);line-height:1.8;padding:8px 4px 0}
@media (max-width:520px){
  .heroFoot{gap:18px}.big b{font-size:32px}.safety{flex-direction:column}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="app">
  <div class="top">
    <p class="date">__ASOF__</p>
    <h1>영천 통합바이오가스화시설 · 혐기성 소화조</h1>
  </div>

  <section class="hero">
    <div class="dotline"><i class="dot d-__OVK__"></i><span class="s-__OVK__">__OVERALL__</span></div>
    <h2>__HEAD__</h2>
    <p>__SUB__</p>
    <div class="heroFoot">
      <div class="mini"><span>기준 벗어남</span><b class="num">__DEV__<em style="font-style:normal;color:var(--t3);font-weight:500;font-size:14px">/__TOT__</em></b></div>
      <div class="mini"><span>내일 예상 메탄</span><b class="num" id="d1">–</b></div>
      <div class="mini"><span>14일 뒤</span><b class="num" id="d14">–</b></div>
    </div>
  </section>

  <section class="card">
    <h3>메탄생성량 예측</h3>
    <p class="cap">투입 계획을 바꾸면 예측이 함께 움직입니다</p>
    <div class="big"><b class="num" id="bigVal">–</b><em>Nm³/일 · 14일 뒤</em>
      <span class="delta" id="delta"></span></div>
    <div class="chartBox"><svg class="ch" id="chart" viewBox="0 0 700 230" role="img"
      aria-label="과거 실측과 향후 14일 메탄생성량 예측"></svg></div>

    <div class="sliderCard">
      <div class="slHead"><span>투입 계획</span><b class="num" id="feedVal">–<em>㎥/일</em></b></div>
      <input type="range" id="feed" min="0" max="260" step="5" aria-label="투입 계획 (㎥/일)">
      <div class="slScale"><span>0</span><span>130</span><span>260</span></div>
      <div class="chips">
        <button class="chip" data-f="0">투입 중단</button>
        <button class="chip" data-f="-20">20% 감량</button>
        <button class="chip" data-f="cur" aria-pressed="true">현재 계획</button>
        <button class="chip" data-f="+20">20% 증량</button>
        <button class="chip" data-f="209">설계 최대</button>
      </div>
      <div class="safety">
        <div class="sfx"><span>유기물부하율 OLR</span><b class="num" id="olr">–</b>
          <small id="olrS"></small></div>
        <div class="sfx"><span>체류시간 HRT</span><b class="num" id="hrt">–</b>
          <small id="hrtS"></small></div>
      </div>
    </div>
  </section>

  <section class="card">
    <h3>지금 소화조 상태</h3>
    <p class="cap">계측값이 기준을 벗어나면 표시됩니다</p>
    __ROWS__
    <h3 style="margin-top:22px;font-size:15px">최근 30일</h3>
    <div class="strip">__STRIP__</div>
  </section>

  <section class="card">
    <h3>지금 해야 할 일</h3>
    <p class="cap">급한 것부터 정렬했습니다</p>
    <ol class="acts">__ACTS__</ol>
  </section>

  <section class="card">
    <h3>며칠 앞까지 믿을 수 있나요</h3>
    <p class="cap">2023년 실측 검증 · 투입 계획을 알 때의 정확도(R²)</p>
    __SKILL__
    <p class="legendTxt"><b>1~3일</b> 소화조 자체 흐름이 이어져 매우 정확합니다.
      <b>4~7일</b> 일상 운전 판단에 쓰기 좋은 구간입니다.
      <b>8~14일</b> 투입 계획을 넣어야 이 정확도가 유지됩니다(모르면 0.50).
      <b>15일 이상</b>은 "이렇게 투입하면 이 정도" 시나리오로만 보세요.</p>
  </section>

  <section class="card">
    <h3>계속 지켜볼 위험</h3>
    <div class="warn">
      <b>유리암모니아(FAN) 만성 저해</b>
      <p>중앙 <b style="display:inline" class="num">__FANMED__ mg/L</b> — 문헌 저해 기준 __FANLIM__ mg/L 를
        측정일의 <b style="display:inline">__FANEX__%</b> 가 넘습니다. 추세는 __FANTREND__.</p>
      <p>__FANNOTE__</p>
      <p>※ __FANREC__</p>
    </div>
  </section>

  <p class="foot">__MODEL__<br>
    2023 홀드아웃 R² __R2__ · RMSE __RMSE__ &nbsp;|&nbsp; 교차검증 R² __CV__<br>
    지연모델 · __BIO__</p>
</div>

<script>
const D = __DATA__;
const $ = s => document.querySelector(s);

/* 예측은 미래 투입에 대해 정확히 선형 : y(h) = eta(h) + bF*Sf(h) + bS*Ss(h) */
function forecast(feed){
  let sf = D.s_fast0, ss = D.s_slow0, out = [];
  for(let h=0; h<D.eta.length; h++){
    sf = (1-D.alpha_fast)*sf + D.alpha_fast*feed;
    ss = (1-D.alpha_slow)*ss + D.alpha_slow*feed;
    const m = D.eta[h] + D.beta_fast*sf + D.beta_slow*ss;
    out.push({m:m, lo:m-D.half[h], hi:m+D.half[h]});
  }
  return out;
}
const nf = n => Math.round(n).toLocaleString('ko-KR');

function draw(fc){
  const W=700,H=230,L=46,R=14,T=12,B=26;
  const hist = D.history.filter(d=>d.value!==null);
  const nH = D.history.length, nF = fc.length, N = nH+nF;
  let vals = hist.map(d=>d.value).concat(fc.map(f=>f.lo), fc.map(f=>f.hi));
  let lo=Math.min(...vals), hi=Math.max(...vals);
  const pad=(hi-lo)*0.14||100; lo-=pad; hi+=pad;
  const X=k=>L+(W-L-R)*k/(N-1), Y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const hidx={}; D.history.forEach((d,k)=>hidx[d.date]=k);
  const pt=(x,y)=>x.toFixed(1)+","+y.toFixed(1);
  const hp = hist.map(d=>pt(X(hidx[d.date]),Y(d.value)));
  const fp = fc.map((f,k)=>pt(X(nH+k),Y(f.m)));
  const band = fc.map((f,k)=>pt(X(nH+k),Y(f.hi)))
    .concat(fc.map((f,k)=>pt(X(nH+nF-1-k),Y(fc[nF-1-k].lo))));
  let g="";
  for(let k=0;k<4;k++){
    const v=lo+(hi-lo)*k/3, y=Y(v);
    g+=`<line x1="${L}" y1="${y.toFixed(1)}" x2="${W-R}" y2="${y.toFixed(1)}" class="gl"/>`
      +`<text x="${L-8}" y="${(y+4).toFixed(1)}" class="tick tickR">${nf(v)}</text>`;
  }
  let tk="", step=Math.max(Math.floor(N/5),1), marks=[];
  for(let k=0;k<N;k+=step) marks.push(k);
  if(marks[marks.length-1] !== N-1) marks.push(N-1);   /* 마지막 예측일 눈금 보장 */
  marks.forEach(k=>{
    const d = k<nH ? D.history[k].date : D.forecastDates[k-nH];
    if(d) tk+=`<text x="${X(k).toFixed(1)}" y="${H-8}" class="tick tickC">${d.slice(5)}</text>`;
  });
  const ex=X(N-1), ey=Y(fc[nF-1].m);
  $("#chart").innerHTML = g
    + `<polygon points="${band.join(' ')}" class="bandP"/>`
    + `<polyline points="${hp.join(' ')}" class="lineH"/>`
    + `<polyline points="${[hp[hp.length-1]].concat(fp).join(' ')}" class="lineF"/>`
    + `<line x1="${X(nH-0.5).toFixed(1)}" y1="${T}" x2="${X(nH-0.5).toFixed(1)}" y2="${H-B}" class="nowl"/>`
    + `<circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="9" class="endHalo"/>`
    + `<circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="4.5" class="endD"/>`
    + tk;
}

function band3(v, b, lowerIsWorse){
  /* 설정값 밴드 → [라벨, 색토큰] */
  const over = lowerIsWorse ? (v<=b[0]) : (v>=b[0]);
  if(!over) return ["정상","ok"];
  const s = lowerIsWorse ? (v<=b[2]?2:(v<=b[1]?1:0)) : (v>=b[2]?2:(v>=b[1]?1:0));
  return [["주의","점검","위험"][s], ["watch","inspect","crit"][s]];
}

function update(feed){
  const fc = forecast(feed);
  const base = forecast(D.plan0);
  draw(fc);
  $("#feedVal").innerHTML = nf(feed)+'<em>㎥/일</em>';
  $("#bigVal").textContent = nf(fc[fc.length-1].m);
  $("#d1").textContent  = nf(fc[0].m);
  $("#d14").textContent = nf(fc[fc.length-1].m);
  const df = fc[fc.length-1].m - base[base.length-1].m;
  const el = $("#delta");
  if(Math.abs(df) < 1){ el.textContent = "현재 계획"; el.className="delta"; }
  else { el.textContent = (df>0?"+":"−")+nf(Math.abs(df))+" Nm³/일";
         el.className = "delta"+(df<0?" down":""); }

  const olr = feed*D.vs_per_m3/D.v_digester;
  const hrt = feed>0 ? D.v_digester/feed : Infinity;
  const [ol,oc] = band3(olr, D.olr_bands, false);
  const [hl,hc] = band3(hrt, D.hrt_bands, true);
  $("#olr").textContent = olr.toFixed(2);
  $("#olrS").textContent = ol+" · 기준 < "+D.olr_bands[0];
  $("#olrS").style.color = "var(--"+oc+")";
  $("#hrt").textContent = isFinite(hrt) ? hrt.toFixed(0)+"일" : "—";
  $("#hrtS").textContent = (isFinite(hrt)?hl:"투입 없음")+" · 기준 > "+D.hrt_bands[0]+"일";
  $("#hrtS").style.color = "var(--"+(isFinite(hrt)?hc:"t3")+")";
}

/* 예측 날짜 라벨 */
(function(){
  const d0 = new Date(D.asOf+"T00:00:00");
  D.forecastDates = D.eta.map((_,k)=>{
    const d = new Date(d0); d.setDate(d.getDate()+k+1);
    return d.toISOString().slice(0,10);
  });
})();

const slider = $("#feed");
slider.value = D.plan0;
update(D.plan0);
slider.addEventListener("input", e=>{
  update(+e.target.value);
  document.querySelectorAll(".chip").forEach(c=>c.setAttribute("aria-pressed","false"));
});
document.querySelectorAll(".chip").forEach(c=>{
  c.addEventListener("click", ()=>{
    const f=c.dataset.f;
    let v = f==="cur" ? D.plan0 : f==="-20" ? D.plan0*0.8 : f==="+20" ? D.plan0*1.2 : +f;
    v = Math.max(0, Math.min(D.feed_max, Math.round(v/5)*5));
    slider.value=v; update(v);
    document.querySelectorAll(".chip").forEach(x=>x.setAttribute("aria-pressed","false"));
    c.setAttribute("aria-pressed","true");
  });
});
</script>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    p = build_payload()
    with open(f"{OUT}/dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
    with open(f"{OUT}/dashboard.html", "w", encoding="utf-8") as f:
        f.write(render_html(p))

    print("===== 통합 관제 대시보드 =====")
    print(f"  기준일 {p['as_of']} · 투입계획 {p['plan_feed']:,.0f} ㎥/일")
    print(f"  예측   익일 {p['forecast'][0]['mean']:,.0f} → 14일 {p['forecast'][-1]['mean']:,.0f} Nm³/일")
    print(f"  경보   {p['alarm']['overall']} (이탈 {p['alarm']['deviating']}/{len(p['alarm']['items'])})")
    for a in p["actions"]:
        print(f"    {a['priority']}. [{a['level']}] {a['name']} → {a['action']}")
    sc = p["scenario"]
    print(f"  시나리오 계수 beta_fast={sc['beta_fast']} beta_slow={sc['beta_slow']} "
          f"(브라우저에서 정확 계산)")
    print(f"\n  산출: {OUT}/dashboard.html, dashboard_data.json")


if __name__ == "__main__":
    main()
