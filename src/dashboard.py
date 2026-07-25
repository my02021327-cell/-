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


def _chart_svg(p: dict) -> str:
    """과거 실측 + 예측(80% 구간) + horizon 신뢰구역을 그린 SVG."""
    W, H = 780, 280
    L, R, T, B = 52, 14, 16, 34
    hist = [h for h in p["history"] if h["value"] is not None]
    fc = p["forecast"]
    n_h, n_f = len(p["history"]), len(fc)
    total = n_h + n_f
    vals = [h["value"] for h in hist] + [f["lo"] for f in fc] + [f["hi"] for f in fc]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12 or 100
    lo, hi = lo - pad, hi + pad

    def X(k): return L + (W - L - R) * k / (total - 1)
    def Y(v): return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    hidx = {h["date"]: i for i, h in enumerate(p["history"])}
    hpts = [(X(hidx[h["date"]]), Y(h["value"])) for h in hist]
    fpts = [(X(n_h + k), Y(f["mean"])) for k, f in enumerate(fc)]
    band = ([(X(n_h + k), Y(f["hi"])) for k, f in enumerate(fc)]
            + [(X(n_h + k), Y(f["lo"])) for k, f in reversed(list(enumerate(fc)))])

    def path(pts): return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # horizon 신뢰구역 (1-3 / 4-7 / 8-14일)
    zones, zdef = [], [(0, 3, .16, "1–3일"), (3, 7, .10, "4–7일"), (7, 14, .05, "8–14일")]
    for a, b, op, lab in zdef:
        if a >= n_f: break
        b = min(b, n_f)
        x0, x1 = X(n_h + a - .5 if a else n_h - .5), X(n_h + b - .5)
        zones.append(f'<rect x="{x0:.1f}" y="{T}" width="{x1-x0:.1f}" height="{H-T-B}" '
                     f'fill="var(--accent)" opacity="{op}"/>')
        zones.append(f'<text x="{(x0+x1)/2:.1f}" y="{T+13}" class="zlab">{lab}</text>')

    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        grid.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" class="grid"/>')
        grid.append(f'<text x="{L-8}" y="{y+3.5:.1f}" class="ylab">{v:,.0f}</text>')

    ticks = []
    for k in range(0, total, max(total // 7, 1)):
        d = (p["history"][k]["date"] if k < n_h else fc[k - n_h]["date"])[5:]
        ticks.append(f'<text x="{X(k):.1f}" y="{H-14}" class="xlab">{d}</text>')

    ex, ey = fpts[-1]
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img"
   aria-label="메탄생성량 과거 실측과 향후 {n_f}일 예측">
  {''.join(zones)}{''.join(grid)}
  <polygon points="{' '.join(f'{x:.1f},{y:.1f}' for x,y in band)}" class="band"/>
  <path d="{path(hpts)}" class="hist"/>
  <path d="{path([hpts[-1]] + fpts)}" class="fcast"/>
  <line x1="{X(n_h-.5):.1f}" y1="{T}" x2="{X(n_h-.5):.1f}" y2="{H-B}" class="now"/>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="4.5" class="endpt"/>
  <text x="{ex-6:.1f}" y="{ey-11:.1f}" class="endlab">{fc[-1]['mean']:,.0f}</text>
  {''.join(ticks)}
</svg>'''


def render_html(p: dict) -> str:
    ov = p["alarm"]["overall"]
    ocls, ocol = SEV[ov]
    rows = "".join(
        f'''<tr class="sev-{SEV[it['level']][0]}">
      <td class="vname">{it['name']}<span class="tier">{it['tier']}</span></td>
      <td class="num">{fmt(it['value'])}<span class="unit">{it['unit']}</span></td>
      <td class="sp">{it['setpoint']}</td>
      <td><span class="chip chip-{SEV[it['level']][0]}">{it['level']}</span></td>
    </tr>''' for it in p["alarm"]["items"])

    acts = "".join(
        f'''<li class="act sev-{SEV[a['level']][0]}">
      <span class="chip chip-{SEV[a['level']][0]}">{a['level']}</span>
      <div><b>{a['name']}</b> <span class="muted">· {a['tier']}</span>
      <p>{a['action']}</p></div></li>''' for a in p["actions"]) or \
        '<li class="act sev-ok"><span class="chip chip-ok">정상</span><div>모든 계측값이 설정값 이내입니다.<p>정규 운전을 유지하십시오.</p></div></li>'

    skill = "".join(
        f'''<tr><td class="num">{s['h']}</td><td class="num strong">{s['known']:.3f}</td>
      <td class="num">{s['unknown']:.3f}</td><td class="num dim">{s['persist']:.3f}</td></tr>'''
        for s in p["horizon_skill"])

    hist_strip = "".join(
        f'<i class="hb sev-{SEV.get(h["level"],SEV["판정불가"])[0]}" title="{h["date"]} {h["level"] or "무측정"}"></i>'
        for h in p["alarm_history"])

    m = p["model"]
    return f'''<title>BioGuard-AI 통합 관제</title>
<style>
:root{{
  --ground:#F2F5F5; --panel:#FFFFFF; --panel2:#E9EEEE; --ink:#121B1D; --ink2:#3D4C50;
  --muted:#5A6C70; --line:#D3DCDC; --accent:#1D8FAD;
  --ok:#3E9E5B; --watch:#C9A21F; --inspect:#D2701F; --crit:#C0342A;
}}
@media (prefers-color-scheme:dark){{:root{{
  --ground:#0D1416; --panel:#141F22; --panel2:#1B282C; --ink:#E4EDEE; --ink2:#B6C7CA;
  --muted:#7E9296; --line:#25353A; --accent:#3FB8D4;
  --ok:#4BB369; --watch:#E0B93C; --inspect:#E07B39; --crit:#DE5147;
}}}}
:root[data-theme="dark"]{{
  --ground:#0D1416; --panel:#141F22; --panel2:#1B282C; --ink:#E4EDEE; --ink2:#B6C7CA;
  --muted:#7E9296; --line:#25353A; --accent:#3FB8D4;
  --ok:#4BB369; --watch:#E0B93C; --inspect:#E07B39; --crit:#DE5147;
}}
:root[data-theme="light"]{{
  --ground:#F2F5F5; --panel:#FFFFFF; --panel2:#E9EEEE; --ink:#121B1D; --ink2:#3D4C50;
  --muted:#5A6C70; --line:#D3DCDC; --accent:#1D8FAD;
  --ok:#3E9E5B; --watch:#C9A21F; --inspect:#D2701F; --crit:#C0342A;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 20px 56px;display:flex;flex-direction:column;gap:18px}}
.eyebrow{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0}}
h1{{font-size:25px;letter-spacing:-.02em;margin:2px 0 0;text-wrap:balance}}
h2{{font-size:14px;letter-spacing:.04em;margin:0 0 12px;color:var(--ink2)}}
.num,.mono{{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}}
.muted{{color:var(--muted)}} .dim{{color:var(--muted)}} .strong{{font-weight:700}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}}

/* 상태 바 */
.status{{display:flex;flex-wrap:wrap;align-items:center;gap:22px;
  border-left:5px solid {ocol};background:var(--panel);border-radius:10px;padding:16px 20px;
  border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}
.lamp{{display:flex;align-items:center;gap:11px}}
.dot{{width:15px;height:15px;border-radius:50%;background:{ocol};
  box-shadow:0 0 0 4px color-mix(in srgb,{ocol} 22%,transparent)}}
.lamp b{{font-size:21px;letter-spacing:-.01em}}
.kv{{display:flex;flex-direction:column;gap:1px}}
.kv span{{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}}
.kv b{{font-size:17px;font-weight:650}}
.spacer{{flex:1}}

.grid2{{display:grid;grid-template-columns:1.55fr 1fr;gap:18px;align-items:start}}
@media (max-width:900px){{.grid2{{grid-template-columns:1fr}}}}

/* 차트 */
.chart{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--line);stroke-width:1}}
.band{{fill:var(--accent);opacity:.17}}
.hist{{fill:none;stroke:var(--ink);stroke-width:2.1;stroke-linejoin:round}}
.fcast{{fill:none;stroke:var(--accent);stroke-width:2.4;stroke-dasharray:6 4;stroke-linejoin:round}}
.now{{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}}
.endpt{{fill:var(--accent)}}
.endlab{{fill:var(--accent);font-size:12px;font-weight:700;text-anchor:end;
  font-family:ui-monospace,monospace}}
.ylab,.xlab,.zlab{{fill:var(--muted);font-size:10.5px;
  font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.ylab{{text-anchor:end}} .xlab,.zlab{{text-anchor:middle}}
.zlab{{fill:var(--accent);font-size:9.5px;letter-spacing:.03em}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}}
.legend i{{display:inline-block;width:16px;height:0;border-top:2.4px solid;vertical-align:middle;margin-right:5px}}

/* 경보 표 */
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th{{text-align:left;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);font-weight:600;padding:0 8px 8px;border-bottom:1px solid var(--line)}}
td{{padding:8px;border-bottom:1px solid var(--line);vertical-align:middle}}
tr.sev-watch td:first-child{{box-shadow:inset 3px 0 0 var(--watch)}}
tr.sev-inspect td:first-child{{box-shadow:inset 3px 0 0 var(--inspect)}}
tr.sev-crit td:first-child{{box-shadow:inset 3px 0 0 var(--crit)}}
.vname{{font-weight:600;padding-left:11px}}
.tier{{display:block;font-size:10.5px;color:var(--muted);font-weight:400}}
td.num{{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
.unit{{color:var(--muted);font-size:11px;margin-left:3px}}
.sp{{color:var(--muted);font-size:12px;white-space:nowrap}}
.chip{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11.5px;font-weight:650;
  border:1px solid currentColor}}
.chip-ok{{color:var(--ok)}} .chip-watch{{color:var(--watch)}}
.chip-inspect{{color:var(--inspect)}} .chip-crit{{color:var(--crit)}} .chip-na{{color:var(--muted)}}
.tblwrap{{overflow-x:auto}}

/* 제어 지시 */
ol.acts{{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px;counter-reset:a}}
.act{{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:8px;
  background:var(--panel2);border-left:4px solid var(--line)}}
.act.sev-watch{{border-left-color:var(--watch)}} .act.sev-inspect{{border-left-color:var(--inspect)}}
.act.sev-crit{{border-left-color:var(--crit)}} .act.sev-ok{{border-left-color:var(--ok)}}
.act p{{margin:3px 0 0;font-size:13.5px;color:var(--ink2)}}
.act b{{font-size:14px}}

/* 신호등 이력 */
.strip{{display:flex;gap:2px;margin-top:4px}}
.hb{{flex:1;height:22px;border-radius:2px;background:var(--line)}}
.hb.sev-ok{{background:var(--ok)}} .hb.sev-watch{{background:var(--watch)}}
.hb.sev-inspect{{background:var(--inspect)}} .hb.sev-crit{{background:var(--crit)}}
.hb.sev-na{{background:var(--line)}}

.note{{border-left:4px solid var(--crit);background:var(--panel2);padding:13px 16px;border-radius:8px;
  font-size:13.5px;color:var(--ink2)}}
.note b{{color:var(--ink)}}
.foot{{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px;
  display:flex;flex-wrap:wrap;gap:18px}}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">영천 통합바이오가스화시설 · 혐기성 소화조</p>
    <h1>BioGuard-AI 통합 관제</h1>
  </header>

  <div class="status">
    <div class="lamp"><span class="dot"></span>
      <div class="kv"><span>종합 상태</span><b>{ov}</b></div></div>
    <div class="kv"><span>기준일</span><b class="mono">{p['as_of']}</b></div>
    <div class="kv"><span>설정값 이탈</span><b class="mono">{p['alarm']['deviating']} / {len(p['alarm']['items'])}</b></div>
    <div class="kv"><span>투입 계획</span><b class="mono">{p['plan_feed']:,.0f} ㎥/일</b></div>
    <div class="spacer"></div>
    <div class="kv"><span>익일 예측</span><b class="mono">{p['forecast'][0]['mean']:,.0f} Nm³/일</b></div>
  </div>

  <section class="panel">
    <h2>메탄생성량 예측 — 향후 {len(p['forecast'])}일 (투입 계획 {p['plan_feed']:,.0f} ㎥/일 기준)</h2>
    {_chart_svg(p)}
    <div class="legend">
      <span><i style="border-color:var(--ink)"></i>실측</span>
      <span><i style="border-color:var(--accent);border-top-style:dashed"></i>예측</span>
      <span><i style="border-color:var(--accent);opacity:.4"></i>80% 예측구간</span>
      <span>음영 = horizon 신뢰도 구역 (1–3 / 4–7 / 8–14일)</span>
    </div>
  </section>

  <div class="grid2">
    <section class="panel">
      <h2>계측값 · 설정값 대비</h2>
      <div class="tblwrap"><table>
        <thead><tr><th>변수</th><th style="text-align:right">현재값</th><th>설정값</th><th>상태</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
      <h2 style="margin-top:18px">최근 30일 종합 신호</h2>
      <div class="strip">{hist_strip}</div>
    </section>

    <section class="panel">
      <h2>공정제어 지시 — 우선순위 순</h2>
      <ol class="acts">{acts}</ol>
    </section>
  </div>

  <div class="grid2">
    <section class="panel">
      <h2>예측 신뢰도 — 며칠 앞까지 믿을 수 있는가</h2>
      <div class="tblwrap"><table>
        <thead><tr><th>horizon(일)</th><th style="text-align:right">투입계획 반영 R²</th>
          <th style="text-align:right">투입 미지 R²</th><th style="text-align:right">persistence R²</th></tr></thead>
        <tbody>{skill}</tbody>
      </table></div>
      <p style="font-size:13px;color:var(--ink2);margin:12px 0 0">
        <b>1–3일</b> 소화조 자체 동특성이 지배 — 매우 정확 ·
        <b>4–7일</b> 일상 운전 판단의 실용 구간 ·
        <b>8–14일</b> 투입계획을 반영해야 유지(미지 시 0.50) — 기질수급·정비 계획용 ·
        <b>15–28일</b> 시나리오 평가용(투입 미지 시 0.19로 붕괴).
        persistence 는 7일 이후 급락하므로 <b>장기일수록 모델의 우위가 커진다</b>.
      </p>
    </section>

    <section class="panel">
      <h2>상시 만성 리스크</h2>
      <div class="note">
        <b>유리암모니아(FAN) 만성 저해</b> — 중앙 <span class="mono">{p['chronic']['중앙값_mg_L']:,.0f} mg/L</span>,
        문헌 저해임계 {p['chronic']['문헌_저해임계_mg_L']} 대비 초과율
        <span class="mono">{p['chronic']['초과율_%']}%</span>. 추세 {p['chronic']['추세']}.
        <p style="margin:8px 0 0">{p['chronic']['상시경고']}</p>
        <p style="margin:8px 0 0">※ {p['chronic']['계측_권고']}</p>
      </div>
    </section>
  </div>

  <div class="foot">
    <span>모델 · {m['name']}</span>
    <span>2023 홀드아웃 R² <b class="mono">{m['holdout_R2']}</b> / RMSE <b class="mono">{m['holdout_RMSE']}</b></span>
    <span>교차검증 R² <b class="mono">{m['cv_R2']}</b></span>
    <span>지연모델 · {m['bio']}</span>
  </div>
</div>'''

if __name__ == "__main__":
    main()
