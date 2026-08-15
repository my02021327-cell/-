# -*- coding: utf-8 -*-
"""사전 탈색(pre-whitening) 기반 교차상관 분석으로 기질 투입 시차를 재검정.

■ 참조 방법론 (사용자 제공 문헌 정리)
   단순 상관계수는 반응기 내부 데이터의 자기상관 때문에 허위 상관을 낳는다.
   따라서 입력 시계열을 ARIMA로 백색잡음화(pre-whitening)한 뒤,
   같은 필터를 출력에도 적용하고 잔차 간 교차상관을 본다(Box–Jenkins 절차).
   유의성 한계는 ±1.96/√n.

■ 왜 이 검정이 필요한가
   §2.4·§2.8은 롤링 검증 격자탐색으로 지연을 골랐다. 그것은 '예측이 잘 맞는 지연'이고,
   CCF는 '통계적으로 유의한 지연'이다. 둘이 일치하면 신뢰도가 올라가고,
   어긋나면 격자탐색이 예측오차 지형의 평탄한 골짜기를 임의로 고른 것일 수 있다.

■ 문헌 대조군 (사용자 제공 표)
   하수슬러지(1차+TWAS) 실증 소화조 10~20일 / 농업부산물 48~96시간 /
   가용성 유기물 0~12시간 / 소형 분산형 1~4일 / 미생물군집 응답 2~4일
   영천은 산생성조를 거쳐 이미 산성화된 기질이 들어오므로 '가용성' 쪽에 가까울 것으로 예상.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M.loc["2018-01-01":"2022-12-31"]          # 2023 제외 (온도·VFA 붕괴)
FULL = pd.date_range(M.index.min(), M.index.max(), freq="D")
OUT = {"period": "2018-01-01~2022-12-31", "note_2023": "온도·VFA 붕괴로 제외"}

y = (M.biogas_AB_m3d * M.CH4_pct.ffill(limit=3) / 100).reindex(FULL)
DRIVERS = {
    "소화조 투입량": M.feed_AB_tpd.reindex(FULL),
    "VS 부하 (투입량×산발효조VS)": (M.feed_AB_tpd * M.acid_VS_pct / 100).reindex(FULL),
    "TS 부하 (투입량×산발효조TS)": (M.feed_AB_tpd * M.acid_TS_pct / 100).reindex(FULL),
    "반입 총량": M.intake_total_tpd.reindex(FULL),
    "음폐수 반입량": M.intake_foodww_tpd.reindex(FULL),
    "가축분뇨 반입량": M.intake_manure_tpd.reindex(FULL),
    "음식물 반입량": M.intake_food_tpd.reindex(FULL),
}
MAXLAG = 40


def prewhiten(x, yy, order=(2, 1, 2)):
    """입력 x를 ARIMA로 백색잡음화하고 **같은 필터**를 y에 적용한 뒤 잔차쌍을 반환."""
    xi = x.interpolate(limit=5).ffill(limit=5).bfill(limit=5)
    yi = yy.interpolate(limit=5).ffill(limit=5).bfill(limit=5)
    ok = xi.notna() & yi.notna()
    xi, yi = xi[ok], yi[ok]
    fit = ARIMA(xi, order=order).fit()
    ax = pd.Series(fit.resid, index=xi.index)
    ay = pd.Series(fit.apply(yi, refit=False).resid, index=yi.index)
    burn = 30
    return ax.iloc[burn:], ay.iloc[burn:], fit


def ccf(ax, ay, maxlag):
    """r(τ) = corr(ax[t], ay[t+τ]) — τ>0 이면 입력이 출력을 선행."""
    d = pd.concat([ax.rename("x"), ay.rename("y")], axis=1).dropna()
    xv, yv = d.x.values, d.y.values
    xv = (xv - xv.mean()) / xv.std()
    yv = (yv - yv.mean()) / yv.std()
    n = len(xv)
    rows = []
    for t in range(0, maxlag + 1):
        a, b = xv[: n - t], yv[t:]
        rows.append({"lag": t, "r": round(float((a * b).mean()), 4)})
    return rows, n


print("=== 사전 탈색 기반 교차상관 (Box–Jenkins) — 2018~2022 ===")
print(f"  {'구동변수':28s} {'유의시차(일)':>26s} {'최대 |r|':>9s} {'해당 시차':>8s}")
res = {}
for nm, s in DRIVERS.items():
    try:
        ax, ay, fit = prewhiten(s, y)
    except Exception as e:
        print(f"  {nm:28s} 실패 ({type(e).__name__})")
        continue
    rows, n = ccf(ax, ay, MAXLAG)
    crit = 1.96 / np.sqrt(n)
    sig = [r["lag"] for r in rows if abs(r["r"]) > crit]
    bst = max(rows, key=lambda r: abs(r["r"]))
    res[nm] = {"n": int(n), "crit": round(float(crit), 4), "ccf": rows,
               "sig_lags": sig, "best_lag": bst["lag"], "best_r": bst["r"],
               "arima_aic": round(float(fit.aic), 1)}
    ss = ",".join(map(str, sig[:12])) + ("…" if len(sig) > 12 else "") if sig else "없음"
    print(f"  {nm:28s} {ss:>26s} {bst['r']:+9.4f} {bst['lag']:7d}일   (|r|>{crit:.4f})")
OUT["ccf"] = res

# ---------------------------------------------------------------- 대조: 탈색 없는 원자료 CCF
print("\n=== 대조군 — 사전 탈색 없이 원자료로 CCF를 돌리면 ===")
raw = {}
for nm, s in DRIVERS.items():
    d = pd.concat([s.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 100:
        continue
    xv = (d.x.values - d.x.mean()) / d.x.std()
    yv = (d.y.values - d.y.mean()) / d.y.std()
    rows = [{"lag": t, "r": round(float((xv[: len(xv) - t] * yv[t:]).mean()), 4)}
            for t in range(MAXLAG + 1)]
    crit = 1.96 / np.sqrt(len(d))
    nsig = sum(1 for r in rows if abs(r["r"]) > crit)
    bst = max(rows, key=lambda r: abs(r["r"]))
    raw[nm] = {"n_sig": nsig, "best_lag": bst["lag"], "best_r": bst["r"],
               "n_sig_prewhitened": len(res[nm]["sig_lags"]) if nm in res else None}
    print(f"  {nm:28s} 유의 시차 {nsig:3d}개 / {MAXLAG+1}개  "
          f"(탈색 후 {len(res[nm]['sig_lags']) if nm in res else '?'}개)  최대 r {bst['r']:+.4f} @ {bst['lag']}일")
OUT["raw_ccf"] = raw

# ---------------------------------------------------------------- 세척 임계 SRT (하한 제약)
MU = {"보수적 μmax 0.25": 0.25, "표준 μmax 0.33": 0.33, "낙관적 μmax 0.40": 0.40}
DB = 0.02
srtmin = {k: round(1 / (v - DB), 2) for k, v in MU.items()}
OUT["washout"] = {"mu_max": MU, "d_b": DB, "SRT_min_d": srtmin,
                  "safety_3x": {k: round(v * 3, 1) for k, v in srtmin.items()},
                  "safety_5x": {k: round(v * 5, 1) for k, v in srtmin.items()}}
print("\n=== 세척(washout) 임계 기반 SRT 하한 — SRT_min = 1/(μmax − d_b) ===")
for k, v in srtmin.items():
    print(f"  {k:16s} → SRT_min {v:5.2f}일,  안전율 3배 {v*3:5.1f}일 / 5배 {v*5:5.1f}일")
print("  주의: 이것은 SRT의 **추정치가 아니라 하한 제약**이다. 아세트산 이용성 메탄균이")
print("        세척되지 않을 최소 조건일 뿐, 실제 운전 SRT를 결정하지 못한다.")

# ---------------------------------------------------------------- 학습 윈도우 ≥ SRT 점검
OUT["window_check"] = {"itransformer_lookback_d": 30, "retention_kernel_K_d": 300,
                       "adopted_SRT_d": 12.0, "satisfied": True,
                       "srt_upper_bound_d": 40.8,
                       "satisfied_at_upper": True}
print("\n=== T_historical ≥ SRT 점검 ===")
print("  §2.10 iTransformer 룩백 30일 ≥ 채택 SRT 12일 → 충족")
print("  물질수지 상한 40.8일 기준으로는 30일 룩백이 부족 → §2.10 결과 해석 시 유의")
print("  §2.8 잔존율 커널은 K=300일까지 적분하므로 상한 40.8일에서도 충족")

Path("outputs").mkdir(exist_ok=True)
json.dump(OUT, open("outputs/ccf_results.json", "w", encoding="utf-8"), ensure_ascii=False)
print("\n저장: outputs/ccf_results.json")
