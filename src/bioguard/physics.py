"""
물리 정합성 검사 — PROMPT §9. 예측 성능과 별개로 반드시 통과해야 한다.
통과하지 못하면 그 계수를 생분해도·수율로 해석해서는 안 된다고 명기한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (COD_CAP_KG_PER_T, COD_PER_VS_KG, COD_TO_CH4_M3_PER_KG,
                     INTERCEPT_WARN_FRAC, SUBSTRATES, SUB_KR, V_DIGESTER_M3,
                     YIELD_CAP_VS_DESTROYED, YIELD_CAP_VS_IN)
from .models import stoichiometry


def check_coefficients(theta: dict, intercept: float, y_mean: float, basis: str) -> list[dict]:
    """계수 부호 · bdCOD 상한 · 절편 비중."""
    out = []
    for s in SUBSTRATES:
        v = float(theta.get(s, np.nan))
        out.append({"검사": f"계수 부호 ≥ 0 — {SUB_KR[s]}", "값": round(v, 4),
                    "기준": "≥ 0", "통과": bool(v >= -1e-9)})
    if basis == "tonnage":
        for s in SUBSTRATES:
            cap = COD_CAP_KG_PER_T[s]
            v = float(theta.get(s, np.nan))
            bd = v / COD_TO_CH4_M3_PER_KG
            zero = v < 1e-6
            out.append({"검사": f"bdCOD ≤ 총COD 상한 — {SUB_KR[s]}",
                        "값": "계수 0 (NNLS 하한)" if zero else round(bd, 1),
                        "기준": f"≤ {cap} kgCOD/t" if cap else "상한 식별 안 됨(판정 불가)",
                        "통과": (None if (cap is None or zero) else bool(bd <= cap))})
    else:
        st = stoichiometry()["기질"]
        for s in SUBSTRATES:
            v = float(theta.get(s, np.nan))
            zero = v < 1e-6                      # NNLS 가 하한(0)으로 밀어낸 계수
            out.append({"검사": f"메탄수율 상한 — {SUB_KR[s]}", "값": round(v, 4),
                        "기준": f"≤ {YIELD_CAP_VS_IN[s]} ㎥CH₄/kgVS_in",
                        "통과": None if zero else bool(v <= YIELD_CAP_VS_IN[s])})
            bd = v / st[s]["B_th"] if st[s]["B_th"] else np.nan
            out.append({"검사": f"생분해도 0 < BD ≤ 1 — {SUB_KR[s]}",
                        "값": "계수 0 (NNLS 하한) — 수율 판정 불가" if zero else round(float(bd), 3),
                        "기준": "0 < BD ≤ 1",
                        "통과": None if zero else bool(0 < bd <= 1)})
    frac = intercept / y_mean if y_mean else np.nan
    out.append({"검사": "절편 비중 (은폐 금지)", "값": f"{intercept:,.0f} ㎥/d = 실측의 {100*frac:.1f}%",
                "기준": f"보고 필수. {INTERCEPT_WARN_FRAC*100:.0f}% 초과 시 T1 단독 예측 중단 신호",
                "통과": bool(frac <= INTERCEPT_WARN_FRAC)})
    return out


def check_kernels(fronts: dict) -> list[dict]:
    return [{"검사": f"커널 총량 보존 — {SUB_KR[s]}", "값": round(float(g.sum()), 6),
             "기준": "Σw = 1", "통과": bool(abs(g.sum() - 1) < 1e-6)}
            for s, g in fronts.items()]


def vs_balance(F: pd.DataFrame, v_digester: float = V_DIGESTER_M3) -> dict:
    """
    PROMPT §4 물질수지 이탈 재현. **억지로 맞추지 않는다 — 이탈 자체가 산출물이다.**
    """
    sub = F.dropna(subset=["VS_in_kgd", "VS_out_kgd", "ch4"])
    vs_in, vs_out = sub.VS_in_kgd.mean(), sub.VS_out_kgd.mean()
    ts_route = (F.TS_load_tpd * 1000 * 0.73).mean()
    cons = vs_in - vs_out
    ch4 = sub.ch4.mean()
    theo_destroyed = COD_TO_CH4_M3_PER_KG * COD_PER_VS_KG
    return {
        "n": int(len(sub)), "V_m3": v_digester,
        "유입VS_OLR경로_kgd": round(float(vs_in)),
        "유입VS_TS부하경로_kgd": round(float(ts_route)),
        "유출VS_kgd": round(float(vs_out)),
        "소비VS_kgd": round(float(cons)),
        "VS분해율_pct": round(float(100 * cons / vs_in), 1),
        "실측CH4_m3d": round(float(ch4)),
        "함축수율_소비VS기준": round(float(ch4 / cons), 3),
        "함축수율_유입VS기준": round(float(ch4 / vs_in), 3),
        "이론상한_소비VS기준": round(theo_destroyed, 3),
        "초과배수": round(float(ch4 / cons / theo_destroyed), 2),
        "판정": "물리적으로 불가능 — T2 절대수준 예측 사용 금지 (기울기만 평가)",
        "V_의존성": "유입·유출·소비 VS 는 모두 V 와 무관하다(유입=부하 원자료, 유출=Q_out×VS%). "
                 "V 는 조내 VS 와 SRT 에만 들어간다. 따라서 **용적을 바꿔 이 이탈을 해소할 수 없다** — "
                 "PROMPT §4 가 든 세 원인 중 'V 오류'는 이 경로로는 배제된다.",
        "가능원인": ["유입 VS 과소 계량 (미기록 병합기질·반송수·침출수 유기물)",
                  "CH₄ 계량 과대 (Nm³ vs ㎥, 온도·압력 보정, 농도 환산)",
                  "V 또는 소화액 VS% 오류 → 유출 VS 과소"],
    }


def identifiability(srt: float, k: dict) -> dict:
    """§5.2 — 관측 가능량은 λ 와 G 둘뿐이고 미지수는 (θ, k, SRT) 셋이다."""
    return {"설명": "감쇠율 λ=1/SRT+k 와 정상이득 G=θ·k·SRT/(1+k·SRT) 두 개만 관측된다. "
                  "k 를 고정해도 비식별성이 θ 로 전가될 뿐이다.",
            "기질별": {SUB_KR[s]: {"k": k[s], "lambda": round(1 / srt + k[s], 4),
                                "G_per_theta": round(k[s] * srt / (1 + k[s] * srt), 4)}
                    for s in SUBSTRATES},
            "지시": "SRT 는 외생 파라미터. 사후 해석에서만 대입한다."}
