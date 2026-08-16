"""
체류시간 커널 — PROMPT §2.1 (전단, 기질별) · §2.2 (소화조 CSTR + 분해)

구조:
    전단커널_기질별 ⊛ 공통커널 ⊛ h(τ)

- 각 조는 완전혼합(CSTR)으로 보고 지수 RTD 를 직렬 합성곱한다.
- 합류(여액저장조) 이전은 기질별 개별 경로, 이후는 공통 — 단일 공통 커널 금지(금지사항 7).
- 커널 총량 보존: 전단·공통 커널은 Σw = 1 (PROMPT §9 물리검사).
- h(τ) = k·exp(−(1/SRT + k)·τ) 는 정규화하지 않는다. 그 총합이 정상이득
  G = k·SRT/(1+k·SRT) 이며, 이것이 §5.2 가 말하는 관측 가능량이다.
"""

from __future__ import annotations

import numpy as np

from .config import (LEGACY_TAU_MIX_D, PATH_COMMON, PATH_INDIVIDUAL, SCENARIOS,
                     STAGE_MODEL, STAGE_TAU, SUBSTRATES)

NK = 600          # 커널 길이 [일]


def cstr(tau: float, n: int = NK) -> np.ndarray:
    """이산 지수 RTD (완전혼합). Σ = 1."""
    e = np.exp(-np.arange(n) / float(tau))
    return e / e.sum()


def delay(tau: float, n: int = NK) -> np.ndarray:
    """순수 지연. 일 단위 격자에서 1일 미만은 사실상 τ=0."""
    k = np.zeros(n)
    k[min(int(round(tau)), n - 1)] = 1.0
    return k


def _conv(a: np.ndarray, b: np.ndarray, n: int = NK) -> np.ndarray:
    c = np.convolve(a, b)[:n]
    return c / c.sum()


def _stage(name: str, scen: str, n: int = NK) -> np.ndarray:
    tau = STAGE_TAU[name][SCENARIOS[scen]]
    return cstr(tau, n) if STAGE_MODEL[name] == "CSTR" else delay(tau, n)


def front_kernel(sub: str, scen: str, n: int = NK) -> np.ndarray:
    """반입 → 소화조 유입. 기질별 개별 경로 → 합류 → 공통 경로."""
    g = np.zeros(n)
    g[0] = 1.0
    for st in PATH_INDIVIDUAL[sub] + PATH_COMMON:
        g = _conv(g, _stage(st, scen, n), n)
    return g


def legacy_front_kernel(n: int = NK) -> np.ndarray:
    """기존 가정: 전 기질 공통 τ_mix=3일 단일 지수 커널 (비교 기준선용)."""
    return cstr(LEGACY_TAU_MIX_D, n)


def front_set(scen: str, n: int = NK) -> dict[str, np.ndarray]:
    if scen == "기존":
        g = legacy_front_kernel(n)
        return {s: g for s in SUBSTRATES}
    if scen == "없음":                       # ablation: 전단 지연 제거
        g = np.zeros(n)
        g[0] = 1.0
        return {s: g for s in SUBSTRATES}
    return {s: front_kernel(s, scen, n) for s in SUBSTRATES}


def digester_h(srt: float, k: float, n: int = NK) -> np.ndarray:
    """소화조 응답. 정규화하지 않는다 — Σh = 정상이득 G."""
    return k * np.exp(-(1.0 / float(srt) + float(k)) * np.arange(n, dtype=float))


def gain(srt: float, k: float) -> float:
    """G = k·SRT/(1+k·SRT). λ = 1/SRT + k 와 함께 유일하게 식별되는 두 양(§5.2)."""
    return k * srt / (1.0 + k * srt)


def kernel_stats(g: np.ndarray) -> dict:
    """RTD 요약 — 평균·분위수 [일]."""
    g = g / g.sum()
    t = np.arange(len(g), dtype=float)
    cdf = np.cumsum(g)
    q = lambda p: float(np.interp(p, cdf, t))
    return {"평균": round(float((t * g).sum()), 2), "t10": round(q(.10), 2),
            "t50": round(q(.50), 2), "t90": round(q(.90), 2), "t95": round(q(.95), 2)}
