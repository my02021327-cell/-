"""
개선 모델 — 속분해/난분해 2풀 분리 + 수율 물리 제약.

`docs/EMPIRICAL_VS_MODEL.md` §5.1 의 개선 방향 2·3 을 구현한다.

**왜 2풀인가.** 단일 풀 모델에서 절편이 실측의 34~43% 를 가져간다. 경험식도 같은 문제를
`C_base·MA30` 으로 처리했다(63%). 둘 다 뿌리가 같다 — **반감기가 크게 다른 두 분획을
하나의 지수로 표현할 수 없다.** 자유 절편이나 자기회귀항은 그 몫을 받아내지만,
부하와 무관해지므로 "부하를 바꾸면 가스가 어떻게 변하는가"에 답하지 못한다.

2풀은 그 몫을 **부하에 연동된 느린 응답**으로 되돌린다:

    h_s(τ) = θ_fast,s · k_f·exp(−(1/SRT + k_f)τ)  +  θ_slow,s · k_s·exp(−(1/SRT + k_s)τ)

두 항 모두 θ 에 대해 선형이므로 기질당 2열, 총 6열 + 절편의 비음수 최소자승으로 풀린다.

**물리적 한계 주의.** CSTR 세척 때문에 λ = 1/SRT + k ≥ 1/SRT 이다. SRT 25일이면
어떤 k 로도 반감기 17.3일보다 느려질 수 없다. (경험식의 k_m=0.010 → 반감기 69일은
SRT 40.6일의 상한 28.1일마저 넘는다 — 그 식에는 세척항이 없기 때문이다.)

**수율 제약.** 기질 s 의 무한지평 수율은 θ_fast,s·G_f + θ_slow,s·G_s [㎥CH₄/투입 t] 이고,
이것이 VSfrac 으로 나뉘어 B_th 를 넘으면 생분해도 BD>1 이 된다. 합에 대한 선형 부등식이라
박스 제약(lsq_linear)으로는 표현되지 않으므로 SLSQP 로 푼다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from .config import SRT_REFERENCE_D, SUBSTRATES
from .empirical import substrate_feed
from .gpu_backend import BK
from .kernels import NK

K_FAST = {"foodww": 0.30, "manure": 0.20, "food": 0.30}
K_SLOW = 0.02                      # 전 기질 공통 난분해 풀 (SRT 25 에서 반감기 11.6일)


def pool_kernel(srt: float, k: float, n: int = NK) -> np.ndarray:
    """정규화하지 않는다 — Σ = 정상이득 G = k·SRT/(1+k·SRT)."""
    return k * np.exp(-(1.0 / float(srt) + float(k)) * np.arange(n, dtype=float))


def gain(srt: float, k: float) -> float:
    return k * srt / (1.0 + k * srt)


def substrate_sources(F: pd.DataFrame) -> dict[str, np.ndarray]:
    """구동 변수 = 실측 투입 × 반입 조성비 (D1 수정 반영)."""
    Qf, Qm = substrate_feed(F)
    inner = (F.ratio_foodww + F.ratio_food).replace(0, np.nan)
    return {
        "foodww": (Qf * (F.ratio_foodww / inner).fillna(1.0)).to_numpy(float),
        "food": (Qf * (F.ratio_food / inner).fillna(0.0)).to_numpy(float),
        "manure": Qm.to_numpy(float),
    }


def design(F: pd.DataFrame, srt: float = SRT_REFERENCE_D, k_fast: dict | None = None,
           k_slow: float = K_SLOW, intercept: bool = True):
    """열 순서: [fast_s for s] + [slow_s for s] (+ 절편)."""
    kf = k_fast or K_FAST
    src = substrate_sources(F)
    cols, meta = [], []
    for s in SUBSTRATES:
        cols.append(BK.conv_causal(np.nan_to_num(src[s]), pool_kernel(srt, kf[s])))
        meta.append((s, "fast", gain(srt, kf[s])))
    for s in SUBSTRATES:
        cols.append(BK.conv_causal(np.nan_to_num(src[s]), pool_kernel(srt, k_slow)))
        meta.append((s, "slow", gain(srt, k_slow)))
    if intercept:
        cols.append(np.ones(len(F)))
        meta.append(("intercept", "-", 0.0))
    return np.column_stack(cols), meta


def yield_caps() -> dict[str, float]:
    """기질별 θ 조합 상한 [㎥CH₄/투입 t] — BD ≤ 1 (이론 최대 메탄포텐셜)."""
    from .models import stoichiometry

    st = stoichiometry()["기질"]
    return {s: st[s]["B_th"] * st[s]["VSfrac"] * 1000.0 for s in SUBSTRATES}


def fit(W: np.ndarray, y: np.ndarray, idx: np.ndarray, meta, cap: bool = False):
    """cap=False → 비음수 최소자승. cap=True → 기질별 수율 상한을 추가로 부과."""
    A, b = W[idx], y[idx]
    if not cap:
        return lsq_linear(A, b, bounds=(0.0, np.inf)).x

    caps = yield_caps()
    n_par = W.shape[1]
    gains = np.array([m[2] for m in meta])
    cons = []
    for s in SUBSTRATES:
        sel = np.array([1.0 if m[0] == s else 0.0 for m in meta]) * gains
        cons.append({"type": "ineq",
                     "fun": (lambda x, sel=sel, c=caps[s]: c - float(sel @ x))})
    x0 = lsq_linear(A, b, bounds=(0.0, np.inf)).x
    # 목적함수를 관측수와 분산으로 정규화한다. 원 스케일(잔차제곱합 ~1e10)에서는
    # SLSQP 의 기본 수렴 판정이 먼저 만족돼 제약이 걸리지 않는다.
    scale = float(len(b)) * float(np.var(b))
    res = minimize(lambda x: float(((A @ x - b) ** 2).sum() / scale), x0,
                   jac=lambda x: 2.0 * A.T @ (A @ x - b) / scale,
                   bounds=[(0.0, None)] * n_par, constraints=cons,
                   method="SLSQP", options={"maxiter": 800, "ftol": 1e-10})
    if not res.success:
        return x0
    # 제약이 실제로 지켜졌는지 확인 — 지켜지지 않으면 무제약 해를 돌려주고 상위에서 보고한다
    for c in cons:
        if c["fun"](res.x) < -1e-6:
            return x0
    return res.x


def make(F, y, srt=SRT_REFERENCE_D, k_fast=None, k_slow=K_SLOW, cap=False, intercept=True):
    W, meta = design(F, srt, k_fast, k_slow, intercept)

    def fit_predict(tr, te):
        beta = fit(W, y, tr, meta, cap)
        fit_predict.last_beta = beta
        return W[te] @ beta

    fit_predict.W, fit_predict.meta = W, meta
    return fit_predict


def report(F, y, srt=SRT_REFERENCE_D, k_fast=None, k_slow=K_SLOW, cap=False) -> dict:
    """전기간 적합 계수와 물리 해석 — 절편 비중, 기질별 수율, BD."""
    from .config import SUB_KR
    from .models import stoichiometry

    st = stoichiometry()["기질"]
    W, meta = design(F, srt, k_fast, k_slow, True)
    obs = np.where(np.isfinite(y))[0]
    beta = fit(W, y, obs, meta, cap)
    contrib = np.nanmean(W[obs] * beta, axis=0)
    kf = k_fast or K_FAST

    out = {"k_fast": kf, "k_slow": k_slow, "SRT_d": srt,
           "절편_m3d": round(float(beta[-1]), 1),
           "절편_비중_pct": round(float(100 * contrib[-1] / np.nanmean(y[obs])), 1),
           "기질별": {}}
    for s in SUBSTRATES:
        i_f = [i for i, m in enumerate(meta) if m == (s, "fast", gain(srt, kf[s]))][0]
        i_s = [i for i, m in enumerate(meta) if m[0] == s and m[1] == "slow"][0]
        y_tot = beta[i_f] * gain(srt, kf[s]) + beta[i_s] * gain(srt, k_slow)
        vs_yield = y_tot / st[s]["VSfrac"] / 1000.0
        out["기질별"][SUB_KR[s]] = {
            "θ_fast": round(float(beta[i_f]), 3), "θ_slow": round(float(beta[i_s]), 3),
            "총수율_m3_per_t": round(float(y_tot), 3),
            "VS수율_m3_per_kgVS": round(float(vs_yield), 4),
            "BD": round(float(vs_yield / st[s]["B_th"]), 3),
            "BD_통과": bool(0 <= vs_yield / st[s]["B_th"] <= 1),
            "느린풀_기여_pct": round(float(100 * beta[i_s] * gain(srt, k_slow) / y_tot), 1)
                          if y_tot > 0 else None,
            "평균기여_m3d": round(float(contrib[i_f] + contrib[i_s]), 0),
        }
    return out
