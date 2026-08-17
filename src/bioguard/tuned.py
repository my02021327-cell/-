"""
채택 개선 모델 — 실측 투입 구동 · 폴드 내부 k 선택 · 수율 물리 제약.

`docs/EMPIRICAL_VS_MODEL.md` §5.1 의 개선을 검증한 결과 남은 것만 담는다.

  · 채택: 구동 변수 = 실측 투입(D1 수정), k 를 **폴드 학습셋 안에서** 선택, 수율 상한 부과
  · 기각: 2풀 분리 — 튜닝된 단일 풀보다 유의하게 나빴다(p=0.024). `twopool.py` 에 근거 보존

**k 선택은 반드시 폴드 안에서 한다.** 전 폴드 평균 CV-RMSE 를 보고 k 를 고르면 그 값으로
다시 CV-RMSE 를 보고하는 순환이 된다(선택 편향). 각 폴드의 학습 구간 끝 20% 를 내부
검증창으로 떼어 k 를 고르고, 고른 k 로 학습 전체를 다시 적합해 검정 구간을 예측한다.

**수율 상한은 박스 제약으로 표현된다.** 기질 s 의 무한지평 수율은 θ_s·G(k_s) 이므로
BD ≤ 1 은 θ_s ≤ B_th,s · VSfrac_s · 1000 / G(k_s) 이고, 이는 계수 하나에 대한 상한이라
비음수 최소자승(`lsq_linear`)이 그대로 처리한다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .config import SRT_REFERENCE_D, SUBSTRATES
from .empirical import feed_driven_design
from .kernels import gain

K_CANDIDATES = (0.05, 0.08, 0.12, 0.20, 0.30, 0.45)
INNER_VALID_FRAC = 0.20
MIN_INNER_OBS = 60


def upper_bounds(k: float, srt: float = SRT_REFERENCE_D, intercept: bool = True) -> np.ndarray:
    """BD ≤ 1 → θ_s ≤ B_th,s · VSfrac_s · 1000 / G(k)."""
    from .models import stoichiometry

    st = stoichiometry()["기질"]
    g = gain(srt, k)
    ub = [st[s]["B_th"] * st[s]["VSfrac"] * 1000.0 / g for s in SUBSTRATES]
    if intercept:
        ub.append(np.inf)          # 절편은 제약하지 않는다 — 크기를 그대로 보고하는 것이 규약
    return np.array(ub, dtype=float)


def _fit(W, y, idx, k, srt, cap):
    ub = upper_bounds(k, srt) if cap else np.full(W.shape[1], np.inf)
    return lsq_linear(W[idx], y[idx], bounds=(np.zeros(W.shape[1]), ub)).x


def make(F, y, srt: float = SRT_REFERENCE_D, cap: bool = True,
         candidates=K_CANDIDATES, intercept: bool = True):
    """전 기질 공통 k 를 후보에서 고른다(기질별 k 는 열간 공선성으로 식별되지 않는다)."""
    designs = {k: feed_driven_design(F, srt, {s: k for s in SUBSTRATES}, intercept)
               for k in candidates}

    def fit_predict(tr, te):
        n_in = max(int(len(tr) * INNER_VALID_FRAC), MIN_INNER_OBS)
        inner_tr, inner_va = tr[:-n_in], tr[-n_in:]
        if len(inner_tr) < 100 or len(inner_va) < 20:
            best_k = candidates[len(candidates) // 2]
        else:
            scores = {}
            for k, W in designs.items():
                beta = _fit(W, y, inner_tr, k, srt, cap)
                r = y[inner_va] - W[inner_va] @ beta
                scores[k] = float(np.sqrt((r @ r) / len(r)))
            best_k = min(scores, key=scores.get)
        W = designs[best_k]
        beta = _fit(W, y, tr, best_k, srt, cap)
        fit_predict.last_k = best_k
        fit_predict.last_beta = beta
        fit_predict.k_history.append(best_k)
        return W[te] @ beta

    fit_predict.k_history = []
    return fit_predict


def report(F, y, srt: float = SRT_REFERENCE_D, cap: bool = True, k: float = 0.20) -> dict:
    """전기간 적합 계수와 물리 해석 (보고용 — 모델 선택에 쓰지 않는다)."""
    from .config import SUB_KR
    from .models import stoichiometry

    st = stoichiometry()["기질"]
    W = feed_driven_design(F, srt, {s: k for s in SUBSTRATES}, True)
    obs = np.where(np.isfinite(y))[0]
    beta = _fit(W, y, obs, k, srt, cap)
    contrib = np.nanmean(W[obs] * beta, axis=0)
    g = gain(srt, k)
    out = {"k": k, "SRT_d": srt, "수율상한_부과": cap,
           "정상이득_G": round(float(g), 3),
           "절편_m3d": round(float(beta[-1]), 1),
           "절편_비중_pct": round(float(100 * contrib[-1] / np.nanmean(y[obs])), 1),
           "기질별": {}}
    for i, s in enumerate(SUBSTRATES):
        y_t = float(beta[i]) * g
        vs = y_t / st[s]["VSfrac"] / 1000.0
        out["기질별"][SUB_KR[s]] = {
            "θ_m3_per_t": round(float(beta[i]), 3),
            "총수율_m3_per_t": round(y_t, 3),
            "VS수율_m3_per_kgVS": round(vs, 4),
            "BD": round(vs / st[s]["B_th"], 3),
            "BD_통과": bool(0 <= vs / st[s]["B_th"] <= 1 + 1e-9),
            "평균기여_m3d": round(float(contrib[i]), 0),
        }
    return out
