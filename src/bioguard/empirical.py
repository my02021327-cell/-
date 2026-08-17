"""
현장 경험식(`docs/영천BGP_메탄발생량_예측경험식.md`) 구현과, 그 비교에서 드러난
우리 T1 의 결함을 고친 변형 모델.

경험식:
    P_CH4(t) = Σ_{d=1..7} [W_f(d)·Q_f(t−d) + W_m(d)·Q_m(t−d)] + C_base·MA30(t−1)
    W_i(d)   = Y_i · k_i · exp(−k_i·d)

두 가지를 반드시 구분해서 평가한다.

  · **Nowcast** — MA30(t−1) 에 실측 메탄을 넣는다. 경험식이 원래 상정한 사용법이고,
    "어제까지 실측을 보며 오늘을 추정"하는 운전 보조용이다.
  · **Forecast** — 90일 앞을 내다볼 때는 MA30(t−1) 을 알 수 없다. 자기 예측으로
    재귀 갱신해야 한다. 이때 C_base·MA30 은 자기회귀항이 되고, 저장소 명세 W1 이
    경고한 구조(반응조 관성이 만드는 허위 설명력)가 그대로 드러난다.

같은 식을 두 모드로 돌려 성능 차이를 보는 것이 이 비교의 핵심이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .config import SRT_REFERENCE_D, SUBSTRATES
from .gpu_backend import BK
from .kernels import digester_h

# 경험식 원 파라미터 (L-BFGS-B 최적화 결과로 문서에 제시된 값)
EMP = {"Y_f": 26.51, "k_f": 0.178, "Y_m": 5.31, "k_m": 0.010, "C_base": 0.575, "D": 7}


def emp_weights(par: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = par or EMP
    d = np.arange(1, p["D"] + 1)
    return (p["Y_f"] * p["k_f"] * np.exp(-p["k_f"] * d),
            p["Y_m"] * p["k_m"] * np.exp(-p["k_m"] * d))


def substrate_feed(F: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    경험식의 Q_f·Q_m — '소화조에 투입된' 기질별 톤수.

    시설은 기질별 투입량을 따로 계량하지 않는다. 유일하게 가능한 분해는
    **실측 총투입량 × 반입 조성비** 이며, 이 근사를 명시적으로 쓴다.
    (반입 조성비는 저류조 체류만큼 지연되지만, 조성 자체가 완만해 일 단위 왜곡은 작다.)
    """
    rf = F.ratio_foodww + F.ratio_food
    rm = F.ratio_manure
    tot = (rf + rm).replace(0, np.nan)
    return F.feed_AB * rf / tot, F.feed_AB * rm / tot


def ma30_observed(y: np.ndarray) -> np.ndarray:
    """MA30(t−1) — 실측 기반. 결측이 많으므로 관측일만으로 이동평균한다."""
    s = pd.Series(y)
    return s.rolling(30, min_periods=5).mean().shift(1).to_numpy()


def empirical_series(F: pd.DataFrame, y: np.ndarray, par: dict | None = None,
                     mode: str = "nowcast", origin: int | None = None) -> np.ndarray:
    """
    mode="nowcast"  : MA30 에 실측을 넣는다 (경험식의 원래 사용법)
    mode="forecast" : origin 이후로는 자기 예측으로 MA30 을 재귀 갱신한다
    """
    p = par or EMP
    Wf, Wm = emp_weights(p)
    Qf, Qm = substrate_feed(F)
    qf, qm = Qf.to_numpy(float), Qm.to_numpy(float)
    n = len(F)

    sub = np.zeros(n)
    for i in range(p["D"]):
        lag = i + 1
        sub[lag:] += Wf[i] * np.nan_to_num(qf[:-lag]) + Wm[i] * np.nan_to_num(qm[:-lag])

    if mode == "nowcast" or origin is None:
        return sub + p["C_base"] * ma30_observed(y)

    # forecast: origin 이전은 실측, 이후는 자기 예측으로 채운 계열의 30일 평균
    hist = y.astype(float).copy()
    out = np.full(n, np.nan)
    out[:origin] = sub[:origin] + p["C_base"] * ma30_observed(y)[:origin]
    filled = hist.copy()
    for t in range(origin, n):
        win = filled[max(0, t - 30):t]
        win = win[np.isfinite(win)]
        ma = win.mean() if len(win) >= 5 else np.nanmean(filled[:t])
        out[t] = sub[t] + p["C_base"] * ma
        filled[t] = out[t]            # 실측이 없으므로 자기 예측을 이어 붙인다
    return out


def make_empirical(F, y, mode="forecast", par=None):
    """CV 용 fit_predict. 파라미터는 문서 값 그대로 — 폴드마다 재적합하지 않는다."""
    def fit_predict(tr, te):
        origin = int(te.min())
        return empirical_series(F, y, par, mode, origin)[te]
    return fit_predict


def make_empirical_refit(F, y, mode="forecast"):
    """
    구조는 경험식 그대로 두고 **선형 계수만 폴드 학습셋에서 다시 적합**한다.
    (Y_f·Y_m·C_base 는 선형, k 는 문서값 고정 → 비음수 최소자승으로 풀 수 있다)
    원 파라미터가 우리 타깃 계열에 맞지 않아 생기는 편향과, 구조 자체의 한계를 분리한다.
    """
    p = EMP
    d = np.arange(1, p["D"] + 1)
    bf, bm = p["k_f"] * np.exp(-p["k_f"] * d), p["k_m"] * np.exp(-p["k_m"] * d)
    Qf, Qm = substrate_feed(F)
    qf, qm = Qf.to_numpy(float), Qm.to_numpy(float)
    n = len(F)
    cf, cm = np.zeros(n), np.zeros(n)
    for i in range(p["D"]):
        lag = i + 1
        cf[lag:] += bf[i] * np.nan_to_num(qf[:-lag])
        cm[lag:] += bm[i] * np.nan_to_num(qm[:-lag])

    def fit_predict(tr, te):
        ma_tr = ma30_observed(y)
        X = np.column_stack([cf, cm, np.nan_to_num(ma_tr)])
        ok = np.isfinite(y[tr]) & np.isfinite(ma_tr[tr])
        if ok.sum() < 30:
            return np.full(len(te), np.nan)
        b = lsq_linear(X[tr][ok], y[tr][ok], bounds=(0.0, np.inf)).x
        par = {**p, "Y_f": float(b[0]), "Y_m": float(b[1]), "C_base": float(b[2])}
        fit_predict.last = {"Y_f": round(par["Y_f"], 3), "Y_m": round(par["Y_m"], 3),
                            "C_base": round(par["C_base"], 4)}
        return empirical_series(F, y, par, mode, int(te.min()))[te]

    return fit_predict


# ---------------------------------------------------------------------------
# 개선안 — 실측 투입량을 구동 변수로 쓰는 기계론 모델
# ---------------------------------------------------------------------------

def feed_driven_design(F: pd.DataFrame, srt: float, k_hyd: dict,
                       intercept: bool = True) -> np.ndarray:
    """
    구동 변수를 **반입(intake) → 전단 커널** 대신 **실측 투입(feed_AB) × 조성비** 로 바꾼다.

    근거: 전단 커널로 재구성한 투입은 실측 feed_AB 와 r=0.548 에 그친다(원 반입 합계는
    r=0.203). 여액저장조는 수동적 CSTR 이 아니라 **투입을 일정하게 유지하도록 조작되는
    완충조**(투입 CV 17.3% vs 반입 CV 46.3%)이므로, 지수 RTD 로는 원리적으로 재현되지
    않는다. 실측 투입은 그 조작의 결과를 이미 담고 있다.

    따라서 이 설계행렬에는 전단 커널을 넣지 않는다 — feed_AB 는 이미 저류조·산발효조를
    통과한 뒤의 값이다. 소화조 응답 h(τ) 만 적용한다.
    """
    Qf, Qm = substrate_feed(F)
    src = {"foodww": Qf.to_numpy(float) * (F.ratio_foodww / (F.ratio_foodww + F.ratio_food)
                                           .replace(0, np.nan)).fillna(1.0).to_numpy(float),
           "food": Qf.to_numpy(float) * (F.ratio_food / (F.ratio_foodww + F.ratio_food)
                                         .replace(0, np.nan)).fillna(0.0).to_numpy(float),
           "manure": Qm.to_numpy(float)}
    cols = [BK.conv_causal(np.nan_to_num(src[s]), digester_h(srt, k_hyd[s])) for s in SUBSTRATES]
    if intercept:
        cols.append(np.ones(len(F)))
    return np.column_stack(cols)


def make_M1_feed(F, y, srt=SRT_REFERENCE_D, k_hyd=None, intercept=True):
    from .config import K_HYD
    W = feed_driven_design(F, srt, k_hyd or K_HYD, intercept)

    def fit_predict(tr, te):
        b = lsq_linear(W[tr], y[tr], bounds=(0.0, np.inf)).x
        fit_predict.last_beta = b
        return W[te] @ b

    fit_predict.W = W
    return fit_predict


def reconstruction_check(F: pd.DataFrame) -> dict:
    """전단 커널이 실측 투입을 재구성하는가 — T1 구동 변수 선택의 근거."""
    from .kernels import front_set

    fr = front_set("표준")
    rec = sum(BK.conv_causal(F[f"recv_{s}"].to_numpy(float), fr[s]) for s in SUBSTRATES)
    raw = F[[f"recv_{s}" for s in SUBSTRATES]].sum(axis=1).to_numpy(float)
    feed = F.feed_AB.to_numpy(float)
    ok = np.isfinite(feed) & np.isfinite(rec)
    return {
        "반입합계_vs_실측투입_r": round(float(np.corrcoef(raw[ok], feed[ok])[0, 1]), 3),
        "전단커널재구성_vs_실측투입_r": round(float(np.corrcoef(rec[ok], feed[ok])[0, 1]), 3),
        "재구성_평균_tpd": round(float(rec[ok].mean()), 1),
        "실측투입_평균_tpd": round(float(feed[ok].mean()), 1),
        "반입_CV_pct": round(float(100 * raw.std() / raw.mean()), 1),
        "투입_CV_pct": round(float(100 * np.nanstd(feed) / np.nanmean(feed)), 1),
        "판정": "전단 커널은 재구성 상관을 0.203 → 0.548 로 크게 올리지만 실측 투입을 "
              "대체하지 못한다. 여액저장조는 수동 CSTR 이 아니라 투입을 일정하게 유지하도록 "
              "조작되는 완충조이므로 지수 RTD 로 표현되지 않는다. 실측 투입이 있으면 그것을 "
              "구동 변수로 쓰는 편이 원리적으로 옳다.",
    }
