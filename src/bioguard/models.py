"""
트랙 T1·T2·T3 와 모델 계열 M1~M5 — PROMPT §3 / §7.

모든 모델은 `make(...)` 로 `fit_predict(tr_idx, te_idx) -> ndarray` 를 돌려준다.
폴드는 바깥에서 한 번만 정해 전 모델이 공유한다(§8-1).

예측 지평 규약 — 이 구현의 핵심 판단:
    rolling-origin 은 원점 이후 90일을 예측한다. 반입량·투입량 같은 **외생 공정 입력**은
    계획·계량되는 값이라 알고 있다고 본다(PROMPT §5.5 기준선도 그렇게 만들어졌다).
    반면 소화조 내부 이화학은 **예측 시점에 알 수 없다**(§3 T3, §8-5). 따라서
      · Forecast 트랙: 내부 이화학은 lag 90일(= 원점 시점 정보)로만 사용
      · Nowcast 트랙: 동시점 사용 — 별도 표기하며 예측 성능 비교에 쓰지 않는다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .config import (BUSWELL_COMP, COD_PER_VS_KG, COD_TO_CH4_M3_PER_KG, CV_HORIZON_D,
                     K_HYD, MOLAR_VOLUME_L, SEED, SRT_REFERENCE_D, SUBSTRATES,
                     SUB_COMPOSITION)
from .gpu_backend import BK
from .kernels import digester_h, front_set

# ---------------------------------------------------------------------------
# T1 — 화학양론 (Buswell)
# ---------------------------------------------------------------------------

def buswell_ch4_per_kg(c: int, h: int, o: int, n: int, Mw: float) -> float:
    """CcHhOoNn 1 kg 당 이론 CH₄ [㎥ at STP]."""
    ch4_mol = c / 2 + h / 8 - o / 4 - 3 * n / 8
    return ch4_mol * MOLAR_VOLUME_L / Mw / 1000.0 * 1000.0


def stoichiometry() -> dict:
    """기질별 이론 메탄포텐셜 B_th [㎥CH₄/kgVS] 와 생분해도 반영값 B_eff."""
    comp = {k: dict(v, B_th=round(buswell_ch4_per_kg(v["c"], v["h"], v["o"], v["n"], v["Mw"]), 4))
            for k, v in BUSWELL_COMP.items()}
    sub = {}
    for s, p in SUB_COMPOSITION.items():
        b_th = sum(p[f] * comp[f]["B_th"] for f in ("carb", "prot", "lipid"))
        sub[s] = {**p,
                  "B_th": round(b_th, 4),                       # ㎥CH₄/kgVS (이론 최대)
                  "B_eff": round(b_th * p["BD"], 4),            # 생분해도 반영
                  "VSfrac": round(p["TSfrac"] * p["VSTS"], 5)}  # 습중량 t 당 VS 톤
    return {"분자식": comp, "기질": sub}


def substrate_vs_load(F: pd.DataFrame) -> dict[str, np.ndarray]:
    """기질별 반입 VS 부하 [kg VS/d] = 반입톤수 × VSfrac × 1000."""
    st = stoichiometry()["기질"]
    return {s: (F[f"recv_{s}"].to_numpy(float) * st[s]["VSfrac"] * 1000.0) for s in SUBSTRATES}


# ---------------------------------------------------------------------------
# 설계행렬 — 기질별 전단커널 ⊛ h(τ)
# ---------------------------------------------------------------------------

def design_matrix(F: pd.DataFrame, scen: str, srt: float, k_hyd: dict,
                  intercept: bool = True, basis: str = "tonnage") -> np.ndarray:
    """
    basis="tonnage" : 열 = 반입 톤수 응답  → 계수 θ_j [㎥CH₄/t]
    basis="vs"      : 열 = 반입 VS 부하 응답 → 계수 = 수율 [㎥CH₄/kgVS]
    """
    fronts = front_set(scen)
    src = (substrate_vs_load(F) if basis == "vs"
           else {s: F[f"recv_{s}"].to_numpy(float) for s in SUBSTRATES})
    cols = [BK.conv_causal(BK.conv_causal(src[s], fronts[s]), digester_h(srt, k_hyd[s]))
            for s in SUBSTRATES]
    if intercept:
        cols.append(np.ones(len(F)))
    return np.column_stack(cols)


def nnls_fit(W: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """계수 ≥ 0 제약 (PROMPT §9 '계수 부호'). 절편도 비음수로 둔다."""
    return lsq_linear(W[idx], y[idx], bounds=(0.0, np.inf)).x


# ---------------------------------------------------------------------------
# M1 — 기계론적 합성곱
# ---------------------------------------------------------------------------

def make_M1(F, y, scen="표준", srt=SRT_REFERENCE_D, k_hyd=None, intercept=True, basis="tonnage"):
    W = design_matrix(F, scen, srt, k_hyd or K_HYD, intercept, basis)

    def fit_predict(tr, te):
        return W[te] @ nnls_fit(W, y, tr)

    fit_predict.W = W
    return fit_predict


# ---------------------------------------------------------------------------
# M2 — ADL (자기회귀 분포시차)
#   PROMPT §7 이 명시적으로 요구한다. 다만 저장소 명세(W1)는 y_lag1 을 금지한다 —
#   HRT 40.5일 관성이 만든 허위 R² 때문이다. 충돌을 인지한 채 지시대로 구현하되,
#   90일 앞 예측에서는 y(t−1) 을 알 수 없으므로 **재귀 시뮬레이션**으로 예측한다.
#   (관측값을 그대로 넣으면 그것이 곧 누출이다.)
# ---------------------------------------------------------------------------

def make_M2(F, y, scen="표준", srt=SRT_REFERENCE_D, k_hyd=None):
    X = design_matrix(F, scen, srt, k_hyd or K_HYD, intercept=True, basis="vs")
    n = len(F)

    def fit_predict(tr, te):
        tr = tr[tr > 0]
        A = np.column_stack([X[tr], y[tr - 1]])
        ok = np.isfinite(A).all(axis=1) & np.isfinite(y[tr])
        if ok.sum() < 20:
            return np.full(len(te), np.nan)
        beta = np.linalg.lstsq(A[ok], y[tr][ok], rcond=None)[0]
        phi = float(np.clip(beta[-1], 0.0, 0.99))
        b = beta[:-1]
        fit_predict.last_phi = phi
        fit_predict.last_implied_hrt = (float(-1.0 / np.log(phi)) if 0 < phi < 1 else None)
        # 재귀 예측: 폴드 시작 직전 관측을 초기값으로 두고 앞으로 굴린다
        start = int(te.min())
        hist = y[:start][np.isfinite(y[:start])]
        prev = float(hist[-1]) if len(hist) else float(np.nanmean(y[tr]))
        path = np.full(n, np.nan)
        for t in range(start, int(te.max()) + 1):
            cur = float(X[t] @ b + phi * prev)
            path[t] = cur
            prev = cur
        return path[te]

    fit_predict.implied_hrt = "조정속도 φ 에서 함축 HRT = −1/ln(φ) [일]"
    return fit_predict


# ---------------------------------------------------------------------------
# M3 — VS 물질수지 회귀 (T2)
# ---------------------------------------------------------------------------

def make_M3(F, y, intercept=True):
    cols = ["VS_consumed_kgd", "VS_reactor_kg", "VS_out_kgd"]
    X = F[cols].to_numpy(float)
    if intercept:
        X = np.column_stack([X, np.ones(len(F))])

    def fit_predict(tr, te):
        ok = np.isfinite(X[tr]).all(axis=1) & np.isfinite(y[tr])
        if ok.sum() < 30:
            return np.full(len(te), np.nan)
        beta = lsq_linear(X[tr][ok], y[tr][ok], bounds=(0.0, np.inf)).x
        p = X[te] @ beta
        p[~np.isfinite(X[te]).all(axis=1)] = np.nan       # 결측일은 예측하지 않는다
        return p

    fit_predict.columns = cols
    return fit_predict


# ---------------------------------------------------------------------------
# T3 특징 — Forecast(lag 90) / Nowcast(동시점)
# ---------------------------------------------------------------------------
T3_VARS = ["dig_pH_A", "VFA_A", "ALK_A", "VFA_ALK_A", "NH3N_A", "FAN_A",
           "dig_TS_A_pct", "dig_VS_A_pct", "OLR", "acid_TS_pct", "dig_CODcr_A"]


def t3_features(F: pd.DataFrame, mode: str = "forecast") -> pd.DataFrame:
    lag = CV_HORIZON_D if mode == "forecast" else 0
    out = {}
    for v in T3_VARS:
        s = F[v]
        out[f"{v}_lag{lag}"] = s.shift(lag)
        out[f"{v}_ma30_lag{lag}"] = s.rolling(30, min_periods=5).mean().shift(lag)
    return pd.DataFrame(out, index=F.index)


LAGS = (0, 1, 2, 3, 5, 7, 10, 14, 20, 30)


def t1_lag_features(F: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for s in SUBSTRATES:
        r = F[f"recv_{s}"]
        for L in LAGS:
            out[f"recv_{s}_lag{L}"] = r.shift(L)
        for w in (7, 14, 30):
            out[f"recv_{s}_ma{w}"] = r.rolling(w, min_periods=1).mean()
        out[f"ratio_{s}"] = F[f"ratio_{s}"]
    out["feed_AB"] = F.feed_AB
    out["TS_load"] = F.TS_load_tpd
    out["dow"] = F.dow
    return pd.DataFrame(out, index=F.index)


# ---------------------------------------------------------------------------
# M4 — 이산 lag 격자 + 트리 (T1+T3)
# ---------------------------------------------------------------------------

def make_M4(F, y, mode="forecast", use_t3=True):
    from sklearn.ensemble import HistGradientBoostingRegressor

    parts = [t1_lag_features(F)]
    if use_t3:
        parts.append(t3_features(F, mode))
    X = pd.concat(parts, axis=1)
    Xv = X.to_numpy(float)          # HistGBM 은 NaN 을 그대로 처리한다

    def fit_predict(tr, te):
        m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=SEED)
        ok = np.isfinite(y[tr])
        if ok.sum() < 50:
            return np.full(len(te), np.nan)
        m.fit(Xv[tr][ok], y[tr][ok])
        return m.predict(Xv[te])

    fit_predict.feature_names = list(X.columns)
    fit_predict.X = Xv
    return fit_predict


# ---------------------------------------------------------------------------
# M5 — 상태공간 (국소수준 칼만): §5.3 의 '미설명 상수'를 느리게 변하는 상태로 추적
# ---------------------------------------------------------------------------

def local_level_filter(resid: np.ndarray, q_over_r: float = 0.002):
    """관측 = 상태 + 잡음, 상태 = 랜덤워크. 결측은 예측만 하고 갱신을 건너뛴다."""
    n = len(resid)
    x = np.zeros(n)
    obs = resid[np.isfinite(resid)]
    r_var = float(np.var(obs)) if len(obs) > 2 else 1.0
    q = q_over_r * r_var
    xh, p = (float(np.nanmean(obs)) if len(obs) else 0.0), r_var
    for i in range(n):
        p += q
        if np.isfinite(resid[i]):
            kg = p / (p + r_var)
            xh += kg * (resid[i] - xh)
            p *= (1 - kg)
        x[i] = xh
    return x


def make_M5(F, y, scen="표준", srt=SRT_REFERENCE_D, k_hyd=None, q_over_r=0.002):
    W = design_matrix(F, scen, srt, k_hyd or K_HYD, intercept=False, basis="tonnage")

    def fit_predict(tr, te):
        beta = nnls_fit(W, y, tr)
        base = W @ beta
        resid = np.where(np.isfinite(y), y - base, np.nan)
        r = resid.copy()
        r[te.min():] = np.nan                 # 검정 구간 정보 차단
        level = local_level_filter(r, q_over_r)
        return base[te] + level[te.min() - 1 if te.min() > 0 else 0]

    fit_predict.W = W
    return fit_predict


def m5_intercept_path(F, y, scen="표준", srt=SRT_REFERENCE_D, k_hyd=None, q_over_r=0.002):
    """시변 절편 궤적 (전기간 in-sample 진단용 — §12 산출물 3)."""
    W = design_matrix(F, scen, srt, k_hyd or K_HYD, intercept=False, basis="tonnage")
    idx = np.where(np.isfinite(y))[0]
    beta = nnls_fit(W, y, idx)
    resid = np.where(np.isfinite(y), y - W @ beta, np.nan)
    return local_level_filter(resid, q_over_r), beta


# ---------------------------------------------------------------------------
# 앙상블 — 단순평균 / 성능가중 / NNLS 스태킹 (§7)
# ---------------------------------------------------------------------------

def combine(oof: dict[str, np.ndarray], y: np.ndarray, folds, method: str) -> np.ndarray:
    """
    메타 학습기에도 rolling-origin 적용: 폴드 i 의 예측에는 폴드 <i 의 OOF 만 쓴다.
    누락 멤버가 있는 날은 남은 멤버로 가중치를 재정규화한다.
    """
    names = list(oof)
    P = np.column_stack([oof[k] for k in names])
    n = len(y)
    out = np.full(n, np.nan)

    for i, (_, te) in enumerate(folds):
        prev = np.concatenate([folds[j][1] for j in range(i)]) if i else np.array([], int)
        prev = prev[np.isfinite(y[prev])] if len(prev) else prev
        if method == "simple" or len(prev) < 30:
            w = np.ones(len(names))
        elif method == "perf":
            err = np.array([np.sqrt(np.nanmean((y[prev] - P[prev, j]) ** 2)) for j in range(len(names))])
            w = np.where(np.isfinite(err) & (err > 0), 1.0 / np.maximum(err, 1e-9) ** 2, 0.0)
        else:                                        # nnls 스태킹
            ok = np.isfinite(P[prev]).all(axis=1) & np.isfinite(y[prev])
            if ok.sum() < 30:
                w = np.ones(len(names))
            else:
                w = lsq_linear(P[prev][ok], y[prev][ok], bounds=(0.0, np.inf)).x
        if not np.isfinite(w).any() or w.sum() <= 0:
            w = np.ones(len(names))
        w = w / w.sum()
        blk = P[te]
        msk = np.isfinite(blk)
        wsum = (msk * w).sum(axis=1)
        vals = np.nansum(np.where(msk, blk, 0.0) * w, axis=1)
        out[te] = np.where(wsum > 0, vals / np.where(wsum == 0, np.nan, wsum), np.nan)
    return out


def final_weights(oof: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """보고용 최종 스태킹 가중치 (전 OOF 기준)."""
    names = list(oof)
    P = np.column_stack([oof[k] for k in names])
    ok = np.isfinite(P).all(axis=1) & np.isfinite(y)
    if ok.sum() < 30:
        return {}
    w = lsq_linear(P[ok], y[ok], bounds=(0.0, np.inf)).x
    return {n_: round(float(v), 4) for n_, v in zip(names, w)}
