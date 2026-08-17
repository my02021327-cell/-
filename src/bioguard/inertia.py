"""
관성 보정 예측 — 반응조의 물리적 관성을 무시하지 않되, 그것에만 기대지 않는다.

## 왜 이것이 `y_lag1 금지` 규칙과 충돌하지 않는가

저장소 규칙 1(명세서 W1)은 `y_lag1` 을 **피처로 회귀에 넣는 것**을 금지한다. 근거는
ACF(1)=0.916 계열에서 그렇게 하면 모델이 공정이 아니라 자기 자신을 설명하고, 원계열 R² 가
허위로 부풀기 때문이다. 그 지적은 옳다.

그러나 **반응조 관성 자체는 실재하는 물리**다. HRT 40.5일 · 일 교체율 2.47% 의 CSTR 은
하루 만에 상태가 바뀔 수 없다. 오늘 조 안의 상태를 알면 내일에 대해 진짜로 아는 것이 있다.
두 사실은 모순되지 않는다 — **관성을 어떻게 쓰느냐**가 다를 뿐이다.

여기서는 관성을 자유 회귀계수가 아니라 **감쇠하는 오차 되먹임**으로 쓴다:

    additive        ŷ(T+h) = ŝ(T+h) + r̄(T)·exp(−h/τ)
    multiplicative  ŷ(T+h) = ŝ(T+h) · ρ̄(T)^exp(−h/τ)

  ŝ  : 기질 기반 예측 (부하가 결정하는 정상상태)
  r̄  : 원점 T 직전 관측들의 잔차 평균 (additive)  /  ρ̄ : 실측/예측 비 (multiplicative)
  τ  : 관성 시간상수 — 학습셋에서 추정, 물리적으로는 1/(1/SRT+k) 근방

이 형태의 성질이 규칙 1 의 우려를 구조적으로 제거한다.

  · **h→0 이면 persistence 로 수렴**하고 **h→∞ 이면 순수 기질 모델로 수렴**한다.
    관성의 몫이 지평에 따라 자동으로 줄어들며, 그 감쇠율이 자유 파라미터가 아니라
    반응조 시간상수다.
  · 보정항은 **수준(level)이 아니라 잔차**에 붙는다. 따라서 계수를 부풀려 R² 를 만드는
    경로가 없다 — ŝ 가 나쁘면 r̄ 가 커질 뿐 성능은 좋아지지 않는다.
  · 90일 앞 예측에서 exp(−90/τ) ≈ 0 이므로 **장기 성능은 기질 모델이 그대로 책임진다.**
    관성이 장기 예측력을 대신 만들어주지 못한다.

경험식의 `C_base·MA30(t−1)` 은 같은 직관을 **수준에** 적용한 것이고, 그래서 예측의 63% 를
자기회귀항이 가져가 버렸다. 잔차에 붙이면 그 문제가 생기지 않는다.
"""

from __future__ import annotations

import numpy as np

# τ 는 자유 파라미터가 아니라 물리적 시간상수다. 상한은 CSTR 세척 한계
# 1/(1/SRT + k) — SRT 25일·k→0 에서 17.3일이 최장이다. 30일 같은 값은 물리 근거가 없다.
TAU_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0)
RESID_WINDOW = 7          # 원점 직전 잔차를 평균낼 관측일 수 (단일일 잡음 방지)
RHO_CLIP = (0.6, 1.6)     # 배수 보정의 물리적 안전 범위


def _recent(values: np.ndarray, end: int, window: int) -> np.ndarray:
    """end 이전(미포함)의 유한값 중 최근 window 개."""
    v = values[:end]
    v = v[np.isfinite(v)]
    return v[-window:] if len(v) else v


def correct(base: np.ndarray, y: np.ndarray, origin: int, idx: np.ndarray,
            tau: float, mode: str = "additive", window: int = RESID_WINDOW) -> np.ndarray:
    """
    origin 시점까지의 정보만 써서 idx 구간을 보정한다.
    idx 는 origin 이후의 절대 인덱스 배열이며, h = idx − origin + 1 로 지평을 잡는다.
    """
    b = np.asarray(base, float)
    if mode == "additive":
        r = _recent(y - b, origin, window)
        adj = float(r.mean()) if len(r) else 0.0
        decay = np.exp(-(idx - origin + 1) / float(tau))
        return b[idx] + adj * decay
    ratio = np.where(np.isfinite(b) & (b > 1e-6), y / b, np.nan)
    r = _recent(ratio, origin, window)
    rho = float(np.clip(r.mean(), *RHO_CLIP)) if len(r) else 1.0
    decay = np.exp(-(idx - origin + 1) / float(tau))
    return b[idx] * np.power(rho, decay)


WINDOW_GRID = (1, 3, 7, 14)


def make(base_factory, F, y, mode: str = "additive", tau_grid=TAU_GRID,
         window_grid=WINDOW_GRID):
    """
    base_factory(F, y) -> fit_predict : 보정 대상이 되는 기질 기반 모델.

    (τ, window) 를 **폴드 학습셋 안에서** 함께 고른다. 폴드 전체 성적으로 고르면 선택 편향이다.

    window 를 격자에 넣는 이유: window=1·τ 큰 값이면 보정이
    `ŷ(T+1) ≈ y(T) + [ŝ(T+1) − ŝ(T)]` 로 수렴하는데, 이것이 곧 설계문서의
    `persistence + Δfeed`(h=1 에서 R²=0.942) 다. 즉 이 틀은 **단기 최강 기준선을
    특수해로 포함**한다. window 를 7 로 고정하면 그 해에 닿지 못한다.
    """
    base_fp = base_factory(F, y)
    n = len(F)

    def fit_predict(tr, te):
        full = np.arange(n)
        base_all = base_fp(tr, full)                # 학습셋으로 적합, 전 구간 예측
        origin = int(te.min())

        # (τ, window) 선택: 학습 구간 끝 90일을 모의 검정창으로 사용
        best, best_score = (tau_grid[len(tau_grid) // 2], RESID_WINDOW), np.inf
        inner = tr[tr >= (tr[-1] + 1 if len(tr) else origin) - 90]
        inner = inner[np.isfinite(y[inner])]
        if len(inner) >= 20:
            io = int(inner.min())
            for t_ in tau_grid:
                for w_ in window_grid:
                    p = correct(base_all, y, io, inner, t_, mode, w_)
                    r = y[inner] - p
                    s = float(np.sqrt(np.nanmean(r ** 2)))
                    if np.isfinite(s) and s < best_score:
                        best, best_score = (t_, w_), s
        fit_predict.tau_history.append(best)
        return correct(base_all, y, origin, te, best[0], mode, best[1])

    fit_predict.tau_history = []
    return fit_predict


def horizon_buckets(folds, y, pred, base=None,
                    edges=((1, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90))):
    """지평 구간별 RMSE — 관성 보정의 몫이 어디서 사라지는지 보여준다."""
    out = []
    for lo, hi in edges:
        e_pred, e_base, e_pers = [], [], []
        for _, te in folds:
            org = int(te.min())
            h = te - org + 1
            sel = te[(h >= lo) & (h <= hi)]
            if len(sel) == 0:
                continue
            m = np.isfinite(y[sel]) & np.isfinite(pred[sel])
            if m.sum() < 2:
                continue
            e_pred.append((y[sel][m] - pred[sel][m]) ** 2)
            if base is not None:
                mb = np.isfinite(y[sel]) & np.isfinite(base[sel])
                if mb.sum() >= 2:
                    e_base.append((y[sel][mb] - base[sel][mb]) ** 2)
            hist = y[:org][np.isfinite(y[:org])]
            if len(hist):
                e_pers.append((y[sel][m] - hist[-1]) ** 2)
        row = {"지평_일": f"{lo}–{hi}",
               "n": int(sum(len(x) for x in e_pred)),
               "보정모델_RMSE": round(float(np.sqrt(np.concatenate(e_pred).mean())), 1)
               if e_pred else None}
        if e_base:
            row["기질모델_RMSE"] = round(float(np.sqrt(np.concatenate(e_base).mean())), 1)
        if e_pers:
            row["persistence_RMSE"] = round(float(np.sqrt(np.concatenate(e_pers).mean())), 1)
        if row.get("기질모델_RMSE") and row.get("보정모델_RMSE"):
            row["관성_이득_pct"] = round(
                100 * (1 - row["보정모델_RMSE"] / row["기질모델_RMSE"]), 1)
        out.append(row)
    return out
