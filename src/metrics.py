"""
정규화 성능지표 — RMSSE 와 naive(persistence) 기준선

Meola & Weinrich (2025) Applied Energy 390:125781 은 전규모 혐기소화 예측의 주지표로
RMSSE(Root Mean Squared Scaled Error)를 쓴다. 정의는 「모델 RMSE ÷ naive 예측 RMSE」이고,
naive 는 다음 시점 출력을 현재 출력으로 예측하는 persistence 다.

    RMSSE = RMSE(model) / RMSE(naive)          (논문은 % 로 표기)

의미:
  RMSSE < 100 %  → 전일값 하나만 쓰는 예측기보다 낫다 (채택 가능)
  RMSSE ≥ 100 %  → 베이스라인 미달. 절대 R² 가 아무리 높아도 쓸 수 없다.

이 지표가 필요한 이유는 저장소 자체 이력이 증명한다. v1 파이프라인의 2023 홀드아웃
R² 0.588(RandomForest)은 그럴듯해 보이지만, 같은 구간 persistence RMSE 가 383.5 ㎥/d 라
RMSSE 는 183 % 다 — 논문 기준으로는 보고 대상이 아니라 기각 대상이다.
(EXPERT_REVIEW §A3 「정규화 성능 지표의 공백」이 지적한 바로 그 공백)

■ 결측 구간 취급
  영천 자료는 CH₄ 농도 결측 때문에 타깃 관측일이 불규칙하다(1,264/2,190일). naive 는
  '직전 관측일의 값'으로 정의하고, 그 간격(gap)을 함께 기록한다. 간격이 벌어질수록
  naive 는 불리해지므로 RMSSE 는 낙관적으로 편향된다. 따라서 gap=1 부분집합에 대한
  RMSSE 를 항상 병기한다 — 이쪽이 논문의 정의(연속 timestep)에 정확히 대응한다.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(np.ravel(y_true), np.ravel(y_pred))))


def naive_reference(y_prev) -> np.ndarray:
    """naive(persistence) 예측 = 직전 관측값."""
    return np.asarray(y_prev, dtype=float).ravel()


def score(y_true, y_pred, y_prev=None, gap=None) -> dict:
    """
    표준 지표 묶음. y_prev 가 주어지면 RMSSE 와 naive 기준선을 함께 낸다.

    gap 이 주어지면 gap==1(연속 관측일) 부분집합의 RMSSE 를 병기한다.
    """
    yt, yp = np.ravel(np.asarray(y_true, float)), np.ravel(np.asarray(y_pred, float))
    out = {
        "R2": round(float(r2_score(yt, yp)), 4),
        "RMSE": round(rmse(yt, yp), 2),
        "MAE": round(float(mean_absolute_error(yt, yp)), 2),
        "MAPE": round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 2),
        "n": int(yt.size),
    }
    if y_prev is None:
        return out

    yn = naive_reference(y_prev)
    naive_rmse = rmse(yt, yn)
    out["naive_RMSE"] = round(naive_rmse, 2)
    out["RMSSE_pct"] = round(100.0 * rmse(yt, yp) / naive_rmse, 1) if naive_rmse > 0 else None

    if gap is not None:
        g = np.ravel(np.asarray(gap, float))
        m = g == 1
        if m.sum() >= 5:
            nr1 = rmse(yt[m], yn[m])
            out["RMSSE_pct_gap1"] = round(100.0 * rmse(yt[m], yp[m]) / nr1, 1) if nr1 > 0 else None
            out["n_gap1"] = int(m.sum())
    return out


def paired_t_test(errors_a, errors_b) -> dict:
    """
    폴드별 대응 t검정. a − b 가 음수면 a 가 개선.
    평균 지표만 비교하고 '개선됐다'고 쓰지 않기 위한 필수 절차(PROMPT §8-4).
    """
    from scipy import stats

    a, b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    d = a - b
    if d.size < 2 or np.allclose(d, 0):
        return {"delta": round(float(d.mean()), 2), "p": None, "n_folds": int(d.size)}
    t, p = stats.ttest_rel(a, b)
    return {
        "delta": round(float(d.mean()), 2),
        "se": round(float(d.std(ddof=1) / np.sqrt(d.size)), 2),
        "t": round(float(t), 3),
        "p": round(float(p), 4),
        "n_folds": int(d.size),
    }
