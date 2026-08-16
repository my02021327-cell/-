"""
검증 프로토콜 — PROMPT §8. 위반하면 결과 전체가 무효인 규칙들.

  1. 확장창 rolling-origin: 초기 학습 365일, 예측창 90일, 원점 90일 전진
  2. 학습셋은 항상 검정점 이전 관측만 포함
  3. 무작위 k-fold 금지
  4. 모델 비교는 폴드별 대응 t검정 — ΔRMSE·SE·p 병기. 평균만 비교 금지
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .config import CV_HORIZON_D, CV_INITIAL_TRAIN_D, CV_STEP_D

MIN_TRAIN_OBS = 200
MIN_TEST_OBS = 20


def make_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """확장창 폴드. 학습 인덱스는 항상 검정 시작 이전이다."""
    return [(np.arange(0, st), np.arange(st, min(st + CV_HORIZON_D, n)))
            for st in range(CV_INITIAL_TRAIN_D, n - CV_HORIZON_D, CV_STEP_D)]


def valid_folds(folds, y: np.ndarray):
    """관측 라벨이 충분한 폴드만. 모든 모델이 **동일한 폴드**를 쓰도록 한 번만 정한다."""
    out = []
    for tr, te in folds:
        tr_o = tr[np.isfinite(y[tr])]
        te_o = te[np.isfinite(y[te])]
        if len(tr_o) >= MIN_TRAIN_OBS and len(te_o) >= MIN_TEST_OBS:
            out.append((tr, te))
    return out


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    r = np.asarray(y, float) - np.asarray(p, float)
    return float(np.sqrt(r @ r / len(r)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return {"n": int(len(y)), "RMSE": None, "MAPE": None, "R2": None}
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": int(len(y)),
            "RMSE": round(rmse(y, p), 1),
            "MAPE": round(float(np.mean(np.abs((y - p) / np.where(y == 0, np.nan, y))) * 100), 2),
            "R2": round(float(1 - ((y - p) ** 2).sum() / ss), 4) if ss > 0 else None}


def paired_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    """폴드별 대응 t검정. a − b 가 음수면 a 가 낫다(RMSE 기준)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return {"비교": f"{label_a} vs {label_b}", "n_folds": int(len(a)), "판정": "폴드 부족"}
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return {"비교": f"{label_a} vs {label_b}", "n_folds": int(len(a)),
            f"{label_a}_RMSE": round(float(a.mean()), 1),
            f"{label_b}_RMSE": round(float(b.mean()), 1),
            "ΔRMSE": round(float(d.mean()), 2),
            "SE": round(float(d.std(ddof=1) / np.sqrt(len(d))), 2),
            "t": round(float(t), 3), "p": round(float(p), 4),
            "유의": bool(p < 0.05),
            "판정": ("개선(유의)" if (p < 0.05 and d.mean() < 0) else
                    "악화(유의)" if (p < 0.05 and d.mean() > 0) else "구별되지 않음")}


def fold_rmse(fit_predict, folds, y: np.ndarray) -> np.ndarray:
    """폴드별 RMSE 벡터. fit_predict(tr_idx, te_idx) -> te 예측값."""
    out = []
    for tr, te in folds:
        tr_o = tr[np.isfinite(y[tr])]
        te_o = te[np.isfinite(y[te])]
        pred = np.asarray(fit_predict(tr_o, te_o), float)
        ok = np.isfinite(pred)
        out.append(rmse(y[te_o][ok], pred[ok]) if ok.sum() >= 2 else np.nan)
    return np.array(out)


def oof_predictions(fit_predict, folds, y: np.ndarray, n: int) -> np.ndarray:
    """out-of-fold 예측 벡터(길이 n). 메타 학습기 입력용 — 누출 방지(§7)."""
    p = np.full(n, np.nan)
    for tr, te in folds:
        tr_o = tr[np.isfinite(y[tr])]
        te_o = te[np.isfinite(y[te])]
        p[te_o] = fit_predict(tr_o, te_o)
    return p
