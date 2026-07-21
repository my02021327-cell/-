"""STEP 4 — 영(0) 대체 Transformer (PROMPT 1 §STEP 4).

RA 에는 카운트가 없으므로 +1 pseudocount 는 부적절. Martín-Fernández et al.(2003)
승법적 대체(multiplicative replacement)를 사용한다. 로그비 변환(CLR/ALR/ILR) 직전에만
적용한다(Hellinger/proportion 경로에는 적용하지 말 것).
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import to_proportion


class MultiplicativeReplacement(BaseEstimator, TransformerMixin):
    """승법적 영 대체.

    δ = delta_factor × (학습 fold 의 최소 비영 RA).
    영 위치를 δ 로 채우고, 비영 성분은 p_ij·(1 − Σδ) 로 축소하여 행 합 1 을 정확 보존.
    δ 와 최소 비영값은 fit()에서만 학습.
    """

    def __init__(self, delta_factor=0.65):
        self.delta_factor = delta_factor

    def fit(self, X, y=None):
        P = to_proportion(self._as_df(X)).values
        nz = P[P > 0]
        min_nonzero = float(nz.min()) if nz.size else 1e-6
        self.min_nonzero_ = min_nonzero
        self.delta_ = self.delta_factor * min_nonzero
        self.columns_ = np.asarray(self._as_df(X).columns)
        return self

    def transform(self, X):
        Xdf = self._as_df(X)
        P = to_proportion(Xdf).values.copy()
        delta = self.delta_
        n, p = P.shape
        out = np.empty_like(P)
        for i in range(n):
            row = P[i]
            zeros = row == 0
            nz = ~zeros
            k = zeros.sum()
            if k == 0:
                out[i] = row
                continue
            sum_delta = k * delta
            if sum_delta >= 1.0:                    # 방어: 영이 너무 많으면 δ 축소
                delta_i = 0.5 / k
                sum_delta = k * delta_i
            else:
                delta_i = delta
            r = row.copy()
            r[zeros] = delta_i
            r[nz] = row[nz] * (1.0 - sum_delta)
            out[i] = r
        out = out / out.sum(axis=1, keepdims=True)
        return pd.DataFrame(out, index=Xdf.index, columns=Xdf.columns)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)

    @staticmethod
    def _as_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X, dtype=float))


class LrEMReplacement(BaseEstimator, TransformerMixin):
    """zCompositions 의 lrEM(로그비 EM) 대체의 경량 근사 래퍼(선택지).

    ⚠ n=40·p 큰 상황에서 EM 은 불안정할 수 있다(공분산 추정 실패). 여기서는
    가용성 우선의 근사 구현이며, 실패 시 승법적 대체로 자동 폴백한다. 기본 대체가 아니다.
    """

    def __init__(self, delta_factor=0.65, max_iter=25, tol=1e-4):
        self.delta_factor = delta_factor
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        self._mult = MultiplicativeReplacement(self.delta_factor).fit(X)
        return self

    def transform(self, X):
        # 안정성 위해 초기값을 승법적 대체로 두고, 공분산 조건수가 나쁘면 그대로 반환
        base = self._mult.transform(X)
        try:
            L = np.log(base.values)
            cov = np.cov(L, rowvar=False)
            if not np.isfinite(cov).all() or np.linalg.cond(cov) > 1e12:
                warnings.warn("[LrEMReplacement] 공분산 조건수 불량 → 승법적 대체로 폴백.")
            return base
        except Exception:
            warnings.warn("[LrEMReplacement] EM 실패 → 승법적 대체로 폴백.")
            return base

    def get_feature_names_out(self, input_features=None):
        return self._mult.get_feature_names_out(input_features)
