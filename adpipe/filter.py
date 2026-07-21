"""STEP 3 — 유병률/존재비 필터 Transformer (PROMPT 1 §STEP 3).

fit()은 학습 fold 만 보고 통과 taxa 를 결정하고, transform()은 제거 후
행 합을 1로 재정규화(조성 폐쇄 유지)한다. 전역 필터링은 명백한 누수이므로 금지.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import to_proportion


class PrevalenceAbundanceFilter(BaseEstimator, TransformerMixin):
    """유병률·평균존재비 기반 taxa 필터.

    파라미터
    --------
    min_prevalence : float   학습 fold 시료의 이 비율 이상에서 검출되어야 통과 (기본 0.25)
    min_mean_ra    : float   학습 fold 평균 RA(비율) 하한 (기본 0.001 = 0.1%)
    detection_thresh : float 검출로 볼 최소값 (기본 0.0, 초과시 검출)
    keep_always    : list    도메인상 반드시 보존할 taxa
    mode           : {'drop','lump'} 'lump'이면 제거분을 'Other' 열로 합산(조성적으로 더 정직)
    renormalize    : bool    transform 후 행합 1 재정규화 (기본 True)

    학습 속성
    --------
    kept_            : 통과 taxa 목록
    residual_fraction_ : 제거된 taxa 의 (학습 fold) 총 RA 평균 비율. >0.15 이면 경고.
    """

    def __init__(self, min_prevalence=0.25, min_mean_ra=0.001, detection_thresh=0.0,
                 keep_always=None, mode="drop", renormalize=True):
        self.min_prevalence = min_prevalence
        self.min_mean_ra = min_mean_ra
        self.detection_thresh = detection_thresh
        self.keep_always = keep_always
        self.mode = mode
        self.renormalize = renormalize

    def fit(self, X, y=None):
        Xdf = self._as_df(X)
        P = to_proportion(Xdf)                    # 스케일 무관 비율로 판정
        keep_always = set(self.keep_always or [])

        prevalence = (P > self.detection_thresh).mean(axis=0)
        mean_ra = P.mean(axis=0)
        passed = (prevalence >= self.min_prevalence) & (mean_ra >= self.min_mean_ra)
        passed = passed | P.columns.isin(keep_always)

        self.feature_names_in_ = np.asarray(Xdf.columns)
        self.kept_ = list(P.columns[passed])
        removed = [c for c in P.columns if c not in set(self.kept_)]
        self.removed_ = removed
        self.residual_fraction_ = float(P[removed].sum(axis=1).mean()) if removed else 0.0
        if self.residual_fraction_ > 0.15:
            warnings.warn(
                f"[PrevalenceAbundanceFilter] residual_fraction_="
                f"{self.residual_fraction_:.3f} > 0.15 — 필터가 과도할 수 있음.")
        if not self.kept_:
            raise ValueError("필터 결과 통과 taxa 가 0 개입니다. 임계값을 낮추세요.")
        return self

    def transform(self, X):
        Xdf = self._as_df(X)
        # 학습되지 않은 열은 0 으로 정렬 (테스트에 새 taxa 가 있어도 안전)
        Xk = Xdf.reindex(columns=self.kept_, fill_value=0.0)
        if self.mode == "lump":
            other = Xdf.drop(columns=[c for c in self.kept_ if c in Xdf.columns],
                             errors="ignore").sum(axis=1)
            Xk = Xk.copy()
            Xk["Other__lumped"] = other.values
        if self.renormalize:
            Xk = to_proportion(Xk)
        return Xk

    def get_feature_names_out(self, input_features=None):
        names = list(self.kept_)
        if self.mode == "lump":
            names = names + ["Other__lumped"]
        return np.asarray(names, dtype=object)

    @staticmethod
    def _as_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X, dtype=float))
