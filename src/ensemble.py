"""
앙상블 모델 : 이질적 Base Model + NNLS 비음수 스태킹 Meta Model

보고서 구성
 - Base  : RandomForest, MLP, LSTM, Transformer
 - Meta  : NNLS(Non-Negative Least Squares) 비음수 제약 선형결합

본 구현 노트
 - 실행 환경에 딥러닝 프레임워크(torch/tensorflow)를 설치할 수 없어(프록시 차단),
   LSTM/Transformer 는 rolling/lag 시계열 피처를 입력받는 sklearn 계열의
   이질적 학습기(HistGradientBoosting, ExtraTrees)로 대체하였다.
   -> Base Model 다양성 확보와 NNLS 비음수 스태킹 방법론은 보고서와 동일하다.

메타러너 학습 전략(핵심)
 - 단순 K-fold OOF 로 NNLS 가중치를 학습하면, 학습기간(2018~2022) 내부 성능 순위가
   2023 국면이동 홀드아웃 순위와 달라 특정 base 를 과대가중하는 문제가 발생한다.
 - 따라서 학습기간에서 '가장 최근 연도' 를 메타 검증 블록으로 분리하여
   (base 는 그 이전 데이터로 학습) 미래 일반화에 맞춘 비음수 가중치를 학습한 뒤,
   최종적으로 base 를 전체 학습데이터로 재학습하여 홀드아웃을 예측한다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_base_models() -> dict:
    """이질적 Base Model 사전 반환. 모든 모델은 다중출력(methane, MY)을 지원."""
    return {
        # Random Forest : 보고서 최고 단일모델 계열
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # MLP(신경망) : 입력 표준화 + 타깃 표준화로 다중출력 스케일 불균형 방지
        "MLP": TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            max_iter=2000,
                            early_stopping=True,
                            n_iter_no_change=30,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            transformer=StandardScaler(),
        ),
        # HistGradientBoosting : LSTM 대체(시계열 피처 기반 부스팅)
        "HistGBM": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
        # ExtraTrees : Transformer 대체(고분산·저편향 이질 학습기)
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


class NNLSStackingEnsemble:
    """
    NNLS 비음수 스태킹 앙상블 (다중출력).

    fit(X, Y, meta_mask)
      - meta_mask=True 인 행 = 메타 검증 블록(가장 최근 연도).
      - base 를 meta_mask=False 데이터로 학습 → 메타 블록 예측 → 종속변수별 NNLS 가중치.
      - 이후 base 를 전체 (X, Y) 로 재학습 → 최종 예측에 사용.
    predict(X) -> (n, n_targets) 앙상블 예측
    """

    def __init__(self, base_models: dict | None = None):
        self.base_protos = base_models or make_base_models()
        self.names = list(self.base_protos)
        self.fitted_: dict = {}
        self.weights_: list = []      # 종속변수별 {name: weight}
        self.meta_base_pred_: dict = {}

    def fit(self, X: np.ndarray, Y: np.ndarray, meta_mask: np.ndarray):
        Y = np.asarray(Y, dtype=float)
        n_targets = Y.shape[1]
        early = ~meta_mask

        # 1) base 를 메타블록 '이전' 데이터로 학습 → 메타블록 예측
        meta_pred = {}
        for name, proto in self.base_protos.items():
            m = clone(proto)
            m.fit(X[early], Y[early])
            meta_pred[name] = np.atleast_2d(m.predict(X[meta_mask]))
            if meta_pred[name].shape[0] == 1 and n_targets > 1:
                meta_pred[name] = meta_pred[name].reshape(-1, n_targets)
        self.meta_base_pred_ = meta_pred

        # 2) 종속변수별 NNLS 비음수 가중치
        self.weights_ = []
        for j in range(n_targets):
            P = np.column_stack([meta_pred[n][:, j] for n in self.names])
            w, _ = nnls(P, Y[meta_mask, j])
            self.weights_.append(dict(zip(self.names, w)))

        # 3) base 전체 학습데이터로 재학습 (최종 예측용)
        self.fitted_ = {}
        for name, proto in self.base_protos.items():
            m = clone(proto)
            m.fit(X, Y)
            self.fitted_[name] = m
        return self

    def base_predict(self, X: np.ndarray) -> dict:
        """개별 base 모델 예측 반환(진단·비교용)."""
        return {name: mdl.predict(X) for name, mdl in self.fitted_.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        base = self.base_predict(X)
        n = X.shape[0]
        n_targets = len(self.weights_)
        out = np.zeros((n, n_targets))
        for j in range(n_targets):
            w = self.weights_[j]
            P = np.column_stack([base[nm][:, j] for nm in self.names])
            out[:, j] = P @ np.array([w[nm] for nm in self.names])
        return out

    def normalized_weights(self) -> list:
        """정규화된(합=1) 가중치 리포트."""
        rep = []
        for w in self.weights_:
            s = sum(w.values()) or 1.0
            rep.append({k: round(float(v / s), 4) for k, v in w.items()})
        return rep
