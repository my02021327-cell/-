"""
앙상블 : 이질적 Base Model(RF·MLP·LSTM·Transformer) + NNLS 비음수 스태킹 Meta

보고서 구성과 동일
 - Base  : RandomForest, MLP, LSTM, Transformer
 - Meta  : NNLS(Non-Negative Least Squares) 비음수 제약 선형결합

입력 표현이 다른 base(테이블형 RF·MLP vs 시퀀스형 LSTM·Transformer)를 함께 쓰기 위해
본 모듈의 메타 결합기는 '각 base 의 예측 행렬'만 입력받는 범용 구조로 설계한다.
base 학습·예측 오케스트레이션은 train.py 가 담당한다.

메타러너 학습 전략
 - 학습기간(2018~2022) 내부 K-fold OOF 로 가중치를 학습하면 2023 국면이동 홀드아웃
   순위와 달라 특정 base 를 과대가중한다. 이를 막기 위해 학습기간의 '가장 최근 연도
   (2022)' 를 메타 검증 블록으로 두고(그 이전 데이터로 base 학습), 미래 일반화에 맞춘
   비음수 가중치를 학습한 뒤 base 를 전체 학습데이터로 재학습하여 홀드아웃을 예측한다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_tabular_models() -> dict:
    """테이블형 Base Model (rolling/lag 피처 입력, 다중출력)."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
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
    }


class NNLSMetaCombiner:
    """
    NNLS 비음수 스태킹 메타 결합기 (다중출력).

    fit(meta_preds, Y_meta)
      meta_preds : dict[name] -> (n_meta, k)  각 base 의 메타 검증블록 예측
      Y_meta     : (n_meta, k)                메타 검증블록 실측
      -> 종속변수별 비음수 가중치 self.weights_ 학습
    combine(test_preds) -> (n, k) 앙상블 예측
    """

    def __init__(self):
        self.names: list = []
        self.weights_: list = []   # 종속변수별 {name: weight}

    def fit(self, meta_preds: dict, Y_meta: np.ndarray):
        self.names = list(meta_preds)
        Y_meta = np.asarray(Y_meta, dtype=float)
        n_targets = Y_meta.shape[1]
        self.weights_ = []
        for j in range(n_targets):
            P = np.column_stack([meta_preds[n][:, j] for n in self.names])
            w, _ = nnls(P, Y_meta[:, j])
            self.weights_.append(dict(zip(self.names, w)))
        return self

    def combine(self, test_preds: dict) -> np.ndarray:
        n = next(iter(test_preds.values())).shape[0]
        n_targets = len(self.weights_)
        out = np.zeros((n, n_targets))
        for j in range(n_targets):
            w = self.weights_[j]
            P = np.column_stack([test_preds[nm][:, j] for nm in self.names])
            out[:, j] = P @ np.array([w[nm] for nm in self.names])
        return out

    def normalized_weights(self) -> list:
        rep = []
        for w in self.weights_:
            s = sum(w.values()) or 1.0
            rep.append({k: round(float(v / s), 4) for k, v in w.items()})
        return rep
