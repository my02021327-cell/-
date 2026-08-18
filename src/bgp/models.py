"""
모델 라인업 — 통계 · 기계학습 · 시계열 심층학습

선정 근거는 Meola & Weinrich (2025) 의 전규모 AD 비교 결과다.
  · 권고    RF, RNN(LSTM/GRU)  — 공정 상태가 불확실하거나 비정상일 때 가장 강건
  · 조건부  EN / BR            — 정상상태에서 충분하나 검증-시험 격차가 크다
  · 대안    GBR, k-NN          — 학습시간이 중요할 때
  · 비권고  PLS, GPR           — 제외
  · 미채택  MLP, Transformer   — 논문이 후보 단계에서 배제(시계열에서 RNN 열위 /
                                 학습시간·하이퍼파라미터 불안정)
여기에 SARIMAX 를 더한다. 논문에서 SARIMAX 는 데이터셋 B 의 24 h OD 에서 최고 성능
(RMSSE 97 %)을 낸 유일한 모델이었고, 우리 자료도 유량 계열이 완전 관측이라 조건이 맞다.

■ 직접 다단계(direct multi-step)
  모든 회귀 모델은 「원점 t 의 피처 + 지평 h」로 y(t+h) 를 직접 맞힌다. 재귀 예측처럼
  오차가 누적되지 않고, 지평 구간 안에서 표본을 공유한다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.bgp import config as C

RS = C.RANDOM_STATE


def _p(est, scale=True, impute=True):
    steps = []
    if impute:
        steps.append(("imp", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("est", est))
    return Pipeline(steps)


def model_zoo(fast: bool = False) -> dict:
    """이름 → 후보 리스트. 후보는 폴드 학습셋 내부에서만 고른다."""
    zoo = {
        # HistGBM 은 결측을 네이티브로 처리한다 — 결측 39~50 % 자료에서 큰 이점
        "HistGBM": [
            _p(HistGradientBoostingRegressor(max_iter=400, learning_rate=lr,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             early_stopping=True, random_state=RS),
               scale=False, impute=False)
            for lr in (0.05, 0.1)],
        "RandomForest": [
            _p(RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     max_features=0.3, n_jobs=-1, random_state=RS),
               scale=False)],
        "ExtraTrees": [
            _p(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.5,
                                   n_jobs=-1, random_state=RS), scale=False)],
        "ElasticNet": [
            _p(ElasticNet(alpha=a, l1_ratio=l, max_iter=20000))
            for a in (0.1, 1.0, 10.0) for l in (0.2, 0.8)],
        "BayesianRidge": [_p(BayesianRidge())],
        "kNN": [_p(KNeighborsRegressor(n_neighbors=k, weights="distance"))
                for k in (10, 25, 50)],
    }
    if not fast:
        zoo["AdaBoost"] = [_p(AdaBoostRegressor(n_estimators=200, learning_rate=lr,
                                                random_state=RS), scale=False)
                           for lr in (0.3, 1.0)]
    return zoo


# ══════════════════════════════════════════════════════════════════════════════
# 기준선 — 이걸 못 이기면 모델을 채택하지 않는다
# ══════════════════════════════════════════════════════════════════════════════
class PersistenceBand(BaseEstimator, RegressorMixin):
    """원점의 마지막 관측 메탄을 그대로 미는 naive. RMSSE 의 분모."""

    def __init__(self, anchor_col: str = "tmp__y_last"):
        self.anchor_col = anchor_col

    def fit(self, X, y=None):
        self.anchor_ = float(np.nanmedian(np.asarray(y, float))) if y is not None else 0.0
        return self

    def predict(self, X):
        v = np.asarray(X[self.anchor_col] if isinstance(X, pd.DataFrame) else X[:, 0], float)
        return np.where(np.isfinite(v), v, self.anchor_)


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """작년 같은 시기 수준 + 최근 편차. 장기 지평에서 persistence 보다 낫다."""

    def __init__(self, ma_col: str = "tmp__y_ma30"):
        self.ma_col = ma_col

    def fit(self, X, y=None):
        self.mean_ = float(np.nanmean(np.asarray(y, float))) if y is not None else 0.0
        return self

    def predict(self, X):
        v = np.asarray(X[self.ma_col], float)
        return np.where(np.isfinite(v), v, self.mean_)


# ══════════════════════════════════════════════════════════════════════════════
# 시계열 심층학습 — GRU (논문 권고 RNN)
# ══════════════════════════════════════════════════════════════════════════════
class GRUForecaster(BaseEstimator, RegressorMixin):
    """
    과거 W일 다변량 시퀀스 + 지평 h → y(t+h) 직접 예측.

    논문이 RNN 을 권고한 근거는 「복잡한 입출력 관계를 담고, 상태가 미지일 때 강건」이다.
    여기서는 sklearn 인터페이스(fit/predict)로 감싸 다른 모델과 **같은 폴드·같은 선택
    절차**를 통과하게 한다. 이기지 못하면 그대로 탈락시킨다.
    """

    def __init__(self, seq_cols: list[str] | None = None, window: int = 30,
                 hidden: int = 48, layers: int = 1, epochs: int = 40,
                 lr: float = 3e-3, batch: int = 256, patience: int = 6, seed: int = RS):
        self.seq_cols = seq_cols
        self.window = window
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.patience = patience
        self.seed = seed

    # 시퀀스는 X 프레임이 아니라 원본 패널에서 만든다 → set_panel 로 주입
    def set_panel(self, panel: np.ndarray):
        self.panel_ = np.asarray(panel, np.float32)
        return self

    def _windows(self, origins: np.ndarray) -> np.ndarray:
        W, P = self.window, self.panel_
        out = np.empty((len(origins), W, P.shape[1]), np.float32)
        for i, t in enumerate(origins):
            lo = t - W + 1
            if lo < 0:
                blk = np.vstack([np.repeat(P[[0]], -lo, axis=0), P[: t + 1]])
            else:
                blk = P[lo : t + 1]
            out[i] = blk
        return out

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        origins = np.asarray(X["__origin__"], int)
        h = np.asarray(X["__h__"], np.float32)
        S = self._windows(origins)
        self.mu_, self.sd_ = np.nanmean(S, (0, 1)), np.nanstd(S, (0, 1)) + 1e-6
        S = np.nan_to_num((S - self.mu_) / self.sd_)
        y = np.asarray(y, np.float32)
        self.ymu_, self.ysd_ = float(y.mean()), float(y.std() + 1e-6)

        n = len(y)
        cut = max(1, int(n * 0.85))
        idx = np.arange(n)
        tr, va = idx[:cut], idx[cut:]

        class Net(nn.Module):
            def __init__(s, nf, hid, nl):
                super().__init__()
                s.gru = nn.GRU(nf, hid, num_layers=nl, batch_first=True)
                s.head = nn.Sequential(nn.Linear(hid + 1, hid), nn.ReLU(), nn.Linear(hid, 1))

            def forward(s, x, hh):
                o, _ = s.gru(x)
                return s.head(torch.cat([o[:, -1, :], hh], 1)).squeeze(-1)

        net = Net(S.shape[2], self.hidden, self.layers).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = nn.SmoothL1Loss()

        St = torch.tensor(S, device=dev)
        ht = torch.tensor((h / 90.0).reshape(-1, 1), device=dev)
        yt = torch.tensor((y - self.ymu_) / self.ysd_, device=dev)

        best, bad, best_state = np.inf, 0, None
        for _ in range(self.epochs):
            net.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), self.batch):
                b = perm[i : i + self.batch]
                opt.zero_grad()
                loss = lossf(net(St[b], ht[b]), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if len(va) == 0:
                continue
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(St[va], ht[va]), yt[va]))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            net.load_state_dict(best_state)
        self.net_, self.dev_ = net.eval(), dev
        return self

    def predict(self, X):
        import torch

        origins = np.asarray(X["__origin__"], int)
        h = np.asarray(X["__h__"], np.float32)
        S = np.nan_to_num((self._windows(origins) - self.mu_) / self.sd_)
        with torch.no_grad():
            p = self.net_(torch.tensor(S, device=self.dev_),
                          torch.tensor((h / 90.0).reshape(-1, 1), device=self.dev_))
        return p.cpu().numpy() * self.ysd_ + self.ymu_


# ══════════════════════════════════════════════════════════════════════════════
# 시계열 통계 — SARIMAX (외생 투입량)
# ══════════════════════════════════════════════════════════════════════════════
class SARIMAXBand:
    """
    유량 계열을 SARIMAX 로 예측하고 농도를 곱해 메탄으로 되돌린다.

    유량은 결측 0 % 라 순수 시계열 모형을 적용할 수 있는 유일한 계열이다.
    외생변수로 투입 물량을 넣어 「기질이 들어와야 가스가 난다」는 인과를 반영한다.
    """

    def __init__(self, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit_forecast(self, flow: pd.Series, exog: pd.DataFrame,
                     origin: int, steps: int) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = flow.iloc[: origin + 1].astype(float)
        ex = exog.iloc[: origin + 1].astype(float)
        ex_f = exog.iloc[origin + 1 : origin + 1 + steps].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = SARIMAX(y, exog=ex, order=self.order,
                              seasonal_order=self.seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, maxiter=60)
                fc = res.forecast(steps=len(ex_f), exog=ex_f)
                return np.asarray(fc, float)
            except Exception:
                return np.full(len(ex_f), float(y.iloc[-1]))


def nnls_stack(pred_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """비음수 스태킹 가중치. 실패한 멤버를 자동으로 0 으로 배제한다."""
    from scipy.optimize import nnls

    w, _ = nnls(pred_matrix, y)
    return w
