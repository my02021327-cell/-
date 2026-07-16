"""
LSTM · Transformer 시퀀스 회귀 모델 (PyTorch, 다중출력)

보고서 사양
 - LSTM        : 2 Layer(hidden 64), 시퀀스 W=14
 - Transformer : 2층 인코더 (d_model=64, heads=4, FF=128)

두 모델 모두 sklearn 유사 인터페이스(fit/predict)를 제공하며,
입력(피처)·출력(타깃) 표준화를 내부에서 수행한다. CPU 학습을 가정한다.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class _StandardScaler3D:
    """시퀀스 피처 표준화 (마지막 축=피처 기준)."""

    def fit(self, X):  # X: (n, W, F)
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class _LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2, n_out=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # 마지막 타임스텝


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, n_feat, d_model=64, heads=4, ff=128, n_layers=2, n_out=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_out))

    def forward(self, x):
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))  # 시퀀스 평균 풀링


class _TorchSeqRegressor:
    """LSTM/Transformer 공통 학습 래퍼 (내부 피처·타깃 표준화, 조기종료)."""

    def __init__(self, net_factory, epochs=400, lr=5e-4, batch_size=64,
                 patience=25, val_frac=0.15, seed=42, weight_decay=1e-3,
                 grad_clip=0.5, n_seeds=3):
        self.net_factory = net_factory
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.n_seeds = n_seeds  # 시드 평균 앙상블(학습 분산 감소)

    def _fit_one(self, Xs, Ys, seed):
        set_seed(seed)
        n = len(Xs)
        n_val = max(1, int(n * self.val_frac))
        tr_idx, va_idx = np.arange(0, n - n_val), np.arange(n - n_val, n)
        tr_dl = DataLoader(
            TensorDataset(torch.tensor(Xs[tr_idx]), torch.tensor(Ys[tr_idx])),
            batch_size=self.batch_size, shuffle=True,
        )
        Xva = torch.tensor(Xs[va_idx]).to(DEVICE)
        Yva = torch.tensor(Ys[va_idx]).to(DEVICE)

        net = self.net_factory(Xs.shape[-1]).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val, best_state, wait = float("inf"), None, 0
        for _ in range(self.epochs):
            net.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = loss_fn(net(Xva), Yva).item()
            if vloss < best_val - 1e-5:
                best_val, best_state, wait = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        self.xscaler = _StandardScaler3D().fit(X)
        self.y_mean_ = Y.mean(axis=0)
        self.y_std_ = Y.std(axis=0) + 1e-8
        Xs = self.xscaler.transform(X)
        Ys = (Y - self.y_mean_) / self.y_std_
        self.nets = [self._fit_one(Xs, Ys, self.seed + k) for k in range(self.n_seeds)]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.tensor(self.xscaler.transform(X)).to(DEVICE)
        preds = []
        with torch.no_grad():
            for net in self.nets:
                preds.append(net(Xt).cpu().numpy())
        return np.mean(preds, axis=0) * self.y_std_ + self.y_mean_


def make_lstm(**kw) -> _TorchSeqRegressor:
    # 2 Layer LSTM. hidden=32 가 소규모·국면이동 데이터에서 hidden=64 보다
    # 안정적으로 일반화되어 기본값으로 채택(보고서의 2 Layer 구조는 동일).
    return _TorchSeqRegressor(lambda nf: _LSTMNet(nf, hidden=32, n_layers=2), **kw)


def make_lstm_wide(**kw) -> _TorchSeqRegressor:
    return _TorchSeqRegressor(lambda nf: _LSTMNet(nf, hidden=64, n_layers=2), **kw)


def make_transformer(**kw) -> _TorchSeqRegressor:
    return _TorchSeqRegressor(
        lambda nf: _TransformerNet(nf, d_model=64, heads=4, ff=128, n_layers=2), **kw
    )
