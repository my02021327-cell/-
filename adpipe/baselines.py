"""베이스라인 B0–B4 (PROMPT 2 §2). 이걸 못 이기면 모델은 실패다.

모두 동일 CV 프로토콜(LOGO, site 그룹)로 평가한다. B4(무작위 특징 동수)가 특히 중요:
실제 taxa 가 무작위 특징과 성능차가 없다면 '마이크로바이옴이 예측한다'는 주장은 성립 못 한다.
"""
import warnings
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

from .derived import AlphaDiversity, BetaPCoA, DesignFeatures


def _logo_oof(build_est, X, y, groups, seed=0):
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        est = build_est(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X.iloc[tr], y[tr])
            oof[te] = np.asarray(est.predict(X.iloc[te])).ravel()
    return oof


def _ridge():
    return RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))


def run_baselines(data, seeds=(0, 1, 2, 3, 4)):
    X, y, groups = data["X"], np.asarray(data["y"]), data["groups"].values
    meta = data["meta"]
    out = {}

    def summarize(name, oof_list, note=""):
        r2s = np.array([r2_score(y, o) for o in oof_list])
        out[name] = dict(r2_mean=float(r2s.mean()),
                         r2_sd=float(r2s.std(ddof=1) if len(r2s) > 1 else 0.0),
                         values=r2s.tolist(), note=note, oof=oof_list[0])

    # B0: 학습 fold 타깃 평균
    oof = _logo_oof(lambda s: DummyRegressor(strategy="mean"), X, y, groups)
    summarize("B0_mean", [oof], "학습 fold 평균")

    # B1: 계절만 (season → target)
    def b1(s):
        return Pipeline([("f", FeatureUnion([("design", DesignFeatures(meta=meta))])),
                         ("sc", StandardScaler()), ("m", _ridge())])
    summarize("B1_season", [_logo_oof(b1, X, y, groups)], "season 원-핫+sin/cos")

    # B2: 알파다양성만
    def b2(s):
        return Pipeline([("f", FeatureUnion([("alpha", AlphaDiversity())])),
                         ("sc", StandardScaler()), ("m", _ridge())])
    summarize("B2_alpha", [_logo_oof(b2, X, y, groups)], "알파다양성 5종")

    # B3: PCoA 상위 3축
    def b3(s):
        return Pipeline([("f", FeatureUnion([("beta", BetaPCoA(n_axes=3))])),
                         ("sc", StandardScaler()), ("m", _ridge())])
    summarize("B3_pcoa", [_logo_oof(b3, X, y, groups)], "Aitchison PCoA 3축")

    # B4: 무작위 특징 동수 (동일 p 로 무작위 정규 특징) — 여러 seed
    p = X.shape[1]
    b4_oofs = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        Xr = X.copy()
        Xr.iloc[:, :] = rng.standard_normal((len(X), p))   # 조성과 무관한 잡음 특징
        b4_oofs.append(_logo_oof(lambda ss: Pipeline(
            [("sc", StandardScaler()), ("m", _ridge())]), Xr, y, groups))
    summarize("B4_random", b4_oofs, f"무작위 정규 특징 {p}개(동일 p), {len(seeds)} seed")

    return out
