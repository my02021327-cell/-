"""모델 레지스트리 + 하이퍼파라미터 공간 (PROMPT 2 §3).

소표본(n=40, 유효 n=10, p≫n)에 이론적 근거가 있는 모델만 채택하고 강하게 규제한다.
딥러닝/AutoML/대규모 스태킹은 배제(사유는 리포트에 명시).

각 모델은 (estimator 생성자, 파라미터 후보, 변환 shortlist)를 제공한다. 변환 선택과
하이퍼파라미터는 PROMPT 2 의 내부 CV 안에서 이루어진다(여기서 최종 고정 아님).
"""
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline as SkPipeline


# 변환 shortlist: 모델 특성에 맞춘 후보(내부 CV 가 이 중에서 선택).
LINEAR_TF = ["clr", "alr", "ilr_taxo"]
KERNEL_TF = ["clr", "hellinger", "rank"]
TREE_TF = ["prop", "clr", "rank"]


from sklearn.base import BaseEstimator, RegressorMixin


class SparsePLS(RegressorMixin, BaseEstimator):
    """sPLS 대용: 단변량 상관 상위 k 특징 선택 후 PLS. 해석 목적(성능보다 희소성)."""
    def __init__(self, k=20, n_components=2):
        self.k = k
        self.n_components = n_components

    def _make(self, p):
        k = min(self.k, p)
        return SkPipeline([("sel", SelectKBest(f_regression, k=k)),
                           ("pls", PLSRegression(n_components=min(self.n_components, k)))])

    def fit(self, X, y):
        self._est = self._make(X.shape[1]).fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._est.predict(X).ravel()

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)


def _matern():
    return (ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=1.0))


# 레지스트리: name -> dict(build=callable(**kw)->est, grid=[kw...], transforms=[...], tier)
MODEL_REGISTRY = {
    "elasticnet": dict(
        tier=1, transforms=LINEAR_TF,
        build=lambda **kw: ElasticNet(max_iter=20000, **kw),
        grid=[dict(alpha=a, l1_ratio=r) for a in (0.03, 0.1, 0.3, 1.0)
              for r in (0.2, 0.5, 0.9)],
    ),
    "pls": dict(
        tier=1, transforms=LINEAR_TF,
        build=lambda **kw: PLSRegression(**kw),
        grid=[dict(n_components=c) for c in (1, 2, 3, 4)],
    ),
    "spls": dict(
        tier=1, transforms=LINEAR_TF,
        build=lambda **kw: SparsePLS(**kw),
        grid=[dict(k=k, n_components=c) for k in (10, 20) for c in (1, 2)],
    ),
    "gpr": dict(
        tier=1, transforms=KERNEL_TF,
        build=lambda **kw: GaussianProcessRegressor(
            kernel=_matern(), normalize_y=True, alpha=1e-6, n_restarts_optimizer=1, **kw),
        grid=[dict()],
    ),
    "krr": dict(
        tier=1, transforms=KERNEL_TF,
        build=lambda **kw: KernelRidge(kernel="rbf", **kw),
        grid=[dict(alpha=a, gamma=g) for a in (1.0, 10.0, 50.0)
              for g in (0.001, 0.01)],
    ),
    "svr": dict(
        tier=1, transforms=KERNEL_TF,
        build=lambda **kw: SVR(kernel="rbf", **kw),
        grid=[dict(C=c, gamma=g, epsilon=0.2) for c in (0.5, 2.0)
              for g in (0.001, 0.01)],
    ),
    "rf": dict(
        tier=2, transforms=TREE_TF,
        build=lambda **kw: RandomForestRegressor(
            n_estimators=400, max_features="sqrt", min_samples_leaf=3, n_jobs=1, **kw),
        grid=[dict(max_depth=d) for d in (2, 3, 4)],
    ),
    "hgb": dict(   # XGBoost/LightGBM 대용(미설치) — 강규제 히스토그램 부스팅
        tier=2, transforms=TREE_TF,
        build=lambda **kw: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, early_stopping=False,
            l2_regularization=5.0, **kw),
        grid=[dict(max_depth=d, max_leaf_nodes=n)
              for d in (2, 3) for n in (7, 15)],
    ),
}

TIER1 = [k for k, v in MODEL_REGISTRY.items() if v["tier"] == 1]
TIER2 = [k for k, v in MODEL_REGISTRY.items() if v["tier"] == 2]

EXCLUDED_MODELS = {
    "mlp_deep_learning": "유효 n=10. 파라미터 수가 표본을 압도 → 추정·검증 불가.",
    "automl_large_stacking": "탐색공간이 클수록 승자의 저주 심화(과거 91셀 앙상블→단일 RF 붕괴 사례).",
    "deep_ensemble": "앙상블 이득은 기저 다양성에서 오나 n=10 에서 다양성=잡음.",
}
