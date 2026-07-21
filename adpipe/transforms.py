"""STEP 5 — 변환 Transformer 세트 + TRANSFORM_REGISTRY (PROMPT 1 §STEP 5).

8종 변환을 모두 구현하고 이름으로 호출 가능하게 등록한다. 어느 것도 미리 배제하지
않는다(변환 선택은 PROMPT 2 의 내부 CV 가 판정). 로그비 계열(clr/alr/ilr)은 fit 에서
참조/기저를 학습한다 — 전역 선택은 누수다.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import to_proportion

_EPS = 1e-9


def _pos_prop(X, eps):
    """비율화 + 양수 보정(로그 계열 안전용). 이미 ZeroReplace 를 거쳤다면 영향 거의 없음."""
    P = to_proportion(X)
    Pv = np.clip(P.values, eps, None)
    Pv = Pv / Pv.sum(axis=1, keepdims=True)
    return pd.DataFrame(Pv, index=P.index, columns=P.columns)


class _Base(BaseEstimator, TransformerMixin):
    def _df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, float))

    def fit(self, X, y=None):
        self.columns_ = np.asarray(self._df(X).columns)
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{self._prefix}{c}" for c in self.columns_], dtype=object)


class PropTransform(_Base):
    """prop — 재정규화된 상대존재비(항등). 영대체 불필요."""
    _prefix = "prop__"
    def transform(self, X):
        return to_proportion(self._df(X)).values
    def inverse_transform(self, Z):
        return to_proportion(pd.DataFrame(np.asarray(Z, float))).values


class HellingerTransform(_Base):
    """hellinger — √p. 영대체 불필요."""
    _prefix = "hell__"
    def transform(self, X):
        return np.sqrt(to_proportion(self._df(X)).values)
    def inverse_transform(self, Z):
        P = np.asarray(Z, float) ** 2
        return P / P.sum(axis=1, keepdims=True)


class ArcsinTransform(_Base):
    """arcsin — arcsin(√p). 영대체 불필요."""
    _prefix = "asin__"
    def transform(self, X):
        return np.arcsin(np.sqrt(np.clip(to_proportion(self._df(X)).values, 0, 1)))
    def inverse_transform(self, Z):
        P = np.sin(np.asarray(Z, float)) ** 2
        return P / P.sum(axis=1, keepdims=True)


class LogPTransform(_Base):
    """logp — log(p + δ). δ 는 fit 에서 학습(자체 shift → 별도 영대체 불필요)."""
    _prefix = "logp__"
    def __init__(self, delta_factor=0.65):
        self.delta_factor = delta_factor
    def fit(self, X, y=None):
        P = to_proportion(self._df(X)).values
        nz = P[P > 0]
        self.delta_ = self.delta_factor * (float(nz.min()) if nz.size else 1e-6)
        self.columns_ = np.asarray(self._df(X).columns)
        return self
    def transform(self, X):
        P = to_proportion(self._df(X)).values
        return np.log(P + self.delta_)
    def inverse_transform(self, Z):
        P = np.exp(np.asarray(Z, float)) - self.delta_
        P = np.clip(P, 0, None)
        return P / P.sum(axis=1, keepdims=True)


class CLRTransform(_Base):
    """clr — log(p) − mean(log(p)) 행별. 로그비 → 영대체 필요(선행 단계에서 처리)."""
    _prefix = "clr__"
    def __init__(self, eps=_EPS):
        self.eps = eps
    def transform(self, X):
        P = _pos_prop(self._df(X), self.eps).values
        L = np.log(P)
        return L - L.mean(axis=1, keepdims=True)
    def inverse_transform(self, Z):
        E = np.exp(np.asarray(Z, float))
        return E / E.sum(axis=1, keepdims=True)


class ALRTransform(_Base):
    """alr — log(p_i / p_ref). ref = 학습 fold 에서 로그분산 최소 taxon (fit 내부 선택)."""
    _prefix = "alr__"
    def __init__(self, eps=_EPS):
        self.eps = eps
    def fit(self, X, y=None):
        df = self._df(X)
        P = _pos_prop(df, self.eps).values
        L = np.log(P)
        logvar = L.var(axis=0)
        self.ref_idx_ = int(np.argmin(logvar))
        self.columns_ = np.asarray(df.columns)
        self.ref_name_ = str(self.columns_[self.ref_idx_])
        return self
    def transform(self, X):
        P = _pos_prop(self._df(X), self.eps).values
        L = np.log(P)
        return L - L[:, [self.ref_idx_]]
    def get_feature_names_out(self, input_features=None):
        names = [f"alr__{c}_vs_{self.ref_name_}"
                 for j, c in enumerate(self.columns_) if j != self.ref_idx_]
        return np.asarray(names, dtype=object)
    def transform_drop_ref(self, X):
        Z = self.transform(X)
        return np.delete(Z, self.ref_idx_, axis=1)


class ILRTaxoTransform(_Base):
    """ilr_taxo — 분류체계 기반 순차이분할(SBP) 밸런스.

    계통수가 없으므로 taxonomy(phylum→family→genus→taxon)로 이진분할 트리를 만들고
    Egozcue 밸런스 기저 Θ 를 구성한다. 구성 실패 시 skip 로그를 남기고 0-특징 반환.
    """
    _prefix = "ilr__"
    def __init__(self, taxonomy=None, levels=("phylum", "family", "genus"), eps=_EPS):
        self.taxonomy = taxonomy
        self.levels = levels
        self.eps = eps

    def fit(self, X, y=None):
        df = self._df(X)
        cols = list(df.columns)
        self.columns_ = np.asarray(cols)
        self.skipped_ = False
        try:
            parts = _sbp_partitions(cols, self.taxonomy, self.levels)
            if not parts or len(cols) < 2:
                raise ValueError("SBP 분할 생성 실패 또는 taxa<2")
            self.theta_ = _balance_basis(parts, len(cols))   # (m, D)
            self.n_balances_ = self.theta_.shape[0]
        except Exception as e:
            self.skipped_ = True
            self.theta_ = None
            self.n_balances_ = 0
            warnings.warn(f"[ILRTaxoTransform] SKIP — SBP 구성 실패: {e}")
        return self

    def transform(self, X):
        df = self._df(X)
        if self.skipped_ or self.theta_ is None:
            return np.empty((len(df), 0))
        P = _pos_prop(df, self.eps).values
        return np.log(P) @ self.theta_.T

    def get_feature_names_out(self, input_features=None):
        if getattr(self, "skipped_", True):
            return np.asarray([], dtype=object)
        return np.asarray([f"ilr__bal{k+1}" for k in range(self.n_balances_)], dtype=object)

    def inverse_transform(self, Z):
        if self.theta_ is None:
            raise NotImplementedError
        pinv = np.linalg.pinv(self.theta_.T)
        L = np.asarray(Z, float) @ pinv.T
        E = np.exp(L)
        return E / E.sum(axis=1, keepdims=True)


class RankNormTransform(_Base):
    """rank — 시료 내 순위 정규화(분포 무관 강건 옵션). 영대체 불필요."""
    _prefix = "rank__"
    def transform(self, X):
        P = to_proportion(self._df(X)).values
        # 행별 평균 순위(0~1). 동점은 평균 순위.
        order = P.argsort(axis=1).argsort(axis=1).astype(float)
        # 동점 처리(간이): argsort 기반이라 근사. p 차원 정규화.
        return order / (P.shape[1] - 1 if P.shape[1] > 1 else 1)


# ------------------------- SBP / balance basis helpers -----------------------
def _tax_lookup(taxonomy, col, level):
    if taxonomy is None or col not in taxonomy.index or level not in taxonomy.columns:
        return "unclassified"
    v = taxonomy.loc[col, level]
    return str(v) if pd.notna(v) else "unclassified"


def _sbp_partitions(cols, taxonomy, levels):
    """taxonomy 로부터 순차 이분할 목록 생성. 각 원소=(plus[list idx], minus[list idx])."""
    idx_of = {c: i for i, c in enumerate(cols)}

    def split(group, level_i):
        # group: 열 이름 list. 재귀적으로 이분할.
        if len(group) <= 1:
            return []
        # 현재 레벨 라벨로 자식 그룹 구성
        if level_i < len(levels):
            level = levels[level_i]
            labels = {}
            for c in group:
                labels.setdefault(_tax_lookup(taxonomy, c, level), []).append(c)
            child_groups = list(labels.values())
        else:
            child_groups = [[c] for c in group]  # 잎까지 왔으면 개별 분해
        parts = []
        if len(child_groups) == 1:
            # 이 레벨에서 안 갈라짐 → 다음 레벨로
            return split(group, level_i + 1)
        # child_groups 를 캐터필러로 이분: 첫 자식 vs 나머지
        # 크기 균형을 위해 큰 자식부터 정렬
        child_groups.sort(key=len, reverse=True)
        left = child_groups[0]
        right = [c for g in child_groups[1:] for c in g]
        parts.append(([idx_of[c] for c in left], [idx_of[c] for c in right]))
        # 각 하위 그룹 재귀 (같은 레벨에서 계속 세분)
        parts += split(left, level_i + 1)
        parts += split(right, level_i)
        return parts

    return split(list(cols), 0)


def _balance_basis(parts, D):
    """이분할 목록 → Egozcue 밸런스 기저 Θ (m×D). m 은 partition 수(=D-1 이상적)."""
    rows = []
    for plus, minus in parts:
        r, s = len(plus), len(minus)
        if r == 0 or s == 0:
            continue
        coef = np.sqrt(r * s / (r + s))
        row = np.zeros(D)
        row[plus] = coef / r
        row[minus] = -coef / s
        rows.append(row)
    if not rows:
        raise ValueError("유효 밸런스 0개")
    return np.vstack(rows)


# ------------------------------- REGISTRY ------------------------------------
# key -> dict(factory, needs_zero_replace, scaler)
TRANSFORM_REGISTRY = {
    "prop":      dict(factory=lambda **k: PropTransform(),          needs_zero_replace=False, scaler="none"),
    "hellinger": dict(factory=lambda **k: HellingerTransform(),     needs_zero_replace=False, scaler="none"),
    "arcsin":    dict(factory=lambda **k: ArcsinTransform(),        needs_zero_replace=False, scaler="standard"),
    "logp":      dict(factory=lambda **k: LogPTransform(),          needs_zero_replace=False, scaler="standard"),
    "clr":       dict(factory=lambda **k: CLRTransform(),           needs_zero_replace=True,  scaler="standard"),
    "alr":       dict(factory=lambda **k: ALRTransform(),           needs_zero_replace=True,  scaler="standard"),
    "ilr_taxo":  dict(factory=lambda taxonomy=None, **k: ILRTaxoTransform(taxonomy=taxonomy),
                      needs_zero_replace=True, scaler="standard"),
    "rank":      dict(factory=lambda **k: RankNormTransform(),      needs_zero_replace=False, scaler="standard"),
}


def make_transform(key, taxonomy=None):
    spec = TRANSFORM_REGISTRY[key]
    return spec["factory"](taxonomy=taxonomy)
