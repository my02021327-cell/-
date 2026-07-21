"""STEP 6 — 파생 특징 (PROMPT 1 §STEP 6).

원 taxa 블록과 '별도 블록'으로 산출한다(블록 단위 ablation 을 위해).
  블록 A 알파다양성  : 필터링 이전 RA 에서 계산
  블록 B 베타좌표    : Aitchison(CLR) 기반 PCoA 상위축, 적재는 학습 fold 에서만
  블록 C 도메인 조성비: 혐기성소화 기능 길드 로그비 (taxonomy 필요)
  블록 D 설계 변수   : season 원-핫 + sin/cos (site 는 특징 금지 — CV 그룹키)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from .config import to_proportion
from .zero_replace import MultiplicativeReplacement
from .transforms import CLRTransform


def _df(X):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, float))


# ----------------------------- 블록 A: 알파다양성 ----------------------------
class AlphaDiversity(BaseEstimator, TransformerMixin):
    """Shannon H', Simpson, Pielou 균등도, 검출 taxa 수(임계 0.001), Berger-Parker.

    ※ Chao1/ACE 등 카운트 기반 추정량은 RA 에서 계산 불가 — 산출하지 않는다(리포트에 명시).
    필터링 '이전' 전체 taxa 에서 계산.
    """
    def __init__(self, detection=0.001):
        self.detection = detection

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        P = to_proportion(_df(X)).values
        eps = 1e-12
        Pl = np.clip(P, eps, None)
        shannon = -(P * np.log(Pl)).sum(axis=1)
        simpson = 1.0 - (P ** 2).sum(axis=1)
        richness = (P > self.detection).sum(axis=1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pielou = np.where(richness > 1, shannon / np.log(richness), 0.0)
        berger = P.max(axis=1)
        return np.column_stack([shannon, simpson, pielou, richness, berger])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["alpha_shannon", "alpha_simpson", "alpha_pielou",
                           "alpha_richness", "alpha_berger_parker"], dtype=object)


# ----------------------------- 블록 B: 베타 PCoA -----------------------------
class BetaPCoA(BaseEstimator, TransformerMixin):
    """Aitchison 거리(CLR 후 유클리드) 기반 PCoA 상위 n_axes.

    CLR 평균·PCA 적재는 fit(학습 fold)에서만 학습하고, 테스트는 out-of-sample 투영.
    전역 적재 학습은 명백한 누수.
    """
    def __init__(self, n_axes=3, delta_factor=0.65):
        self.n_axes = n_axes
        self.delta_factor = delta_factor

    def fit(self, X, y=None):
        df = _df(X)
        self._zr = MultiplicativeReplacement(self.delta_factor).fit(df)
        Z = CLRTransform().fit_transform(self._zr.transform(df))
        self._mean = Z.mean(axis=0)
        k = min(self.n_axes, min(Z.shape) - 1) if min(Z.shape) > 1 else 1
        k = max(k, 1)
        self._pca = PCA(n_components=k).fit(Z - self._mean)
        self.n_out_ = k
        return self

    def transform(self, X):
        df = _df(X)
        Z = CLRTransform().fit_transform(self._zr.transform(df))
        return self._pca.transform(Z - self._mean)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"pcoa_ax{i+1}" for i in range(self.n_out_)], dtype=object)


# --------------------------- 블록 C: 도메인 조성비 ---------------------------
class DomainRatios(BaseEstimator, TransformerMixin):
    """혐기성소화 기능 길드 총 RA 및 로그비.

    비율은 반드시 로그비(단순비 금지). 분모 0 은 δ(fit 학습) 적용.
    매칭 실패 taxa 는 fit 시 보고(matched_/unmatched_).
    """
    def __init__(self, taxonomy=None, domain_taxa=None, delta_factor=0.65):
        self.taxonomy = taxonomy
        self.domain_taxa = domain_taxa or {}
        self.delta_factor = delta_factor

    def _guild_members_cols(self, columns):
        """길드 이름 -> 해당 컬럼(taxon) 목록. taxonomy 의 genus 로 매칭."""
        gmap = {}
        matched, unmatched = set(), set()
        genus_of = {}
        for c in columns:
            if self.taxonomy is not None and c in self.taxonomy.index and "genus" in self.taxonomy.columns:
                genus_of[c] = str(self.taxonomy.loc[c, "genus"])
            else:
                genus_of[c] = "unclassified"
        for guild, spec in self.domain_taxa.items():
            if guild == "ratios":
                continue
            members = set(spec.get("members", []))
            cols = [c for c in columns if genus_of[c] in members]
            gmap[guild] = cols
            if cols:
                matched.add(guild)
            else:
                unmatched.add(guild)
        return gmap, matched, unmatched

    def fit(self, X, y=None):
        df = _df(X)
        cols = list(df.columns)
        self.gmap_, self.matched_, self.unmatched_ = self._guild_members_cols(cols)
        self.guild_names_ = [g for g in self.domain_taxa if g != "ratios"]
        self.ratios_ = self.domain_taxa.get("ratios", [])
        P = to_proportion(df).values
        nz = P[P > 0]
        self.delta_ = self.delta_factor * (float(nz.min()) if nz.size else 1e-6)
        return self

    def _guild_sums(self, df):
        P = to_proportion(df)
        sums = {}
        for g in self.guild_names_:
            cols = [c for c in self.gmap_[g] if c in P.columns]
            sums[g] = P[cols].sum(axis=1).values if cols else np.zeros(len(P))
        return sums

    def transform(self, X):
        df = _df(X)
        sums = self._guild_sums(df)
        feats, self._names = [], []
        d = self.delta_
        # (1) 각 길드 log(합+δ)
        for g in self.guild_names_:
            feats.append(np.log(sums[g] + d)); self._names.append(f"domain_log_{g}")
        # (2) 설정된 로그비
        for spec in self.ratios_:
            num = sums.get(spec["numerator"], np.zeros(len(df)))
            den = sums.get(spec["denominator"], np.zeros(len(df)))
            feats.append(np.log((num + d) / (den + d)))
            self._names.append(f"domain_{spec['name']}")
        return np.column_stack(feats) if feats else np.empty((len(df), 0))

    def get_feature_names_out(self, input_features=None):
        names = [f"domain_log_{g}" for g in self.guild_names_]
        names += [f"domain_{s['name']}" for s in self.ratios_]
        return np.asarray(names, dtype=object)


# ----------------------------- 블록 D: 설계 변수 -----------------------------
class DesignFeatures(BaseEstimator, TransformerMixin):
    """season 원-핫 + 순환(sin/cos) 인코딩. site 는 절대 포함하지 않는다(CV 그룹키).

    season 은 예측시점에도 알려진 설계 공변량이므로 고정 카테고리로 인코딩(누수 아님).
    """
    def __init__(self, meta=None, season_order=("spring", "summer", "autumn", "winter")):
        self.meta = meta
        self.season_order = tuple(season_order)

    def fit(self, X, y=None):
        return self

    def _season_for(self, index):
        s = self.meta.loc[list(index), "season"]
        return list(s.values)

    def transform(self, X):
        df = _df(X)
        seasons = self._season_for(df.index)
        order = list(self.season_order)
        onehot = np.zeros((len(df), len(order)))
        ang = np.zeros(len(df));
        sin = np.zeros(len(df)); cos = np.zeros(len(df))
        for i, s in enumerate(seasons):
            j = order.index(s) if s in order else 0
            onehot[i, j] = 1.0
            theta = 2 * np.pi * j / len(order)
            sin[i] = np.sin(theta); cos[i] = np.cos(theta)
        return np.column_stack([onehot, sin, cos])

    def get_feature_names_out(self, input_features=None):
        names = [f"season_{s}" for s in self.season_order] + ["season_sin", "season_cos"]
        return np.asarray(names, dtype=object)
