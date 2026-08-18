"""
결측 보간 — Meola & Weinrich (2025) Applied Energy 390:125781 의 소프트센서 이식

논문 §2.3 Algorithm 1 : 결측값은 「같은 시점의 다른 관측으로 학습한 모델」로 대치하며,
개별 알고리즘이 유효하지 않은 출력을 내면 다음 모델로 넘어간다. 논문의 후보는
BayesianRidge → AdaBoost → GradientBoosting 이고, 여기서도 같은 순서를 쓴다.

■ 왜 이 자료에 특히 잘 맞나
  타깃 결측 39.4 % 는 전부 CH₄ 농도 미측정일이고, 같은 날의 바이오가스 유량·투입량·
  소화조 상태는 관측돼 있다. 즉 「같은 시점의 다른 관측」이 실제로 풍부하다.
  게다가 농도는 매우 매끄럽다 — ACF(1)=0.968, 일간 변화 SD 0.86 %p.

■ 두 가지 원칙 (이걸 어기면 성능이 가짜가 된다)
  1. **폴드 내부 적합.** 소프트센서는 각 폴드의 학습 구간으로만 적합한다. 전 구간으로
     한 번 적합해 두면 시험구간 정보가 학습으로 새어 들어온다.
  2. **보간값은 학습 라벨로만, 평가 라벨로는 절대 쓰지 않는다.** 보간이 만든 라벨로
     평가하면 「보간 모델을 얼마나 잘 흉내 내는가」를 재게 된다.
     `TargetReconstructor` 는 두 종류를 구분해 표시한다(`label_is_observed`).

■ 보간 비용은 가정하지 않고 가림 실험으로 잰다
  `masking_validation()` 이 관측값을 일부러 가리고 복원해 간격별 오차를 낸다.
  "보간 금지"를 교조로 두는 대신 **비용을 알고 결정**하기 위한 절차다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.bgp import config as C


def _candidate_models():
    """논문 Algorithm 1 의 후보 순서. 앞에서부터 시도하고 무효 출력이면 다음으로 넘어간다."""
    return [
        ("BayesianRidge", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                    ("est", BayesianRidge())])),
        ("AdaBoost", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("est", AdaBoostRegressor(n_estimators=200,
                                                         random_state=C.RANDOM_STATE))])),
        ("GradientBoosting", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("est", GradientBoostingRegressor(
                                           random_state=C.RANDOM_STATE))])),
    ]


class SoftSensor:
    """
    한 열의 결측을 같은 시점의 다른 열로 대치하는 소프트센서.

    fit(df_train, target_col, predictor_cols)
      target_col 이 관측된 학습 행만 써서 후보 모델을 순서대로 적합하고,
      **학습셋 안에서** 유효성(물리 범위·비퇴화)을 통과한 첫 모델을 채택한다.
    transform(df) → 결측 위치만 예측값으로 채운 Series
    """

    def __init__(self, valid_range: tuple[float, float] | None = None):
        self.valid_range = valid_range
        self.model_name_: str | None = None
        self.model_ = None
        self.cols_: list[str] = []
        self.fallback_: float = np.nan

    def fit(self, df_train: pd.DataFrame, target_col: str, predictor_cols: list[str]):
        self.cols_ = [c for c in predictor_cols if c in df_train.columns and c != target_col]
        obs = df_train[target_col].notna()
        y = df_train.loc[obs, target_col].to_numpy(float)
        X = df_train.loc[obs, self.cols_]
        self.fallback_ = float(np.nanmedian(y)) if len(y) else np.nan
        if len(y) < 30:
            return self

        lo, hi = self.valid_range or (-np.inf, np.inf)
        for name, mdl in _candidate_models():
            try:
                mdl.fit(X, y)
                p = np.asarray(mdl.predict(X), float)
            except Exception:
                continue
            # 논문의 "invalid results" 판정 : 비유한값 / 물리범위 이탈 / 상수 출력
            if (not np.all(np.isfinite(p))) or p.min() < lo or p.max() > hi or p.std() < 1e-9:
                continue
            self.model_name_, self.model_ = name, mdl
            break
        return self

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        out = df[target_col].copy()
        miss = out.isna()
        if not miss.any():
            return out
        if self.model_ is None:
            out.loc[miss] = self.fallback_
            return out
        p = np.asarray(self.model_.predict(df.loc[miss, self.cols_]), float)
        if self.valid_range:
            p = np.clip(p, *self.valid_range)
        p = np.where(np.isfinite(p), p, self.fallback_)
        out.loc[miss] = p
        return out


def flag_measurement_errors(df: pd.DataFrame, cols: list[str],
                            contamination: float = 0.02) -> np.ndarray:
    """
    논문 파이프라인의 isolation forest 계측오류 탐지.
    반환 True = 이상치로 지목된 행. 삭제하지 않고 **플래그로 남긴다** — 판단은 사람이 한다.
    """
    use = [c for c in cols if c in df.columns]
    X = SimpleImputer(strategy="median").fit_transform(df[use])
    iso = IsolationForest(contamination=contamination, random_state=C.RANDOM_STATE, n_jobs=-1)
    return iso.fit_predict(X) == -1


class TargetReconstructor:
    """
    타깃 CH₄ 재구성 — 유량은 완전 관측이므로 **농도만** 보간한다.

        CH4_m3d = biogas_AB_m3d × CH4_pct / 100

    결측 39.4 % 를 통째로 회귀로 만들어내는 대신, 결측의 실제 원인(농도 미측정)만
    소프트센서로 메운다. 물리식이 나머지를 결정하므로 자유도가 훨씬 작다.
    """

    def __init__(self):
        self.sensor = SoftSensor(valid_range=C.DQ["ranges"][C.CONC])

    def fit(self, df_train: pd.DataFrame, predictor_cols: list[str]):
        self.sensor.fit(df_train, C.CONC, predictor_cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label_is_observed"] = out[C.TARGET].notna().astype(int)
        conc = self.sensor.transform(out, C.CONC)
        out["CH4_pct_filled"] = conc
        recon = out[C.FLOW] * conc / 100.0
        out["CH4_m3d_filled"] = out[C.TARGET].where(out[C.TARGET].notna(), recon)
        out["softsensor_model"] = self.sensor.model_name_ or "median"
        return out


def masking_validation(df: pd.DataFrame, predictor_cols: list[str],
                       gaps=(1, 2, 3, 5, 7), n_blocks: int = 60,
                       seed: int = C.RANDOM_STATE) -> dict:
    """
    가림 실험 — 관측된 CH₄ 농도를 일부러 가리고 복원해 간격별 오차를 잰다.

    "보간이 위험하다"를 가정으로 두지 않고 **비용을 수치로** 만든다.
    간격 g 마다 n_blocks 개의 연속 블록을 한꺼번에 가리고 소프트센서를 **한 번** 적합해
    전부 복원한다(블록마다 재적합하면 같은 값을 얻는 데 수백 배 시간이 든다).

    반환 : 간격별 농도 MAE(%p)와 그것을 메탄량으로 환산한 오차(㎥/d).
    """
    rng = np.random.default_rng(seed)
    conc = df[C.CONC].to_numpy(float)
    obs_idx = np.where(np.isfinite(conc))[0]
    flow_med = float(df[C.FLOW].median())
    res = {}

    for g in gaps:
        starts = rng.choice(obs_idx, size=min(n_blocks, len(obs_idx)), replace=False)
        mask = np.zeros(len(df), bool)
        for i in starts:
            blk = np.arange(i, min(i + g, len(df)))
            mask[blk[np.isfinite(conc[blk])]] = True
        if mask.sum() < 10:
            continue

        work = df.copy()
        work.loc[work.index[mask], C.CONC] = np.nan
        train = work.loc[~mask]
        s = SoftSensor(valid_range=C.DQ["ranges"][C.CONC]).fit(train, C.CONC, predictor_cols)
        pred = s.transform(work, C.CONC).to_numpy(float)[mask]
        truth = conc[mask]
        mae = float(np.mean(np.abs(pred - truth)))
        res[f"gap{g}"] = {
            "conc_MAE_pp": round(mae, 3),
            "conc_RMSE_pp": round(float(np.sqrt(np.mean((pred - truth) ** 2))), 3),
            "ch4_equiv_m3d": round(mae / 100.0 * flow_med, 1),
            "n_masked": int(mask.sum()),
            "model": s.model_name_ or "median",
        }
    return res


def naive_fill_baselines(df: pd.DataFrame, gaps=(1, 2, 3, 5, 7),
                         n_blocks: int = 60, seed: int = C.RANDOM_STATE) -> dict:
    """
    소프트센서와 비교할 단순 대안 — 전방보간(ffill)과 선형보간.
    소프트센서를 쓰는 것이 실제로 이득인지 확인하지 않고 도입하면 복잡도만 늘어난다.
    """
    rng = np.random.default_rng(seed)
    conc = df[C.CONC].to_numpy(float)
    obs_idx = np.where(np.isfinite(conc))[0]
    out = {}
    for g in gaps:
        starts = rng.choice(obs_idx, size=min(n_blocks, len(obs_idx)), replace=False)
        mask = np.zeros(len(df), bool)
        for i in starts:
            blk = np.arange(i, min(i + g, len(df)))
            mask[blk[np.isfinite(conc[blk])]] = True
        if mask.sum() < 10:
            continue
        ser = pd.Series(conc).mask(mask)
        out[f"gap{g}"] = {
            "ffill_MAE_pp": round(float(np.mean(np.abs(
                ser.ffill().to_numpy()[mask] - conc[mask]))), 3),
            "linear_MAE_pp": round(float(np.mean(np.abs(
                ser.interpolate("linear", limit_direction="both").to_numpy()[mask]
                - conc[mask]))), 3),
            "n_masked": int(mask.sum()),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 결측 처리 방식 자체를 '고르는' 층 — 논문 파이프라인의 NaN-handling 파라미터
# ══════════════════════════════════════════════════════════════════════════════
# 논문은 결측 처리 방식을 고정하지 않고 **최적화 대상 데이터준비 파라미터**로 둔다
# (Sobol 분석에서 세 데이터셋 모두 NaN handling 이 상위 기여 인자였다).
# 그래서 여기서도 소프트센서를 무조건 쓰지 않고, 후보들을 가림 실험으로 재서 고른다.
#
# 실제로 재보면 CH₄ 농도에서는 소프트센서 단독이 선형보간보다 **나쁘다**
# (MAE 1.50 %p vs 0.37 %p, 간격 1일). 농도가 시간적으로 매우 매끄러운 반면
# (ACF(1)=0.968) 동시점 설명변수는 약하기 때문이다. 이 사실을 무시하고 논문 방식을
# 그대로 쓰면 보간 오차를 4배로 키운다.

def _neighbour_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    직전/직후 관측값과 그 거리 — 시간 이웃 정보를 소프트센서에 넣기 위한 열.

    **반드시 전체 프레임에서 한 번 계산한 뒤 행을 잘라 써야 한다.** 학습용 부분집합에서
    따로 계산하면 실제로는 떨어져 있는 행이 '이웃'으로 잡혀 학습과 예측의 피처 정의가
    달라진다.
    """
    v = df[col]
    idx = pd.Series(np.arange(len(df)), index=df.index)
    prev_v, next_v = v.ffill(), v.bfill()
    prev_i = idx.where(v.notna()).ffill()
    next_i = idx.where(v.notna()).bfill()
    return pd.DataFrame({
        f"{col}_prev_obs": prev_v,
        f"{col}_next_obs": next_v,
        f"{col}_gap_prev": idx - prev_i,
        f"{col}_gap_next": next_i - idx,
        f"{col}_neigh_mean": (prev_v + next_v) / 2.0,
    }, index=df.index)


class HybridImputer:
    """
    결측 처리 방식을 **가림 실험으로 골라** 적용한다.

    후보
      linear      : 시간 선형보간 (매끄러운 계열에 강함)
      ffill       : 전방보간 (인과적. 실시간 운전에서 항상 가능)
      softsensor  : 논문 Algorithm 1 (동시점 다른 관측 → 회귀)
      soft_neigh  : 소프트센서 + 시간 이웃 피처 (동시점 정보와 시간 구조를 함께)

    `causal_only=True` 면 미래 관측을 쓰는 후보(linear, soft_neigh 의 next 계열)를
    제외한다 — 실시간 서빙 경로에서 쓰는 모드다.
    """

    CANDIDATES = ("linear", "ffill", "softsensor", "soft_neigh")

    def __init__(self, col: str = C.CONC, valid_range=None, causal_only: bool = False):
        self.col = col
        self.valid_range = valid_range or C.DQ["ranges"].get(col)
        self.causal_only = causal_only
        self.method_: str | None = None
        self.scores_: dict = {}
        self._sensor: SoftSensor | None = None
        self._pred_cols: list[str] = []

    # ── 후보별 복원 ──────────────────────────────────────────────────────────
    def _apply(self, df: pd.DataFrame, method: str, fit_df: pd.DataFrame) -> np.ndarray:
        col = self.col
        if method == "linear":
            return df[col].interpolate("linear", limit_direction="both").to_numpy(float)
        if method == "ffill":
            return df[col].ffill().bfill().to_numpy(float)

        work = df
        pcols = list(self._pred_cols)
        fit_src = fit_df
        if method == "soft_neigh":
            nb = _neighbour_features(df, col)          # 전체 프레임에서 한 번
            work = pd.concat([df, nb], axis=1)
            fit_src = work.loc[fit_df.index]           # 행만 잘라 쓴다
            pcols = pcols + list(nb.columns)
            if self.causal_only:
                # 미래 관측을 참조하는 이웃 피처는 실시간 경로에서 존재하지 않는다
                future = (f"{col}_next_obs", f"{col}_gap_next", f"{col}_neigh_mean")
                pcols = [c for c in pcols if c not in future]
        s = SoftSensor(valid_range=self.valid_range).fit(fit_src, col, pcols)
        return s.transform(work, col).to_numpy(float)

    # ── 가림 실험으로 후보 선택 ─────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame, predictor_cols: list[str],
            gaps=(1, 3, 7), n_blocks: int = 60, seed: int = C.RANDOM_STATE):
        self._pred_cols = [c for c in predictor_cols
                           if c in df_train.columns and c != self.col]
        rng = np.random.default_rng(seed)
        v = df_train[self.col].to_numpy(float)
        obs = np.where(np.isfinite(v))[0]

        cands = [c for c in self.CANDIDATES
                 if not (self.causal_only and c == "linear")]
        totals = {c: [] for c in cands}
        for g in gaps:
            starts = rng.choice(obs, size=min(n_blocks, len(obs)), replace=False)
            mask = np.zeros(len(df_train), bool)
            for i in starts:
                blk = np.arange(i, min(i + g, len(df_train)))
                mask[blk[np.isfinite(v[blk])]] = True
            if mask.sum() < 10:
                continue
            work = df_train.copy()
            work.loc[work.index[mask], self.col] = np.nan
            for c in cands:
                try:
                    pred = self._apply(work, c, work.loc[~mask])
                    totals[c].append(float(np.mean(np.abs(pred[mask] - v[mask]))))
                except Exception:
                    totals[c].append(np.inf)

        self.scores_ = {c: round(float(np.mean(e)), 3) for c, e in totals.items() if e}
        self.method_ = min(self.scores_, key=self.scores_.get)
        return self

    def transform(self, df: pd.DataFrame, fit_df: pd.DataFrame | None = None) -> pd.Series:
        assert self.method_, "fit() 먼저"
        vals = self._apply(df, self.method_, fit_df if fit_df is not None else df)
        if self.valid_range:
            vals = np.clip(vals, *self.valid_range)
        return pd.Series(vals, index=df.index, name=f"{self.col}_filled")
