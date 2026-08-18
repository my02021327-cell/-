"""
논문 정합 벤치마크 — Meola & Weinrich (2025) Applied Energy 390:125781 프로토콜의 이식

「Full-scale dynamic anaerobic digestion process simulation with machine and deep
learning algorithms at intra-day resolution」이 전규모 AD 예측에 대해 확립한 세 가지
절차를 영천 자료에 그대로 적용한다.

  1) 주지표를 RMSSE(= 모델 RMSE / naive RMSE)로 둔다.        → src/metrics.py
  2) 모델 라인업을 논문의 권고·기각 목록에 맞춘다.
  3) 검증 오차와 시험 오차의 격차(Fig. 10d)를 모델 선택의 1급 정보로 보고한다.

■ 논문과 다른 점 — 예측 지평을 반드시 분리한다
  논문의 OD(output distance)는 1 timestep ~ 24 h 다. 그 지평에서는 '직전 시점 메탄'이
  실제로 알려져 있고, 논문의 SHAP 결과에서 그것이 1위 피처다. 반면 영천 프로젝트의
  기준선(CV-RMSE 861.5 ㎥/d)은 90일 앞 블록 예측이고, 그 지평에서는 직전 메탄도
  소화조 내부 이화학도 알 수 없다. 같은 자료에서 persistence R² 가 0.877(1일)과
  −0.767(90일 블록)로 갈리는 이유가 이것이다(LESSONS_LEARNED §C2).

  따라서 두 체제를 각각 돌리고 절대 섞지 않는다:

    od1  : 1일 앞. 직전 관측 메탄은 쓰되 내부 이화학은 전일값으로 민다(동시점 금지).
           논문의 OD(=t 까지의 측정으로 t+OD 예측)와 직접 대응하는 체제.
    h90  : 90일 앞 블록. 원점 이후로는 타깃 되먹임과 내부 이화학을 원점값에 동결.
           투입 물량·달력만 미래값을 쓴다(운전원이 투입을 계획한다는 전제).
           PROMPT §8-5 「Forecast 트랙에서 동시점 내부 이화학 사용 금지」를 만족한다.

■ 누출 차단
  v1 파이프라인(src/data.py)은 분할 전에 전체 계열을 `interpolate(limit_direction="both")`
  로 채웠다. bfill 성분이 미래값을 과거로 끌어오므로 학습셋이 시험구간 정보를 본다.
  여기서는 (a) 인과적 ffill 만 쓰고 (b) 남은 결측은 폴드 학습셋 중앙값으로 채운다.

실행 :  python -m src.benchmark            # od1 + h90 둘 다
        python -m src.benchmark --regime od1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import paired_t_test, rmse, score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")

# ── 외생 상수 (PROMPT §2.3 : V 는 추정 대상이 아니라 외부 입력) ────────────────
V_DIGESTER_M3 = 8000.0          # 설계 유효용적. 운전 실적 확보 시 교체
# ── rolling-origin 구성 (PROMPT §8-1) ────────────────────────────────────────
INITIAL_TRAIN_DAYS = 365
HORIZON_DAYS = 90
STEP_DAYS = 90
VAL_DAYS = 90                   # 학습셋 말미 검증블록 (논문 Fig.10d 의 val/test 격차용)
RANDOM_STATE = 42

TARGET = "methane"

# 투입·달력 = 원점 이후에도 알 수 있다고 보는 그룹
FEED_COLS = ["투입량합계", "반입량", "feed_음폐수", "feed_가축분뇨", "feed_음식물", "VS_in"]
# 소화조/유입 내부 이화학 = h90 에서 원점 동결
STATE_COLS = [
    "소화조_TS", "소화조_VS", "소화조_VFA", "소화조_TAlk", "소화조_MAlk",
    "소화조_pH", "소화조_온도", "소화조_CODcr",
    "유입_pH", "유입_TS", "유입_VS", "유입_CODcr",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 피처 구성 (누출 없음 — 인과적 연산만)
# ══════════════════════════════════════════════════════════════════════════════
def build_frame(master_path: str, targets_path: str) -> pd.DataFrame:
    m = pd.read_excel(master_path)
    t = pd.read_excel(targets_path)[["date", TARGET]]
    df = m.merge(t, on="date", how="left").sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # 인과적 ffill : 과거값만 사용하므로 누출 없음
    fill_cols = [c for c in FEED_COLS + STATE_COLS if c in df.columns]
    df[fill_cols] = df[fill_cols].ffill()

    # 파생 — 논문 FI 상위(OLR, HRT, 투입량)와 프로젝트 건강지표
    df["OLR"] = df["VS_in"] / V_DIGESTER_M3
    df["HRT"] = V_DIGESTER_M3 / df["투입량합계"].replace(0, np.nan)
    df["VFA_ALK"] = df["소화조_VFA"] / df["소화조_TAlk"].replace(0, np.nan)
    df["d_feed"] = df["투입량합계"].diff()

    # 체류창 누적 부하 (과거만)
    for v in ["투입량합계", "VS_in"]:
        for w in (5, 10, 30):
            df[f"{v}_ma{w}"] = df[v].rolling(w, min_periods=1).mean()

    # 달력
    df["dow"] = df["date"].dt.dayofweek
    df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)

    # 타깃 되먹임 (직전 '관측' 메탄과 그 간격) — 논문 SHAP 1위 피처
    obs = df[TARGET].notna()
    last_val = df[TARGET].where(obs).ffill().shift(1)
    last_pos = pd.Series(np.where(obs, np.arange(len(df)), np.nan), index=df.index).ffill().shift(1)
    df["methane_prev"] = last_val
    df["gap_days"] = np.arange(len(df)) - last_pos
    df["methane_ma7"] = df[TARGET].where(obs).ffill().shift(1).rolling(7, min_periods=1).mean()
    return df


FEED_FEATURES = FEED_COLS + ["OLR", "HRT", "d_feed"] + [
    f"{v}_ma{w}" for v in ("투입량합계", "VS_in") for w in (5, 10, 30)
]
CAL_FEATURES = ["dow", "doy_sin", "doy_cos"]
STATE_FEATURES = STATE_COLS + ["VFA_ALK"]
TARGET_FEATURES = ["methane_prev", "gap_days", "methane_ma7"]
ALL_FEATURES = FEED_FEATURES + CAL_FEATURES + STATE_FEATURES + TARGET_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# 2. rolling-origin 폴드
# ══════════════════════════════════════════════════════════════════════════════
def rolling_origin_folds(dates: pd.Series, observed: np.ndarray) -> list[dict]:
    """
    확장창 rolling-origin. 학습셋은 항상 검정점 이전 관측만 포함한다(PROMPT §8-1).
    각 폴드는 train / val(학습셋 말미 VAL_DAYS) / test(원점 이후 HORIZON_DAYS).
    """
    d0 = dates.min()
    day = (dates - d0).dt.days.to_numpy()
    folds = []
    origin = INITIAL_TRAIN_DAYS
    while origin + HORIZON_DAYS <= day.max():
        tr = np.where(observed & (day < origin))[0]
        te = np.where(observed & (day >= origin) & (day < origin + HORIZON_DAYS))[0]
        va = tr[day[tr] >= origin - VAL_DAYS]
        fit = tr[day[tr] < origin - VAL_DAYS]
        if len(te) >= 10 and len(fit) >= 100 and len(va) >= 10:
            folds.append({"origin_day": int(origin), "origin": d0 + pd.Timedelta(days=int(origin)),
                          "fit": fit, "val": va, "train": tr, "test": te})
        origin += STEP_DAYS
    return folds


def freeze_state(X: pd.DataFrame, idx: np.ndarray, anchor: int, cols: list[str]) -> pd.DataFrame:
    """h90 : 원점 시점(anchor) 값으로 지정 열을 동결한다."""
    out = X.loc[idx].copy()
    cols = [c for c in cols if c in out.columns]
    out[cols] = X.loc[anchor, cols].to_numpy()
    return out


def lag_state(X: pd.DataFrame, idx: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """
    od1 : 내부 이화학을 전일(t-1) 값으로 밀어 '진짜 1일 앞 예측'으로 만든다.

    동시점 이화학을 그대로 쓰면 그것은 예측이 아니라 nowcast 다(PROMPT 3 T3, 8-5).
    논문의 OD 도 「t 까지의 측정으로 t+OD 를 예측」이므로 이쪽이 논문과 대응한다.
    자료가 2,190일 연속 달력이므로 shift(1) 이 곧 전일이다.
    """
    out = X.loc[idx].copy()
    cols = [c for c in cols if c in out.columns]
    out[cols] = X[cols].shift(1).loc[idx].to_numpy()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모델 라인업 — 논문 §3.6 권고/기각 반영
# ══════════════════════════════════════════════════════════════════════════════
def make_model_grids() -> dict:
    """
    모델별 (이름 → 후보 파이프라인 리스트).

    후보는 **각 폴드의 학습셋 내부**(fit 블록으로 적합 → val 블록으로 평가)에서만 고른다.
    논문은 GA 로 전역 검증셋 하나에서 고르는데, 그 방식은 우리 자료에서 이미 크기가 측정된
    선택편향을 만든다 — 폴드 밖 선택 749.7 vs 폴드 안 선택 777.3, **27 ㎥/d**
    (LESSONS_LEARNED §C3). 그래서 절차만 가져오고 선택 범위는 폴드 안으로 닫는다.

    논문 결론 :
      · 권고    RF, RNN(LSTM/GRU)  — 공정 상태가 불확실하거나 비정상일 때 가장 강건
      · 조건부  LR / EN / BR       — 정상상태·비저해 조건에서 충분. 단 val-test 격차 큼
      · 대안    GBR, ABR, k-NN     — 학습시간이 중요할 때
      · 비권고  PLS, GPR           — 어떤 시나리오에서도 충분한 성능을 내지 못함
      · 제외    MLP, Transformer   — 논문이 애초에 후보에서 뺀 모델(시계열에서 RNN 열위 /
                                     학습시간·하이퍼파라미터 불안정)
    PLS·GPR 은 '비권고 판정이 영천 자료에서도 재현되는가'를 확인하기 위해 포함한다.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def pipe(est, scale=True):
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("sc", StandardScaler()))
        steps.append(("est", est))
        return Pipeline(steps)

    return {
        "LinearRegression": [pipe(LinearRegression())],
        "ElasticNet": [pipe(ElasticNet(alpha=a, l1_ratio=l, max_iter=20000))
                       for a in (0.1, 1.0, 10.0) for l in (0.2, 0.8)],
        "BayesianRidge": [pipe(BayesianRidge(alpha_1=a, lambda_1=a)) for a in (1e-6, 1e-3)],
        "PLSCanonical": [pipe(PLSRegression(n_components=k)) for k in (2, 5, 10)],
        "GaussianProcess": [pipe(
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=ls) + WhiteKernel(1.0),
                normalize_y=True, alpha=1e-6, random_state=RANDOM_STATE))
            for ls in (5.0,)],
        "RandomForest": [pipe(RandomForestRegressor(
            n_estimators=400, min_samples_leaf=m, max_features=f,
            n_jobs=-1, random_state=RANDOM_STATE), scale=False)
            for m in (2, 10) for f in (0.3, 1.0)],
        "GradientBoosting": [pipe(GradientBoostingRegressor(
            n_estimators=n, learning_rate=lr, max_depth=3, random_state=RANDOM_STATE),
            scale=False) for n in (100, 400) for lr in (0.03, 0.1)],
        "AdaBoost": [pipe(AdaBoostRegressor(
            n_estimators=n, learning_rate=lr, random_state=RANDOM_STATE), scale=False)
            for n in (100, 300) for lr in (0.3, 1.0)],
        "kNN": [pipe(KNeighborsRegressor(n_neighbors=k, weights="distance"))
                for k in (5, 10, 25, 50)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 실행
# ══════════════════════════════════════════════════════════════════════════════
# h90 결과를 리드타임으로 층화해 보고한다. 평균은 지평 구조를 숨긴다
# (LESSONS_LEARNED §C4 : 관성 보정 이득이 h=1–3 에서 +46 %, h≥15 에서 0 %).
LEAD_BINS = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 90)]


def _fit_predict(model, Xtr, ytr, Xte):
    from sklearn.base import clone

    mdl = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(Xtr, ytr)
        return mdl, np.asarray(mdl.predict(Xte)).reshape(-1)


def _select_in_fold(candidates, Xfit, yfit, Xval, yval):
    """
    폴드 학습셋 안에서만 후보를 고른다. 반환 (선택된 후보, 검증 RMSE, 선택 인덱스).
    검증 RMSE 는 논문 Fig.10d 의 val 쪽 값이 되고, 시험 RMSE 와의 격차가 곧 선택 낙관편향이다.
    """
    best, best_rmse, best_i = None, np.inf, -1
    for i, cand in enumerate(candidates):
        try:
            _, p = _fit_predict(cand, Xfit, yfit, Xval)
        except Exception:
            continue
        e = rmse(yval, p)
        if e < best_rmse:
            best, best_rmse, best_i = cand, e, i
    return best, best_rmse, best_i


def run_regime(df: pd.DataFrame, regime: str) -> dict:
    """regime ∈ {'od1', 'h90'}"""
    observed = df[TARGET].notna().to_numpy()
    folds = rolling_origin_folds(df["date"], observed)
    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = df[TARGET].to_numpy(dtype=float)
    day = (df["date"] - df["date"].min()).dt.days.to_numpy()

    grids = make_model_grids()
    per_fold = {n: [] for n in grids}
    per_fold["Naive"] = []
    rows, lead_rows = [], []

    for fi, f in enumerate(folds):
        fit_i, val_i, tr_i, te_i = f["fit"], f["val"], f["train"], f["test"]
        anchor = int(tr_i[-1])                       # 원점 직전 마지막 관측일
        val_anchor = int(fit_i[-1])

        if regime == "od1":
            Xfit, Xva = lag_state(X, fit_i, STATE_FEATURES), lag_state(X, val_i, STATE_FEATURES)
            Xtr, Xte = lag_state(X, tr_i, STATE_FEATURES), lag_state(X, te_i, STATE_FEATURES)
            naive_te = df.loc[te_i, "methane_prev"].to_numpy(float)
            naive_va = df.loc[val_i, "methane_prev"].to_numpy(float)
            gap_te = df.loc[te_i, "gap_days"].to_numpy(float)
        else:                                        # h90 : 원점 이후 상태 동결
            frozen = STATE_FEATURES + TARGET_FEATURES
            Xfit, Xtr = X.loc[fit_i], X.loc[tr_i]
            Xva = freeze_state(X, val_i, val_anchor, frozen)
            Xte = freeze_state(X, te_i, anchor, frozen)
            naive_te = np.full(len(te_i), y[anchor])  # 원점값 동결 = 블록 naive
            naive_va = np.full(len(val_i), y[val_anchor])
            gap_te = None

        yfit, yva, ytr, yte = y[fit_i], y[val_i], y[tr_i], y[te_i]
        lead = day[te_i] - f["origin_day"] + 1        # 원점으로부터의 리드타임(일)

        naive_rmse_te = rmse(yte, naive_te)
        naive_rmse_va = rmse(yva, naive_va)
        per_fold["Naive"].append(naive_rmse_te)

        for name, cands in grids.items():
            best, val_rmse, best_i = _select_in_fold(cands, Xfit, yfit, Xva, yva)
            if best is None:
                per_fold[name].append(np.nan)
                continue
            _, p_te = _fit_predict(best, Xtr, ytr, Xte)   # 선택 후 전체 학습셋으로 재적합
            s_te = score(yte, p_te, naive_te, gap_te)
            per_fold[name].append(s_te["RMSE"])
            rows.append({
                "fold": fi, "origin": str(f["origin"].date()), "model": name,
                "selected_cand": best_i, "n_cand": len(cands),
                "test_RMSE": s_te["RMSE"], "test_RMSSE_pct": s_te["RMSSE_pct"],
                "test_R2": s_te["R2"], "test_MAE": s_te["MAE"],
                "val_RMSE": round(val_rmse, 2),
                "val_RMSSE_pct": round(100.0 * val_rmse / naive_rmse_va, 1),
                "naive_RMSE": s_te["naive_RMSE"], "n_test": s_te["n"],
            })
            if regime != "od1":
                for lo, hi in LEAD_BINS:
                    m = (lead >= lo) & (lead <= hi)
                    if m.sum() >= 3:
                        nr = rmse(yte[m], naive_te[m])
                        lead_rows.append({
                            "fold": fi, "model": name, "lead": f"{lo}-{hi}",
                            "RMSE": round(rmse(yte[m], p_te[m]), 2),
                            "naive_RMSE": round(nr, 2), "n": int(m.sum())})

    fold_df = pd.DataFrame(rows)
    naive_mean = float(np.nanmean(per_fold["Naive"]))

    summary = {}
    for name in grids:
        sub = fold_df[fold_df["model"] == name]
        errs = np.asarray(per_fold[name], float)
        summary[name] = {
            "CV_RMSE": round(float(np.nanmean(errs)), 1),
            "RMSSE_pct": round(100.0 * float(np.nanmean(errs)) / naive_mean, 1),
            "val_RMSSE_pct": round(float(sub["val_RMSSE_pct"].mean()), 1),
            # 논문 Fig.10d : 검증-시험 격차 = 폴드 안 선택이 남긴 낙관편향
            "val_test_gap_pp": round(
                float(sub["test_RMSSE_pct"].mean() - sub["val_RMSSE_pct"].mean()), 1),
            "R2_mean": round(float(sub["test_R2"].mean()), 3),
            "beats_naive_folds": int((sub["test_RMSSE_pct"] < 100).sum()),
            "n_distinct_selected": int(sub["selected_cand"].nunique()),
            "vs_naive": paired_t_test(errs, np.asarray(per_fold["Naive"], float)),
        }

    out = {
        "regime": regime,
        "n_folds": len(folds),
        "horizon_days": HORIZON_DAYS if regime == "h90" else 1,
        "naive_CV_RMSE": round(naive_mean, 1),
        "folds": [{"fold": i, "origin": str(f["origin"].date()), "n_test": int(len(f["test"]))}
                  for i, f in enumerate(folds)],
        "models": dict(sorted(summary.items(), key=lambda kv: kv[1]["RMSSE_pct"])),
        "per_fold": fold_df.to_dict(orient="records"),
    }
    if lead_rows:
        ld = pd.DataFrame(lead_rows)
        agg = (ld.groupby(["model", "lead"])[["RMSE", "naive_RMSE"]].mean()
                 .assign(RMSSE_pct=lambda d: (100 * d["RMSE"] / d["naive_RMSE"]).round(1))
                 .round(1).reset_index())
        out["by_lead_time"] = agg.to_dict(orient="records")
    return out


def permutation_importance_report(df: pd.DataFrame, regime: str, model_name: str) -> list[dict]:
    """
    피처 중요도. 논문은 SHAP 을 쓰지만 추가 의존성 없이 같은 순위 질문에 답하기 위해
    마지막 폴드 시험구간의 순열중요도를 사용한다.
    """
    from sklearn.inspection import permutation_importance

    observed = df[TARGET].notna().to_numpy()
    f = rolling_origin_folds(df["date"], observed)[-1]
    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = df[TARGET].to_numpy(float)
    tr_i, te_i, fit_i, val_i = f["train"], f["test"], f["fit"], f["val"]
    anchor = int(tr_i[-1])
    if regime == "od1":
        Xfit, Xva = lag_state(X, fit_i, STATE_FEATURES), lag_state(X, val_i, STATE_FEATURES)
        Xte = lag_state(X, te_i, STATE_FEATURES)
    else:
        frozen = STATE_FEATURES + TARGET_FEATURES
        Xfit = X.loc[fit_i]
        Xva = freeze_state(X, val_i, int(fit_i[-1]), frozen)
        Xte = freeze_state(X, te_i, anchor, frozen)

    best, _, _ = _select_in_fold(make_model_grids()[model_name], Xfit, y[fit_i], Xva, y[val_i])
    Xtr = lag_state(X, tr_i, STATE_FEATURES) if regime == "od1" else X.loc[tr_i]
    mdl, _ = _fit_predict(best, Xtr, y[tr_i], Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(mdl, Xte, y[te_i], n_repeats=20,
                                   random_state=RANDOM_STATE,
                                   scoring="neg_root_mean_squared_error")
    order = np.argsort(-r.importances_mean)
    return [{"feature": Xte.columns[i], "delta_RMSE": round(float(r.importances_mean[i]), 1),
             "sd": round(float(r.importances_std[i]), 1)} for i in order[:15]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["od1", "h90", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_frame(os.path.join(DATA_DIR, "master.xlsx"),
                     os.path.join(DATA_DIR, "targets.xlsx"))
    print(f"자료 {len(df)}일 · 타깃 관측 {int(df[TARGET].notna().sum())}일 "
          f"({df['date'].min().date()} ~ {df['date'].max().date()})", flush=True)

    regimes = ["od1", "h90"] if args.regime == "both" else [args.regime]
    out = {"paper": "Meola & Weinrich (2025) Applied Energy 390:125781",
           "V_digester_m3": V_DIGESTER_M3,
           "protocol": {"initial_train_days": INITIAL_TRAIN_DAYS,
                        "horizon_days": HORIZON_DAYS, "step_days": STEP_DAYS,
                        "val_days": VAL_DAYS,
                        "selection": "in-fold (fit→val), refit on full train"}}

    for reg in regimes:
        res = run_regime(df, reg)
        out[reg] = res
        title = ("OD = 1일 (논문 체제 · 직전 관측 메탄 사용 가능)" if reg == "od1"
                 else "H = 90일 블록 (프로젝트 기준선 체제 · 원점 이후 상태 동결)")
        print(f"\n{'='*78}\n{title}\n{res['n_folds']}폴드 · naive CV-RMSE {res['naive_CV_RMSE']} ㎥/d\n{'='*78}", flush=True)
        print(f"{'Model':<18}{'CV-RMSE':>9}{'RMSSE%':>9}{'val%':>8}{'gap(pp)':>9}{'R²':>8}{'>naive':>9}{'p':>9}")
        for name, sm in res["models"].items():
            p = sm["vs_naive"]["p"]
            print(f"{name:<18}{sm['CV_RMSE']:>9.1f}{sm['RMSSE_pct']:>9.1f}"
                  f"{sm['val_RMSSE_pct']:>8.1f}{sm['val_test_gap_pp']:>+9.1f}"
                  f"{sm['R2_mean']:>8.3f}{sm['beats_naive_folds']:>5d}/{res['n_folds']:<3d}"
                  f"{(f'{p:.4f}' if p is not None else '   —'):>9}", flush=True)
        best = next(iter(res["models"]))
        out[reg]["permutation_importance_best"] = {
            "model": best, "top": permutation_importance_report(df, reg, best)}
        print("\n  [순열중요도 · %s · 마지막 폴드] " % best + ", ".join(
            f"{d['feature']}({d['delta_RMSE']:+.0f})"
            for d in out[reg]["permutation_importance_best"]["top"][:8]), flush=True)

        if "by_lead_time" in res:
            print("\n  [리드타임 층화 · RMSSE %] — 평균은 지평 구조를 숨긴다")
            ld = pd.DataFrame(res["by_lead_time"])
            piv = ld.pivot(index="model", columns="lead", values="RMSSE_pct")
            piv = piv[[c for c in ["1-3", "4-7", "8-14", "15-30", "31-90"] if c in piv.columns]]
            print("  " + piv.sort_values(piv.columns[-1]).to_string().replace("\n", "\n  "), flush=True)

    path = os.path.join(OUT_DIR, "benchmark_meola2025.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n저장 → {path}")
    return out


if __name__ == "__main__":
    main()
