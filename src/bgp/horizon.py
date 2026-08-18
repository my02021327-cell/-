"""
지평 구간별 학습·검증 — 확장창 rolling-origin

구간(요구사항): 1–3 / 3–5 / 5–7 / 7–14 / 14–30 / 30–60 / 60–90일.
구간마다 **독립적으로** 모델을 고르고 평가한다. 어떤 모델이 최적인지는 지평이 정하며,
"이 시설에는 X 모델" 식의 단일 처방은 성립하지 않는다.

■ 폴드 규약 (위반하면 결과 전체가 무효)
  1. 원점 O 의 학습 표본은 **t+h < O 인 쌍만** 포함한다. 라벨이 O 이후를 보면 안 된다.
  2. 시험 표본은 t ∈ [O, O+STEP) 이며 라벨은 **실측된 것만** 쓴다(재구성 라벨 평가 금지).
  3. 하이퍼파라미터·피처 선택·스태킹 가중치는 전부 **폴드 학습셋 안에서** 정한다.
     폴드 밖에서 고르면 선택편향이 성능으로 둔갑한다.
  4. 무작위 k-fold 금지 — 일별 자기상관으로 인접일이 새어 들어온다.

■ 보고 지표
  · pooled OOF R²  : 전 폴드의 폴드밖 예측을 모아 계산. 사용자가 요구한 0.85 기준선.
  · RMSSE          : naive(원점값 유지) 대비 정규화 오차. 100 % 미만이어야 채택 가능.
  · 폴드별 대응 t검정 : 평균만 비교하고 "개선됐다"고 쓰지 않기 위한 절차.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.bgp import config as C
from src.bgp.features import horizon_pairs
from src.bgp.models import (
    GRUForecaster,
    SARIMAXBand,
    PersistenceBand,
    SeasonalNaive,
    model_zoo,
    nnls_stack,
)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def select_features(Xfit: pd.DataFrame, yfit: np.ndarray, k: int = 120) -> list[str]:
    """
    폴드 내부 피처 선택 — 논문의 mRMR 자리에 해당하는 단계.

    상관 기반 필터 후 **중복 제거**(선택된 피처와 |r|>0.98 인 후보는 버림)를 건다.
    mRMR 의 relevance-redundancy 취지를 유지하면서 계산 비용은 훨씬 낮다.
    """
    num = Xfit.select_dtypes(include=[np.number])
    sd = num.std(numeric_only=True)
    num = num.loc[:, sd.to_numpy() > 1e-12]
    if num.shape[1] == 0:
        return []
    A = num.to_numpy(float)
    A = np.where(np.isfinite(A), A, np.nan)
    col_mean = np.nanmean(A, axis=0)
    A = np.where(np.isfinite(A), A, col_mean)
    y = np.asarray(yfit, float)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    rel = np.abs(Az.T @ yz) / len(y)

    order = np.argsort(-rel)
    chosen: list[int] = []
    for j in order:
        if len(chosen) >= k:
            break
        if chosen:
            r = np.abs(Az[:, chosen].T @ Az[:, j]) / len(y)
            if r.max() > 0.98:
                continue
        chosen.append(int(j))
    return [num.columns[j] for j in chosen]


def _fit_predict(est, Xtr, ytr, Xte):
    m = clone(est)
    m.fit(Xtr, ytr)
    return m, np.asarray(m.predict(Xte), float).ravel()


def run_band(df: pd.DataFrame, X: pd.DataFrame, y_train_label: pd.Series,
             y_true: pd.Series, band: tuple[str, int, int],
             fast: bool = True, with_gru: bool = False,
             n_features: int = 120, max_train: int = 8000,
             verbose: bool = True, with_flowconc: bool = True,
             with_sarimax: bool = False) -> dict:
    """
    한 지평 구간의 rolling-origin 전 과정.

    `with_flowconc` 는 CH₄ = 유량 × 농도 분해 모델을 라인업에 넣는다. 유량은 결측 0 % 라
    라벨이 2,086일 전부 있고(타깃은 1,265일), 농도는 ACF(1)=0.968 로 매우 매끄럽다.
    두 성분을 따로 맞혀 곱하면 합성 타깃을 직접 맞히는 것과 **다른 귀납 편향**이 되고,
    유량 쪽은 재구성 잡음이 섞이지 않은 라벨로 학습된다.
    """
    name, lo, hi = band
    n = len(df)
    observed = y_true.notna()
    min_origin = 60                                   # 이동통계 워밍업

    o_all, h_all, tgt_all, obs_all = horizon_pairs(
        X, y_train_label, observed, lo, hi, min_origin=min_origin)
    tstar = o_all + h_all                              # 라벨 시점

    Xn = X.to_numpy(np.float32)
    cols = list(X.columns)

    zoo = model_zoo(fast=fast)
    if with_gru:
        zoo["GRU"] = ["__gru__"]
    if with_flowconc:
        zoo["FlowConc"] = ["__flowconc__"]
    if with_sarimax:
        zoo["SARIMAX"] = ["__sarimax__"]

    # ── 잔차 표적(delta) 변형 ────────────────────────────────────────────────
    # y(t+h) 를 직접 맞히는 대신 「원점 실측 대비 변화량」 y(t+h) − anchor(t) 를 맞히고
    # 예측 시 anchor 를 되더한다. 같은 정보인데 손실이 변화량에 집중되고, 트리 계열은
    # 수준을 외삽하지 못하므로 이 형태에서 크게 유리하다.
    #   · delta ≡ 0 이면 정확히 naive 가 되므로 하한이 보장된다.
    #   · 선형모델은 anchor 가 이미 피처(tmp__y_last)라 두 형태가 거의 같아 제외한다.
    DELTA_FAMILIES = {"HistGBM", "RandomForest", "ExtraTrees", "kNN", "GradientBoosting"}

    origins = list(range(C.CV_INITIAL_TRAIN_DAYS, n - hi, C.CV_STEP_DAYS))
    per_fold: dict[str, list[float]] = {k: [] for k in list(zoo) + ["Naive", "SeasonalNaive", "Stack"]}
    oof: dict[str, list[np.ndarray]] = {k: [] for k in per_fold}
    oof_y, oof_t, oof_h, oof_fold = [], [], [], []
    sel_hist: dict[str, int] = {}

    anchor_all = y_true.ffill().to_numpy(float)
    flow_all = df[C.FLOW].ffill().to_numpy(float)
    conc_all = df.get("CH4_pct_filled", df[C.CONC]).ffill().to_numpy(float)

    gru_panel = None
    if with_gru:
        gcols = [c for c in cols if c.startswith("tmp__") or "__flow" in c]
        gp = X[gcols].to_numpy(np.float32)
        gru_panel = np.nan_to_num(gp, nan=np.float32(0.0))

    for fi, O in enumerate(origins):
        tr = np.where(tstar < O)[0]
        te = np.where((o_all >= O) & (o_all < O + C.CV_STEP_DAYS) & obs_all)[0]
        if len(tr) < 500 or len(te) < 10:
            continue
        va = tr[tstar[tr] >= O - C.CV_VAL_DAYS]
        fit = tr[tstar[tr] < O - C.CV_VAL_DAYS]
        if len(va) < 30 or len(fit) < 300:
            continue

        # 학습 표본 상한 — 최근 구간 우선(오래된 국면의 비중을 낮춘다)
        if len(fit) > max_train:
            fit = fit[-max_train:]
        if len(tr) > max_train:
            tr = tr[-max_train:]

        feat = select_features(pd.DataFrame(Xn[o_all[fit]], columns=cols),
                               tgt_all[fit], k=n_features)
        fidx = [cols.index(c) for c in feat]

        def mk(idx, with_meta=False):
            d = pd.DataFrame(Xn[o_all[idx]][:, fidx], columns=feat)
            d["__h__"] = h_all[idx].astype(np.float32)
            if with_meta:
                d["__origin__"] = o_all[idx]
            return d

        Xfit, Xva, Xtr, Xte = mk(fit), mk(va), mk(tr), mk(te)
        yfit, yva, ytr, yte = tgt_all[fit], tgt_all[va], tgt_all[tr], tgt_all[te]

        # naive : 원점의 마지막 관측 메탄을 그대로 민다
        anchor = anchor_all
        naive_te = anchor[o_all[te]]
        naive_va = anchor[o_all[va]]
        per_fold["Naive"].append(_rmse(yte, naive_te))
        oof["Naive"].append(naive_te)

        sn = SeasonalNaive().fit(pd.DataFrame({"tmp__y_ma30": X["tmp__y_ma30"].to_numpy()[o_all[tr]]}), ytr)
        p_sn = sn.predict(pd.DataFrame({"tmp__y_ma30": X["tmp__y_ma30"].to_numpy()[o_all[te]]}))
        per_fold["SeasonalNaive"].append(_rmse(yte, p_sn))
        oof["SeasonalNaive"].append(p_sn)

        # 잔차 표적용 앵커 : naive 와 같은 값을 써야 delta=0 이 naive 로 수렴한다
        a_fit, a_va = anchor[o_all[fit]], anchor[o_all[va]]
        a_tr, a_te = anchor[o_all[tr]], anchor[o_all[te]]

        val_preds, test_preds = {}, {}
        for mname, cands in zoo.items():
            if mname == "SARIMAX":
                # 유량 계열을 상태공간 모형으로 앞으로 굴리고 농도를 곱해 메탄으로 되돌린다.
                # 농도는 원점의 최근값을 쓴다 — ACF(1)=0.968 로 지속성이 매우 강하다.
                sx = SARIMAXBand()
                exog = df[["feed_AB_tpd"]].ffill().bfill().to_numpy(float)
                try:
                    f_va = sx.rolling_forecast(flow_all, exog, int(o_all[fit].max()),
                                               o_all[va], h_all[va])
                    f_te = sx.rolling_forecast(flow_all, exog, int(o_all[tr].max()),
                                               o_all[te], h_all[te])
                except Exception:
                    continue
                if not np.isfinite(f_va).any() or not np.isfinite(f_te).any():
                    continue
                cva = conc_all[o_all[va]]
                cte = conc_all[o_all[te]]
                p_va = np.where(np.isfinite(f_va), f_va, flow_all[o_all[va]]) * cva / 100.0
                p_te = np.where(np.isfinite(f_te), f_te, flow_all[o_all[te]]) * cte / 100.0
            elif mname == "FlowConc":
                # CH₄ = 유량 × 농도 / 100. 두 성분을 각자 맞히고 곱한다.
                from sklearn.ensemble import HistGradientBoostingRegressor
                from sklearn.pipeline import Pipeline

                def _hgb():
                    return Pipeline([("est", HistGradientBoostingRegressor(
                        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                        l2_regularization=1.0, early_stopping=True,
                        random_state=C.RANDOM_STATE))])

                try:
                    tf_fit, tc_fit = flow_all[tstar[fit]], conc_all[tstar[fit]]
                    tf_tr, tc_tr = flow_all[tstar[tr]], conc_all[tstar[tr]]
                    _, pf_va = _fit_predict(_hgb(), Xfit, tf_fit, Xva)
                    _, pc_va = _fit_predict(_hgb(), Xfit, tc_fit, Xva)
                    _, pf_te = _fit_predict(_hgb(), Xtr, tf_tr, Xte)
                    _, pc_te = _fit_predict(_hgb(), Xtr, tc_tr, Xte)
                except Exception:
                    continue
                p_va = pf_va * np.clip(pc_va, *C.DQ["ranges"][C.CONC]) / 100.0
                p_te = pf_te * np.clip(pc_te, *C.DQ["ranges"][C.CONC]) / 100.0
            elif mname == "GRU":
                g = GRUForecaster(window=30).set_panel(gru_panel)
                Xfit_g, Xva_g = mk(fit, True), mk(va, True)
                Xtr_g, Xte_g = mk(tr, True), mk(te, True)
                try:
                    g.fit(Xfit_g, yfit)
                    p_va = g.predict(Xva_g)
                    g2 = GRUForecaster(window=30).set_panel(gru_panel).fit(Xtr_g, ytr)
                    p_te = g2.predict(Xte_g)
                except Exception:
                    continue
            else:
                modes = ("level", "delta") if mname in DELTA_FAMILIES else ("level",)
                best, best_e, best_i, best_mode = None, np.inf, -1, "level"
                for mode in modes:
                    yf = yfit if mode == "level" else yfit - a_fit
                    for i, cand in enumerate(cands):
                        try:
                            _, p = _fit_predict(cand, Xfit, yf, Xva)
                        except Exception:
                            continue
                        p = p if mode == "level" else p + a_va
                        e = _rmse(yva, p)
                        if e < best_e:
                            best, best_e, best_i, best_mode = cand, e, i, mode
                if best is None:
                    continue
                key = f"{mname}#{best_i}/{best_mode}"
                sel_hist[key] = sel_hist.get(key, 0) + 1
                yf = yfit if best_mode == "level" else yfit - a_fit
                _, p_va = _fit_predict(best, Xfit, yf, Xva)
                p_va = p_va if best_mode == "level" else p_va + a_va
                try:
                    yt_ = ytr if best_mode == "level" else ytr - a_tr
                    _, p_te = _fit_predict(best, Xtr, yt_, Xte)
                    p_te = p_te if best_mode == "level" else p_te + a_te
                except Exception:
                    continue
            val_preds[mname] = p_va
            test_preds[mname] = p_te
            per_fold[mname].append(_rmse(yte, p_te))
            oof[mname].append(p_te)

        # NNLS 스태킹 — 가중치는 검증블록의 폴드밖 예측으로만 학습
        if len(val_preds) >= 2:
            names = list(val_preds)
            Pv = np.column_stack([val_preds[k] for k in names])
            w = nnls_stack(Pv, yva)
            Pt = np.column_stack([test_preds[k] for k in names])
            p_stack = Pt @ w
            per_fold["Stack"].append(_rmse(yte, p_stack))
            oof["Stack"].append(p_stack)

        oof_y.append(yte)
        oof_t.append(o_all[te])
        oof_h.append(h_all[te])
        oof_fold.append(np.full(len(te), fi))
        if verbose:
            print(f"    [{name}] fold {fi:2d} O={O:4d} tr={len(tr):6d} te={len(te):4d} "
                  f"naive={per_fold['Naive'][-1]:7.1f} "
                  f"best={min((v[-1] for k, v in per_fold.items() if v and k not in ('Naive','SeasonalNaive')), default=np.nan):7.1f}",
                  flush=True)

    if not oof_y:
        return {"band": name, "error": "no folds"}

    Y = np.concatenate(oof_y)
    res = {"band": name, "lo": lo, "hi": hi, "n_folds": len(oof_y),
           "n_test": int(len(Y)), "y_mean": round(float(Y.mean()), 1),
           "y_sd": round(float(Y.std()), 1), "models": {}, "selected_candidates": sel_hist}

    naive_pool = np.concatenate(oof["Naive"])
    naive_rmse_pool = _rmse(Y, naive_pool)
    res["naive"] = {"pooled_R2": round(float(r2_score(Y, naive_pool)), 4),
                    "pooled_RMSE": round(naive_rmse_pool, 1),
                    "CV_RMSE": round(float(np.mean(per_fold["Naive"])), 1)}

    from scipy import stats
    for mname in per_fold:
        if mname == "Naive" or not oof[mname]:
            continue
        P = np.concatenate(oof[mname])
        errs = np.asarray(per_fold[mname], float)
        nerr = np.asarray(per_fold["Naive"], float)[: len(errs)]
        t_p = None
        if len(errs) > 1 and not np.allclose(errs, nerr):
            t_p = float(stats.ttest_rel(errs, nerr).pvalue)
        res["models"][mname] = {
            "pooled_R2": round(float(r2_score(Y, P)), 4),
            "pooled_RMSE": round(_rmse(Y, P), 1),
            "pooled_MAE": round(float(mean_absolute_error(Y, P)), 1),
            "pooled_MAPE": round(float(np.mean(np.abs((Y - P) / Y)) * 100), 2),
            "RMSSE_pct": round(100 * _rmse(Y, P) / naive_rmse_pool, 1),
            "CV_RMSE": round(float(errs.mean()), 1),
            "p_vs_naive": None if t_p is None else round(t_p, 4),
        }
    res["models"] = dict(sorted(res["models"].items(), key=lambda kv: -kv[1]["pooled_R2"]))
    res["_oof"] = {"y": Y, "t": np.concatenate(oof_t), "h": np.concatenate(oof_h),
                   "fold": np.concatenate(oof_fold),
                   "pred": {k: np.concatenate(v) for k, v in oof.items() if v}}
    return res
