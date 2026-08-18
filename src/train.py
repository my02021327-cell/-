"""
BioGuard-AI 앙상블 학습·평가 파이프라인
목표 : 메탄생성량(단일) 예측 + 소화조 건강상태(VFA/ALK) 관제

Base : RandomForest·MLP(테이블) + LSTM·Transformer(시퀀스)  →  NNLS 비음수 스태킹
분할 : 2018~2022 학습 / 2023 홀드아웃, 학습기간 내 2022 = 메타 검증블록

■ 수율(MY)은 예측하지 않는다(분모가 금일 투입 VS이므로 예측이 순환적).
  대신 메탄생성량을 예측하고, 소화조 건강은 VFA/알칼리도 비로 관제한다.

실행 : python -m src.train
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (  # noqa: E402
    TARGET_COL,
    build_features,
    get_feature_columns,
    load_raw,
)
from src.ensemble import NNLSMetaCombiner, make_tabular_models  # noqa: E402
from src.metrics import score  # noqa: E402
from src.sequences import build_sequences  # noqa: E402
from src.torch_models import make_lstm, make_transformer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
WINDOW = 14
HOLDOUT_YEAR = 2023


def evaluate(y_true, y_pred, y_prev=None, gap=None) -> dict:
    """
    R²·RMSE·MAE 에 더해 naive(persistence) 대비 RMSSE 를 함께 낸다.

    절대 R² 만 보고하면 자기상관이 강한 계열에서 성능을 크게 과대평가한다.
    Meola & Weinrich (2025) 의 주지표를 병기해 '전일값 하나보다 나은가'를 즉시 보이게 한다
    (EXPERT_REVIEW §A3 정규화 지표 공백). RMSSE ≥ 100 % 는 채택 불가를 뜻한다.
    """
    return score(y_true, y_pred, y_prev, gap)


def _fit_predict(proto_or_factory, is_torch, X_fit, Y_fit, X_pred):
    mdl = proto_or_factory() if is_torch else clone(proto_or_factory)
    mdl.fit(X_fit, Y_fit)
    return np.asarray(mdl.predict(X_pred)).reshape(-1, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    raw = load_raw(os.path.join(DATA_DIR, "master.xlsx"), os.path.join(DATA_DIR, "targets.xlsx"))
    daily = build_features(raw)
    feature_cols = get_feature_columns(daily)

    # 시퀀스(캐노니컬 라벨 집합)
    Xseq, Yseq, yrs, dates, seq_fc = build_sequences(daily, window=WINDOW)  # Yseq:(n,1)

    # 테이블 표현을 시퀀스와 동일 날짜로 정합
    lab = daily.dropna(subset=[TARGET_COL]).set_index("date")
    dt_index = pd.to_datetime(dates)
    Xtab = lab.loc[dt_index, feature_cols].to_numpy(dtype=float)
    Y = lab.loc[dt_index, TARGET_COL].to_numpy(dtype=float).reshape(-1, 1)
    assert np.allclose(Y, Yseq, atol=1e-6), "테이블/시퀀스 타깃 정합 실패"

    # 진단·건강 참조값(관측)
    diag = lab.loc[dt_index, ["MY", "VS_in", "VFA_ALK"]].reset_index(drop=True)

    train_mask = yrs < HOLDOUT_YEAR
    test_mask = yrs == HOLDOUT_YEAR
    # 메타 검증블록 = 학습기간의 최근 2년(국면이동 전이 안정화). base 는 그 이전으로 선학습.
    meta_start = int(yrs[train_mask].max()) - 1
    meta_mask = train_mask & (yrs >= meta_start)
    early_mask = train_mask & (yrs < meta_start)

    print(f"타깃: 메탄생성량(단일) | 테이블 피처 {len(feature_cols)}개 / "
          f"시퀀스 피처 {len(seq_fc)}개 (W={WINDOW})")
    print(f"학습 {int(train_mask.sum())} / 홀드아웃(2023) {int(test_mask.sum())} "
          f"/ 메타블록({meta_start}~{int(yrs[train_mask].max())}) {int(meta_mask.sum())}")

    tab_models = make_tabular_models()
    seq_models = {
        "LSTM": lambda: make_lstm(n_seeds=5),
        "Transformer": lambda: make_transformer(n_seeds=5),
    }

    meta_preds, test_preds = {}, {}
    print("\n[1/2] Base 학습 · 메타블록/홀드아웃 예측 중...")
    for name, est in tab_models.items():
        meta_preds[name] = _fit_predict(est, False, Xtab[early_mask], Y[early_mask], Xtab[meta_mask])
        test_preds[name] = _fit_predict(est, False, Xtab[train_mask], Y[train_mask], Xtab[test_mask])
        print(f"  - {name} 완료")
    for name, fac in seq_models.items():
        meta_preds[name] = _fit_predict(fac, True, Xseq[early_mask], Y[early_mask], Xseq[meta_mask])
        test_preds[name] = _fit_predict(fac, True, Xseq[train_mask], Y[train_mask], Xseq[test_mask])
        print(f"  - {name} 완료")

    print("[2/2] NNLS 비음수 스태킹 결합 중...")
    meta = NNLSMetaCombiner().fit(meta_preds, Y[meta_mask])
    ens_test = meta.combine(test_preds)

    Y_te = Y[test_mask]

    # naive(persistence) 기준선 : 직전 '관측' 메탄과 그 간격
    y_all = Y.ravel()
    prev_all = np.concatenate([[np.nan], y_all[:-1]])
    gap_all = np.concatenate([[np.nan], np.diff(dt_index.values).astype("timedelta64[D]").astype(float)])
    y_prev_te, gap_te = prev_all[test_mask], gap_all[test_mask]

    results = {name: evaluate(Y_te, pred, y_prev_te, gap_te) for name, pred in test_preds.items()}
    results["Ensemble_NNLS"] = evaluate(Y_te, ens_test, y_prev_te, gap_te)
    results["Naive_persistence"] = evaluate(Y_te[1:], y_prev_te[1:], y_prev_te[1:], gap_te[1:])
    weight_report = {"methane": meta.normalized_weights()[0]}

    print("\n================ 2023 홀드아웃 : 메탄생성량 ================")
    print(f"{'Model':<18}|   R²     RMSE      MAE    RMSSE%   RMSSE%(gap1)")
    for name, r in results.items():
        rs = r.get("RMSSE_pct")
        rs1 = r.get("RMSSE_pct_gap1")
        print(f"{name:<18}| {r['R2']:>7.3f}  {r['RMSE']:>8.1f}  {r['MAE']:>8.1f}  "
              f"{(f'{rs:8.1f}' if rs is not None else '       —')}  "
              f"{(f'{rs1:10.1f}' if rs1 is not None else '         —')}")
    print("  * RMSSE = 모델 RMSE / naive RMSE. 100% 이상이면 전일값 예측기보다 못하다 → 채택 불가.")
    print("\n---------------- NNLS 비음수 가중치(정규화) ----------------")
    print(weight_report["methane"])

    metrics = {
        "target": "methane",
        "n_tab_features": len(feature_cols),
        "n_seq_features": len(seq_fc),
        "window": WINDOW,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(test_mask.sum()),
        "holdout_year": HOLDOUT_YEAR,
        "base_models": ["RandomForest", "MLP", "LSTM", "Transformer"],
        "results": results,
        "nnls_weights_normalized": weight_report,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 예측 + 건강 진단 CSV
    te_diag = diag[test_mask].reset_index(drop=True)
    pred_df = pd.DataFrame({
        "date": dt_index[test_mask],
        "methane_true": Y_te.ravel(),
        "methane_pred": ens_test.ravel(),
        "VS_in": te_diag["VS_in"].values,
        "MY_observed": te_diag["MY"].values,         # 진단(관측 효율)
        "VFA_ALK": te_diag["VFA_ALK"].values,          # 건강 지표
    })
    pred_csv = os.path.join(OUT_DIR, "predictions_2023.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료 → {os.path.join(OUT_DIR, 'metrics.json')}")
    print(f"          → {pred_csv}")

    # 건강상태 관제 요약 + 시각화
    try:
        from src.signal import health_summary, plot_dashboard

        png = os.path.join(OUT_DIR, "predictions_2023.png")
        plot_dashboard(pred_csv, png)
        summ = health_summary(pred_csv)
        metrics["health_distribution_2023"] = summ
        with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("\n---------- 소화조 건강상태(VFA/ALK) 분포 (2023) ----------")
        for k, v in summ.items():
            print(f"  {k:16s}: {v} 일")
        print(f"그래프 저장 → {png}")
    except Exception as exc:
        print(f"(시각화 생략: {exc})")

    return metrics


if __name__ == "__main__":
    main()
