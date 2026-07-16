"""
BioGuard-AI 앙상블 학습·평가 파이프라인 (RF·MLP·LSTM·Transformer + NNLS 스태킹)

절차
 1) 데이터 로딩·병합 → 전처리 + 시계열 Feature Engineering(테이블) + 시퀀스 생성
 2) 테이블/시퀀스 표현을 동일 라벨 날짜로 정합
 3) 2018~2022 학습 / 2023 홀드아웃, 학습기간 내 2022 = 메타 검증블록
 4) Base : RandomForest·MLP(테이블) + LSTM·Transformer(시퀀스)
    - 메타블록 이전으로 학습 → 메타블록 예측(가중치 학습)
    - 전체 학습데이터로 재학습 → 2023 홀드아웃 예측
 5) NNLS 비음수 스태킹으로 결합, 개별·앙상블 성능 평가(R²/RMSE/MAE)
 6) 결과 저장(outputs/metrics.json, predictions_2023.csv, 그래프)

실행 : python -m src.train
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TARGET_COLS, build_features, get_feature_columns, load_raw  # noqa: E402
from src.ensemble import NNLSMetaCombiner, make_tabular_models  # noqa: E402
from src.sequences import build_sequences  # noqa: E402
from src.torch_models import make_lstm, make_transformer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
TARGET_LABELS = {"methane": "메탄발생량", "MY": "메탄발생비(수율)"}
WINDOW = 14
HOLDOUT_YEAR = 2023


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for j, col in enumerate(TARGET_COLS):
        yt, yp = y_true[:, j], y_pred[:, j]
        out[col] = {
            "R2": round(float(r2_score(yt, yp)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "MAE": round(float(mean_absolute_error(yt, yp)), 4),
        }
    return out


def _fit_predict(proto_or_factory, is_torch, X_fit, Y_fit, X_pred):
    """base 하나를 학습 후 예측. torch 모델은 factory 호출, sklearn 은 clone."""
    mdl = proto_or_factory() if is_torch else clone(proto_or_factory)
    mdl.fit(X_fit, Y_fit)
    return mdl.predict(X_pred)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    # 1) 로딩 + 피처
    raw = load_raw(os.path.join(DATA_DIR, "master.xlsx"), os.path.join(DATA_DIR, "targets.xlsx"))
    daily = build_features(raw)
    feature_cols = get_feature_columns(daily)

    # 시퀀스(캐노니컬 라벨 집합) 생성
    Xseq, Yseq, yrs, dates, seq_fc = build_sequences(daily, window=WINDOW)

    # 2) 테이블 표현을 시퀀스와 동일 날짜로 정합
    lab = daily.dropna(subset=TARGET_COLS).set_index("date")
    Xtab = lab.loc[pd.to_datetime(dates), feature_cols].to_numpy(dtype=float)
    Ytab = lab.loc[pd.to_datetime(dates), TARGET_COLS].to_numpy(dtype=float)
    assert np.allclose(Ytab, Yseq, atol=1e-6), "테이블/시퀀스 타깃 정합 실패"
    Y = Yseq.astype(float)

    # 3) 분할 마스크
    train_mask = yrs < HOLDOUT_YEAR
    test_mask = yrs == HOLDOUT_YEAR
    meta_year = int(yrs[train_mask].max())
    meta_mask = yrs == meta_year                 # 메타 검증블록(전체 기준)
    early_mask = train_mask & (yrs < meta_year)  # base 선학습(메타블록 이전)

    print(f"테이블 피처 {len(feature_cols)}개 / 시퀀스 피처 {len(seq_fc)}개 (W={WINDOW})")
    print(f"학습 {int(train_mask.sum())} / 홀드아웃(2023) {int(test_mask.sum())} "
          f"/ 메타블록({meta_year}) {int(meta_mask.sum())}")

    # 4) Base 정의 (테이블 / 시퀀스)
    tab_models = make_tabular_models()  # {name: estimator}
    # 시퀀스 모델은 학습 분산이 크므로 시드 5개 평균으로 안정화
    seq_models = {
        "LSTM": lambda: make_lstm(n_seeds=5),
        "Transformer": lambda: make_transformer(n_seeds=5),
    }  # {name: factory}

    def inputs(mask, kind):
        return (Xtab[mask] if kind == "tab" else Xseq[mask])

    meta_preds, test_preds = {}, {}

    print("\n[1/2] Base 학습 · 메타블록/홀드아웃 예측 중...")
    # 테이블형
    for name, est in tab_models.items():
        meta_preds[name] = _fit_predict(est, False, Xtab[early_mask], Y[early_mask], Xtab[meta_mask])
        test_preds[name] = _fit_predict(est, False, Xtab[train_mask], Y[train_mask], Xtab[test_mask])
        print(f"  - {name} 완료")
    # 시퀀스형(torch)
    for name, fac in seq_models.items():
        meta_preds[name] = _fit_predict(fac, True, Xseq[early_mask], Y[early_mask], Xseq[meta_mask])
        test_preds[name] = _fit_predict(fac, True, Xseq[train_mask], Y[train_mask], Xseq[test_mask])
        print(f"  - {name} 완료")

    # 5) NNLS 메타 결합
    print("[2/2] NNLS 비음수 스태킹 결합 중...")
    meta = NNLSMetaCombiner().fit(meta_preds, Y[meta_mask])
    ens_test = meta.combine(test_preds)

    Y_te = Y[test_mask]
    results = {name: evaluate(Y_te, pred) for name, pred in test_preds.items()}
    results["Ensemble_NNLS"] = evaluate(Y_te, ens_test)
    norm_w = meta.normalized_weights()
    weight_report = {col: norm_w[j] for j, col in enumerate(TARGET_COLS)}

    # 출력
    print("\n================ 2023 홀드아웃 성능 ================")
    print(f"{'Model':<16}| 메탄발생량 R²  RMSE    | 수율 R²   RMSE")
    for name, r in results.items():
        print(f"{name:<16}|   {r['methane']['R2']:>6.3f}  {r['methane']['RMSE']:>7.1f}  "
              f"|  {r['MY']['R2']:>6.3f}  {r['MY']['RMSE']:>6.3f}")
    print("\n---------------- NNLS 비음수 가중치(정규화) ----------------")
    for col in TARGET_COLS:
        print(f"[{TARGET_LABELS[col]}]", weight_report[col])

    # 6) 저장
    metrics = {
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

    pred_csv = os.path.join(OUT_DIR, "predictions_2023.csv")
    pd.DataFrame({
        "date": pd.to_datetime(dates[test_mask]),
        "methane_true": Y_te[:, 0], "methane_pred": ens_test[:, 0],
        "MY_true": Y_te[:, 1], "MY_pred": ens_test[:, 1],
    }).to_csv(pred_csv, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료 → {os.path.join(OUT_DIR, 'metrics.json')}")
    print(f"          → {pred_csv}")

    # 7) 신호등 요약 + 시각화
    try:
        from src.signal import plot_predictions, signal_summary

        png = os.path.join(OUT_DIR, "predictions_2023.png")
        plot_predictions(pred_csv, png)
        summ, acc = signal_summary(pred_csv)
        metrics["signal_light_accuracy"] = round(acc, 4)
        with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("\n---------------- 신호등 관제 등급 분포 (2023) ----------------")
        print(summ.to_string())
        print(f"신호등 등급 예측 정확도 : {acc:.1%}")
        print(f"그래프 저장 → {png}")
    except Exception as exc:
        print(f"(시각화 생략: {exc})")

    return metrics


if __name__ == "__main__":
    main()
