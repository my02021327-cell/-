"""
BioGuard-AI 앙상블 학습·평가 파이프라인

절차
 1) 데이터 로딩·병합
 2) 전처리 + 시계열 Feature Engineering
 3) 2018~2022 학습 / 2023 홀드아웃 분할
 4) Base Model 학습 + TimeSeriesSplit OOF 예측
 5) NNLS 비음수 스태킹 메타러너 학습
 6) 2023 홀드아웃에서 개별 모델 · 앙상블 성능 평가 (R², RMSE, MAE)
 7) 결과 저장 (outputs/metrics.json, outputs/predictions_2023.csv)

실행 : python -m src.train   (또는 python src/train.py)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (  # noqa: E402
    TARGET_COLS,
    build_features,
    get_feature_columns,
    load_raw,
    train_holdout_split,
)
from src.ensemble import NNLSStackingEnsemble  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
TARGET_LABELS = {"methane": "메탄발생량", "MY": "메탄발생비(수율)"}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """종속변수별 R²/RMSE/MAE 계산."""
    out = {}
    for j, col in enumerate(TARGET_COLS):
        yt, yp = y_true[:, j], y_pred[:, j]
        out[col] = {
            "R2": round(float(r2_score(yt, yp)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "MAE": round(float(mean_absolute_error(yt, yp)), 4),
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    # 1) 로딩
    raw = load_raw(
        os.path.join(DATA_DIR, "master.xlsx"),
        os.path.join(DATA_DIR, "targets.xlsx"),
    )

    # 2) 전처리 + 피처
    feat = build_features(raw)
    feature_cols = get_feature_columns(feat)

    # 3) 분할 (2018~2022 학습 / 2023 홀드아웃)
    train_df, test_df = train_holdout_split(feat, holdout_year=2023)
    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    Y_tr = train_df[TARGET_COLS].to_numpy(dtype=float)
    X_te = test_df[feature_cols].to_numpy(dtype=float)
    Y_te = test_df[TARGET_COLS].to_numpy(dtype=float)

    # 메타 검증 블록 = 학습기간 내 '가장 최근 연도'(2022) → 미래 일반화에 맞춘 가중치 학습
    meta_year = int(train_df["year"].max())
    meta_mask = (train_df["year"] == meta_year).to_numpy()

    print(f"피처 수 : {len(feature_cols)}")
    print(f"학습(2018~2022) : {X_tr.shape[0]} set / 홀드아웃(2023) : {X_te.shape[0]} set")
    print(f"메타 검증 블록 : {meta_year}년 {int(meta_mask.sum())} set")

    # 4) NNLS 비음수 스태킹 앙상블 학습
    print("\n[1/2] Base Model 학습 + 메타 검증 블록 예측 중...")
    ens = NNLSStackingEnsemble()
    ens.fit(X_tr, Y_tr, meta_mask)

    print("[2/2] NNLS 비음수 가중치 결합 후 2023 홀드아웃 예측 중...")
    base_test_pred = ens.base_predict(X_te)
    ens_test_pred = ens.predict(X_te)

    # 5) 평가
    results = {name: evaluate(Y_te, pred) for name, pred in base_test_pred.items()}
    results["Ensemble_NNLS"] = evaluate(Y_te, ens_test_pred)

    # NNLS 가중치 정리 (정규화 비중)
    norm_w = ens.normalized_weights()
    weight_report = {col: norm_w[j] for j, col in enumerate(TARGET_COLS)}

    # 7) 출력
    print("\n================ 2023 홀드아웃 성능 ================")
    header = f"{'Model':<16}"
    for col in TARGET_COLS:
        header += f"| {TARGET_LABELS[col]:<14} R²   RMSE     "
    print(header)
    for name, r in results.items():
        line = f"{name:<16}"
        for col in TARGET_COLS:
            line += f"|   {r[col]['R2']:>6.3f}  {r[col]['RMSE']:>8.2f}  "
        print(line)

    print("\n---------------- NNLS 비음수 가중치(정규화) ----------------")
    for col in TARGET_COLS:
        print(f"[{TARGET_LABELS[col]}]", weight_report[col])

    # 저장
    metrics = {
        "n_features": len(feature_cols),
        "n_train": int(X_tr.shape[0]),
        "n_holdout": int(X_te.shape[0]),
        "holdout_year": 2023,
        "results": results,
        "nnls_weights_normalized": weight_report,
        "feature_columns": feature_cols,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 예측 CSV
    pred_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "methane_true": Y_te[:, 0],
            "methane_pred": ens_test_pred[:, 0],
            "MY_true": Y_te[:, 1],
            "MY_pred": ens_test_pred[:, 1],
        }
    )
    pred_df.to_csv(
        os.path.join(OUT_DIR, "predictions_2023.csv"), index=False, encoding="utf-8-sig"
    )

    pred_csv = os.path.join(OUT_DIR, "predictions_2023.csv")
    print(f"\n저장 완료 → {os.path.join(OUT_DIR, 'metrics.json')}")
    print(f"          → {pred_csv}")

    # 8) 신호등 관제 요약 + 시각화 (선택)
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
    except Exception as exc:  # matplotlib 미설치 등
        print(f"(시각화 생략: {exc})")

    return metrics


if __name__ == "__main__":
    main()
