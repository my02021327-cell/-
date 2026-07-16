"""
신호등(Traffic-light) 관제 로직 + 예측 결과 시각화

메탄발생비(수율) 기준선 (보고서)
  정상  : MY >= 0.7
  주의  : 0.6 <= MY < 0.7
  점검  : 0.5 <= MY < 0.6
  위험  : MY < 0.5
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 폰트 : 한글 미설치 환경 대비, 라벨은 영문 병기 사용
plt.rcParams["axes.unicode_minus"] = False

SIGNAL_BANDS = [
    ("Normal (>=0.70)", 0.70, np.inf, "#2e7d32"),
    ("Caution (0.60-0.70)", 0.60, 0.70, "#f9a825"),
    ("Inspect (0.50-0.60)", 0.50, 0.60, "#ef6c00"),
    ("Danger (<0.50)", -np.inf, 0.50, "#c62828"),
]


def classify_signal(my_value: float) -> str:
    """수율 값 → 신호등 등급."""
    if my_value >= 0.70:
        return "Normal"
    if my_value >= 0.60:
        return "Caution"
    if my_value >= 0.50:
        return "Inspect"
    return "Danger"


def plot_predictions(pred_csv: str, out_png: str):
    """2023 홀드아웃 실측 vs 예측 시계열 + 신호등 밴드 시각화."""
    df = pd.read_csv(pred_csv, parse_dates=["date"]).sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # (1) 메탄발생량
    ax = axes[0]
    ax.plot(df["date"], df["methane_true"], color="#455a64", lw=1.4, label="Actual")
    ax.plot(df["date"], df["methane_pred"], color="#1e88e5", lw=1.4, ls="--", label="Ensemble pred")
    ax.set_ylabel("Methane (Nm3 CH4/d)")
    ax.set_title("2023 Holdout : Methane production (methane)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # (2) 메탄발생비(수율) + 신호등 밴드
    ax = axes[1]
    for label, lo, hi, color in SIGNAL_BANDS:
        ax.axhspan(max(lo, 0), min(hi, 2.0), color=color, alpha=0.08)
    for y in (0.5, 0.6, 0.7):
        ax.axhline(y, color="grey", lw=0.8, ls=":")
    ax.plot(df["date"], df["MY_true"], color="#455a64", lw=1.4, label="Actual")
    ax.plot(df["date"], df["MY_pred"], color="#d81b60", lw=1.4, ls="--", label="Ensemble pred")
    ax.set_ylabel("Yield (Nm3 CH4/kgVS)")
    ax.set_title("2023 Holdout : Methane yield (MY) with signal bands")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def signal_summary(pred_csv: str) -> pd.DataFrame:
    """예측/실측 각각의 신호등 등급 분포 + 등급 정확도."""
    df = pd.read_csv(pred_csv)
    df["signal_true"] = df["MY_true"].apply(classify_signal)
    df["signal_pred"] = df["MY_pred"].apply(classify_signal)
    order = ["Normal", "Caution", "Inspect", "Danger"]
    summary = pd.DataFrame(
        {
            "actual_days": df["signal_true"].value_counts().reindex(order, fill_value=0),
            "pred_days": df["signal_pred"].value_counts().reindex(order, fill_value=0),
        }
    )
    acc = float((df["signal_true"] == df["signal_pred"]).mean())
    return summary, acc


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(root, "outputs", "predictions_2023.csv")
    png = os.path.join(root, "outputs", "predictions_2023.png")
    plot_predictions(csv, png)
    summ, acc = signal_summary(csv)
    print("신호등 등급 분포 (2023):")
    print(summ.to_string())
    print(f"\n신호등 등급 예측 정확도 : {acc:.1%}")
    print(f"그래프 저장 → {png}")
