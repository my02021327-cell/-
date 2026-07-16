"""
소화조 건강상태(신호등) 관제 + 메탄생성량 예측 시각화

■ 건강상태는 VFA/알칼리도 비(산성화·완충능 지표)로 정의한다(AD 문헌 밴드).
    안정(정상) : VFA/ALK < 0.3
    주의        : 0.3 ≤ VFA/ALK < 0.4
    불안정(점검): 0.4 ≤ VFA/ALK < 0.8
    위험        : VFA/ALK ≥ 0.8
  수율(효율)이 정상이어도 완충능이 소진(VFA/ALK 상승)되면 사전 경보한다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["axes.unicode_minus"] = False

HEALTH_BANDS = [
    ("Stable",   0.00, 0.30, "#22c55e"),
    ("Caution",  0.30, 0.40, "#f5b301"),
    ("Unstable", 0.40, 0.80, "#f97316"),
    ("Danger",   0.80, 9.99, "#ef4444"),
]


def classify_health(vfa_alk: float) -> str:
    if vfa_alk < 0.30:
        return "Stable"
    if vfa_alk < 0.40:
        return "Caution"
    if vfa_alk < 0.80:
        return "Unstable"
    return "Danger"


def health_summary(pred_csv: str) -> dict:
    df = pd.read_csv(pred_csv)
    grades = df["VFA_ALK"].apply(classify_health)
    order = ["Stable", "Caution", "Unstable", "Danger"]
    return {g: int((grades == g).sum()) for g in order}


def plot_dashboard(pred_csv: str, out_png: str):
    """2023 홀드아웃 : 메탄생성량 실측 vs 예측 + 소화조 건강(VFA/ALK) 밴드."""
    df = pd.read_csv(pred_csv, parse_dates=["date"]).sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # (1) 메탄생성량 예측
    ax = axes[0]
    ax.plot(df["date"], df["methane_true"], color="#334155", lw=1.5, label="Actual")
    ax.plot(df["date"], df["methane_pred"], color="#0ea5e9", lw=1.5, ls="--", label="Ensemble pred")
    ax.fill_between(df["date"], df["methane_true"], df["methane_pred"], color="#0ea5e9", alpha=0.08)
    ax.set_ylabel("Methane (Nm3 CH4/d)")
    ax.set_title("2023 Holdout : Methane production forecast")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # (2) 소화조 건강 : VFA/ALK + 밴드
    ax = axes[1]
    for _, lo, hi, color in HEALTH_BANDS:
        ax.axhspan(lo, min(hi, df["VFA_ALK"].max() * 1.2 + 0.1), color=color, alpha=0.08)
    for y in (0.30, 0.40, 0.80):
        ax.axhline(y, color="grey", lw=0.8, ls=":")
    ax.plot(df["date"], df["VFA_ALK"], color="#b91c1c", lw=1.5, label="VFA/Alkalinity")
    ax.set_ylabel("VFA / Alkalinity")
    ax.set_title("2023 Holdout : Digester health (acidification) — VFA/ALK signal")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(root, "outputs", "predictions_2023.csv")
    png = os.path.join(root, "outputs", "predictions_2023.png")
    plot_dashboard(csv, png)
    print("건강상태 분포:", health_summary(csv))
    print("그래프 저장 →", png)
