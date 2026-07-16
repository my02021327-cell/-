"""
EDA : 운전변수–수율(MY) Spearman 상관분석 (보고서 상관분석 재현·검증)

보고서 주요 결과
  투입 VS ρ≈-0.78 / OLR(VS부하) ρ≈-0.67 / VFA/ALK ρ≈-0.18 / 소화조 온도 ρ≈+0.01
  투입 pH·알칼리도는 양(+)의 상관.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import TARGET_COLS, build_features, load_raw

# 상관분석 대상 운전변수 (라벨: 영문 병기)
CAND_VARS = {
    "유입_VS": "Feed VS (invest VS)",
    "VS_in": "VS load (OLR proxy)",
    "VS_load_ratio": "VS load ratio",
    "소화조_VFA": "Digester VFA",
    "VFA_TAlk_ratio": "VFA/Alkalinity",
    "소화조_TAlk": "Total Alkalinity",
    "소화조_pH": "Digester pH",
    "유입_pH": "Feed pH",
    "소화조_온도": "Digester temp.",
    "투입량합계": "Total feed",
}


def spearman_table(df: pd.DataFrame, target: str = "MY") -> pd.DataFrame:
    rows = []
    for col, label in CAND_VARS.items():
        if col in df.columns:
            rho = df[col].corr(df[target], method="spearman")
            rows.append({"variable": col, "label": label, "spearman_rho": round(float(rho), 3)})
    out = pd.DataFrame(rows).sort_values("spearman_rho")
    return out.reset_index(drop=True)


def plot_corr(tbl: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c62828" if v < 0 else "#1565c0" for v in tbl["spearman_rho"]]
    ax.barh(tbl["label"], tbl["spearman_rho"], color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Spearman rho  (vs Methane yield MY)")
    ax.set_title("Operating variables vs methane yield — Spearman correlation")
    for i, v in enumerate(tbl["spearman_rho"]):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    raw = load_raw(os.path.join(root, "data", "master.xlsx"),
                   os.path.join(root, "data", "targets.xlsx"))
    feat = build_features(raw)
    lab = feat.dropna(subset=TARGET_COLS)

    tbl = spearman_table(lab, "MY")
    tbl.to_csv(os.path.join(out_dir, "correlation_MY.csv"), index=False, encoding="utf-8-sig")
    plot_corr(tbl, os.path.join(out_dir, "correlation_MY.png"))

    print("운전변수–수율(MY) Spearman 상관 (오름차순):")
    print(tbl.to_string(index=False))
    print(f"\n저장 → {os.path.join(out_dir, 'correlation_MY.csv')}")
    print(f"     → {os.path.join(out_dir, 'correlation_MY.png')}")


if __name__ == "__main__":
    main()
