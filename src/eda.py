"""
EDA : 생물학적 메커니즘 정량 분석
 (1) 투입 VS부하 → 메탄생성량 : 시차 교차상관(생물학적 지연) + 체류창 누적
 (2) 소화조 내부상태 → 메탄생성량 상관
 (3) VFA/알칼리도 건강 밴드 분포

혐기성소화의 다단계 미생물 반응(가수분해→산생성→메탄생성)으로 투입 유기물이
체류시간만큼 지연되어 메탄으로 전환됨을 데이터로 확인한다.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_raw

STATE_VARS = ["소화조_VS", "소화조_온도", "소화조_TS", "소화조_VFA", "소화조_pH", "소화조_CODcr"]


def _prep(root: str) -> pd.DataFrame:
    raw = load_raw(os.path.join(root, "data", "master.xlsx"),
                   os.path.join(root, "data", "targets.xlsx")).sort_values("date").reset_index(drop=True)
    cols = ["VS_in", "투입량합계", "methane", "소화조_VFA", "소화조_TAlk"] + STATE_VARS
    cols = list(dict.fromkeys(cols))
    raw[cols] = raw[cols].interpolate(limit_direction="both")
    raw["VFA_ALK"] = raw["소화조_VFA"] / raw["소화조_TAlk"]
    return raw


def lag_crosscorr(df: pd.DataFrame, max_lag: int = 21) -> pd.DataFrame:
    rows = []
    for k in range(0, max_lag + 1):
        rows.append({"lag_days": k, "pearson_r": round(float(df["methane"].corr(df["VS_in"].shift(k))), 3)})
    return pd.DataFrame(rows)


def loading_window_corr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in [1, 3, 5, 7, 10, 14, 21, 30]:
        r = df["methane"].corr(df["VS_in"].rolling(w, min_periods=1).mean())
        rows.append({"window_days": w, "pearson_r": round(float(r), 3)})
    return pd.DataFrame(rows)


def health_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    v = df["VFA_ALK"].dropna()
    bands = [("Stable(<0.3)", (v < 0.3)), ("Caution(0.3-0.4)", (v >= 0.3) & (v < 0.4)),
             ("Unstable(0.4-0.8)", (v >= 0.4) & (v < 0.8)), ("Danger(>=0.8)", (v >= 0.8))]
    return pd.DataFrame([{"band": n, "days": int(m.sum()), "pct": round(m.mean() * 100, 1)} for n, m in bands])


def plot_biology(df: pd.DataFrame, out_png: str):
    lag = lag_crosscorr(df)
    win = loading_window_corr(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(lag["lag_days"], lag["pearson_r"], "-o", color="#0ea5e9", ms=4)
    ax.set_xlabel("Lag of VS load (days)")
    ax.set_ylabel("Pearson r with methane_t")
    ax.set_title("Biological delay: VS load lag vs methane")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(win["window_days"], win["pearson_r"], "-o", color="#16a34a", ms=4)
    best = win.loc[win["pearson_r"].idxmax()]
    ax.axvline(best["window_days"], color="#b91c1c", ls=":", lw=1)
    ax.set_xlabel("Cumulative loading window (days)")
    ax.set_ylabel("Pearson r with methane_t")
    ax.set_title(f"Retention window: best ~{int(best['window_days'])}d (r={best['pearson_r']})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    df = _prep(root)

    lag = lag_crosscorr(df)
    win = loading_window_corr(df)
    best_w = int(win.loc[win["pearson_r"].idxmax(), "window_days"])
    state = pd.DataFrame([{"variable": v, "spearman_r": round(float(df["methane"].corr(df[v], method="spearman")), 3)}
                          for v in STATE_VARS]).sort_values("spearman_r", ascending=False)
    health = health_band_distribution(df)

    lag.to_csv(os.path.join(out_dir, "eda_vsload_lag.csv"), index=False, encoding="utf-8-sig")
    win.to_csv(os.path.join(out_dir, "eda_loading_window.csv"), index=False, encoding="utf-8-sig")
    health.to_csv(os.path.join(out_dir, "eda_health_bands.csv"), index=False, encoding="utf-8-sig")
    plot_biology(df, os.path.join(out_dir, "eda_biology.png"))

    print("■ 투입 VS부하 → 메탄 : 순간(lag0) r=%.3f, 최적 누적창 %d일 r=%.3f"
          % (lag.loc[0, "pearson_r"], best_w, win["pearson_r"].max()))
    print("  → 생물학적 지연(체류시간) 존재. 순간값보다 누적 부하가 더 강한 상관.")
    print("\n■ 소화조 내부상태 vs 메탄 (Spearman):")
    print(state.to_string(index=False))
    print("\n■ VFA/알칼리도 건강 밴드 분포:")
    print(health.to_string(index=False))
    print(f"\n저장 → {out_dir}/eda_*.csv, eda_biology.png")


if __name__ == "__main__":
    main()
