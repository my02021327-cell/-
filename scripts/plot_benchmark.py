"""
벤치마크 결과 그림 — outputs/benchmark_rmsse.png

좌: 체제별 모델 RMSSE (100 % = naive 동률선)
우: h90 리드타임 층화 RMSSE

축·범례는 ASCII 로만 쓴다(한글 폰트가 없는 실행 환경에서 두부 현상 방지).
실행 : python scripts/plot_benchmark.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(ROOT, "outputs", "benchmark_meola2025.json")
PNG = os.path.join(ROOT, "outputs", "benchmark_rmsse.png")
LEADS = ["1-3", "4-7", "8-14", "15-30", "31-90"]


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    regimes = [r for r in ("od1", "h90") if r in d]
    names = list(d[regimes[0]]["models"])
    width = 0.38
    xs = np.arange(len(names))
    for i, reg in enumerate(regimes):
        vals = [d[reg]["models"][n]["RMSSE_pct"] for n in names]
        ax.bar(xs + (i - 0.5) * width, vals, width,
               label=f"{reg}  (naive RMSE {d[reg]['naive_CV_RMSE']:.0f})")
    ax.axhline(100, color="crimson", ls="--", lw=1.5)
    ax.text(len(names) - 0.4, 103, "naive (persistence)", color="crimson",
            ha="right", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("RMSSE  [%]   (lower is better)")
    ax.set_title("Model skill vs naive, by forecast regime\n"
                 "Meola & Weinrich (2025) protocol, rolling-origin 19 folds", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    if "by_lead_time" in d.get("h90", {}):
        by = {}
        for r in d["h90"]["by_lead_time"]:
            by.setdefault(r["model"], {})[r["lead"]] = r["RMSSE_pct"]
        for n in d["h90"]["models"]:
            ys = [by.get(n, {}).get(l, np.nan) for l in LEADS]
            ax.plot(range(len(LEADS)), ys, marker="o", label=n, lw=1.6)
        ax.axhline(100, color="crimson", ls="--", lw=1.5)
        ax.set_xticks(range(len(LEADS)))
        ax.set_xticklabels([f"h {l} d" for l in LEADS])
        ax.set_ylabel("RMSSE  [%]")
        ax.set_title("h90 regime, stratified by lead time\n"
                     "(a mean over the whole block hides this structure)", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(PNG, dpi=140)
    print(f"저장 → {PNG}")


if __name__ == "__main__":
    main()
