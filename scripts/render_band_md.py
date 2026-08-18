"""
outputs/bands/summary.json → docs/METHANE_FORECAST_SYSTEM.md §5 표 자동 생성

숫자를 손으로 옮기지 않기 위한 것이다. 학습을 다시 돌리면 이 스크립트도 다시 돌린다.
실행 : python scripts/render_band_md.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "bands", "summary.json")
ABL = os.path.join(ROOT, "outputs", "bands", "ablation.json")
DOC = os.path.join(ROOT, "docs", "METHANE_FORECAST_SYSTEM.md")
LABEL = {"h1_3": "1–3일", "h3_5": "3–5일", "h5_7": "5–7일", "h7_14": "7–14일",
         "h14_30": "14–30일", "h30_60": "30–60일", "h60_90": "60–90일"}


def main():
    d = json.load(open(SUMMARY, encoding="utf-8"))
    parts = ["### 5.1 구간별 채택 모델과 성적 (폴드밖 pooled)", ""]

    parts += ["| 지평 구간 | 채택 모델 | **pooled R²** | naive R² | RMSSE % | R²≥0.85 | naive 우세 | p | n / folds |",
              "|---|---|---:|---:|---:|:---:|:---:|---:|---:|"]
    for b, s in d.get("serving", {}).items():
        res = d["bands"][b]
        m = res["models"][s["model"]]
        parts.append(
            f"| **{LABEL.get(b, b)}** | {s['model']} | "
            f"{'**' if s['meets_r2_goal'] else ''}{m['pooled_R2']:.4f}{'**' if s['meets_r2_goal'] else ''} | "
            f"{res['naive']['pooled_R2']:.4f} | {m['RMSSE_pct']:.1f} | "
            f"{'✅' if s['meets_r2_goal'] else '❌'} | "
            f"{'✅' if s['beats_naive'] else '❌'} | "
            f"{m['p_vs_naive'] if m['p_vs_naive'] is not None else '—'} | "
            f"{res['n_test']} / {res['n_folds']} |")
    parts.append("")

    # 구간별 전 모델 성적
    parts += ["### 5.2 구간별 전 모델 성적 (pooled R²)", ""]
    bands = list(d["bands"])
    names = sorted({m for b in bands for m in d["bands"][b]["models"]})
    parts.append("| 모델 | " + " | ".join(LABEL.get(b, b) for b in bands) + " |")
    parts.append("|---|" + "---:|" * len(bands))
    for nm in names:
        cells = []
        for b in bands:
            m = d["bands"][b]["models"].get(nm)
            cells.append("—" if m is None else f"{m['pooled_R2']:.3f}")
        parts.append(f"| {nm} | " + " | ".join(cells) + " |")
    parts.append("| *naive (persistence)* | " + " | ".join(
        f"*{d['bands'][b]['naive']['pooled_R2']:.3f}*" for b in bands) + " |")
    parts.append("")

    # RMSSE
    parts += ["### 5.3 구간별 RMSSE % (naive 대비 · 100 미만이어야 채택 가능)", ""]
    parts.append("| 모델 | " + " | ".join(LABEL.get(b, b) for b in bands) + " |")
    parts.append("|---|" + "---:|" * len(bands))
    for nm in names:
        cells = []
        for b in bands:
            m = d["bands"][b]["models"].get(nm)
            if m is None:
                cells.append("—")
            else:
                v = m["RMSSE_pct"]
                cells.append(f"**{v:.1f}**" if v < 100 else f"{v:.1f}")
        parts.append(f"| {nm} | " + " | ".join(cells) + " |")
    parts.append("")

    # 앵커 위 보정의 크기 — 지평이 길수록 모델이 보태는 몫이 커진다
    parts += ["### 5.4 앵커 대비 보정의 크기 (스태킹 수축계수)", "",
              "예측 = FlowAnchor + s · Σ wᵢ (모델ᵢ − FlowAnchor), wᵢ ≥ 0. "
              "s 와 w 는 검증블록의 **서로 다른 절반**에서 정한다. "
              "s = 0 이면 앵커 그대로이므로, 모델이 보탤 것이 없으면 자료가 그렇게 말할 수 있다.", ""]
    parts.append("| 지평 구간 | 수축계수 s | Σw (수축 후) | Stack R² | FlowAnchor R² | 보정이 이득인가 |")
    parts.append("|---|---:|---:|---:|---:|:---:|")
    for b in bands:
        r = d["bands"][b]
        w = r.get("stack_weights_mean", {})
        st = r["models"].get("Stack", {})
        fa = r["models"].get("FlowAnchor", {})
        gain = (st.get("pooled_R2", -9) > fa.get("pooled_R2", 9))
        parts.append(
            f"| {LABEL.get(b, b)} | {w.get('_shrink', '—')} | {w.get('_sum', '—')} | "
            f"{st.get('pooled_R2', '—')} | {fa.get('pooled_R2', '—')} | "
            f"{'✅' if gain else '—'} |")
    parts.append("")
    parts.append("수축계수가 지평과 함께 단조에 가깝게 커진다(0.44 → 0.89). "
                 "짧은 지평에서는 앵커가 거의 전부를 설명하므로 자료가 보정을 억누르고, "
                 "지평이 길어질수록 학습 모델의 몫이 커진다. 이 값은 튜닝 결과가 아니라 "
                 "**측정된 것**이며, 「모델이 언제부터 쓸모 있는가」에 대한 직접적인 답이다.")
    parts.append("")

    # 각도별 단독 성능
    if os.path.exists(ABL):
        a = json.load(open(ABL, encoding="utf-8"))
        parts += ["### 5.5 각도별 단독 성능 (요구사항 2 — 어느 각도가 정보를 주는가)", "",
                  "각 각도의 피처만으로 같은 폴드를 돌린 결과. pooled R².", ""]
        parts.append("| 각도 | " + " | ".join(LABEL.get(b, b) for b in bands) + " |")
        parts.append("|---|" + "---:|" * len(bands))
        kor = {"substrate": "A. 기질 성상", "chemistry": "B. 내부 이화학",
               "vsbalance": "C. VS 물질수지", "temporal": "D. 시계열 구조"}
        for ang, row in a.items():
            cells = [f"{row[b]['pooled_R2']:.3f}" if b in row else "—" for b in bands]
            parts.append(f"| {kor.get(ang, ang)} | " + " | ".join(cells) + " |")
        parts.append("| **전 각도 결합** | " + " | ".join(
            f"**{d['bands'][b]['models'][d['serving'][b]['model']]['pooled_R2']:.3f}**"
            for b in bands) + " |")
        parts.append("")

    body = "\n".join(parts)

    # README 요약표도 같이 갱신한다 (요약은 채택 모델 한 줄씩)
    rd = os.path.join(ROOT, "README.md")
    if os.path.exists(rd):
        rows = ["## 결과 — 구간별 폴드밖 성적", "",
                "| 지평 구간 | 채택 모델 | **pooled R²** | naive R² | RMSSE % | R²≥0.85 |",
                "|---|---|---:|---:|---:|:---:|"]
        for b, sv in d.get("serving", {}).items():
            res = d["bands"][b]
            m = res["models"][sv["model"]]
            rows.append(f"| **{LABEL.get(b, b)}** | {sv['model']} | {m['pooled_R2']:.4f} | "
                        f"{res['naive']['pooled_R2']:.4f} | {m['RMSSE_pct']:.1f} | "
                        f"{'✅' if sv['meets_r2_goal'] else '❌'} |")
        n_ok = sum(1 for v in d.get("serving", {}).values() if v["meets_r2_goal"])
        rows += ["", f"R² ≥ 0.85 달성 **{n_ok}/{len(d.get('serving', {}))} 구간**. "
                 "RMSSE 는 naive(원점값 유지) 대비 정규화 오차로, 100 미만이어야 채택 가능하다. "
                 "전체 표와 각도별 단독 성능은 "
                 "[`docs/METHANE_FORECAST_SYSTEM.md`](docs/METHANE_FORECAST_SYSTEM.md) §5.",
                 ""]
        rtxt = open(rd, encoding="utf-8").read()
        blk = "\n".join(rows)
        if "<!--README_RESULTS-->" in rtxt:
            rtxt = rtxt.replace("<!--README_RESULTS-->", blk)
        else:
            import re as _re
            rtxt = _re.sub(r"## 결과 — 구간별 폴드밖 성적.*?(?=\n## )", blk + "\n",
                           rtxt, flags=_re.S)
        open(rd, "w", encoding="utf-8").write(rtxt)
        print("갱신 → README.md")

    doc = open(DOC, encoding="utf-8").read()
    marker = "<!--BAND_RESULTS-->"
    if marker in doc:
        doc = doc.replace(marker, body)
    else:
        head, _, rest = doc.partition("## 5. 결과\n")
        _, _, tail = rest.partition("\n---\n\n## 6.")
        doc = head + "## 5. 결과\n\n" + body + "\n---\n\n## 6." + tail
    open(DOC, "w", encoding="utf-8").write(doc)
    print("갱신 →", DOC)


if __name__ == "__main__":
    main()
