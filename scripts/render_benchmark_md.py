"""
outputs/benchmark_meola2025.json → 마크다운 표.

docs/PAPER_Meola2025_대조분석.md 의 <!--RESULTS--> 자리에 결과를 끼워 넣는다.
숫자를 손으로 옮기지 않기 위한 것이므로, 벤치마크를 다시 돌리면 이 스크립트도 다시 돌린다.

실행 : python scripts/render_benchmark_md.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(ROOT, "outputs", "benchmark_meola2025.json")
DOC = os.path.join(ROOT, "docs", "PAPER_Meola2025_대조분석.md")

TITLE = {
    "od1": ("7.1 체제 A — OD = 1일 (논문과 직접 비교 가능)",
            "직전 관측 메탄은 쓰되 소화조·유입 이화학은 전일값으로 민다. "
            "naive = 직전 관측일의 메탄."),
    "h90": ("7.2 체제 B — H = 90일 블록 (프로젝트 기준선 체제)",
            "원점 이후 타깃 되먹임과 내부 이화학을 원점값으로 동결한다(PROMPT §8-5). "
            "naive = 원점값 고정."),
}


def model_table(res: dict) -> str:
    rows = ["| 모델 | CV-RMSE (㎥/d) | **RMSSE %** | val RMSSE % | val→test 격차 (pp) | R² | naive 이긴 폴드 | p (vs naive) |",
            "|---|---:|---:|---:|---:|---:|:---:|---:|"]
    for name, s in res["models"].items():
        p = s["vs_naive"]["p"]
        mark = "**" if s["RMSSE_pct"] < 100 else ""
        rows.append(
            f"| {name} | {s['CV_RMSE']:.1f} | {mark}{s['RMSSE_pct']:.1f}{mark} | "
            f"{s['val_RMSSE_pct']:.1f} | {s['val_test_gap_pp']:+.1f} | {s['R2_mean']:.3f} | "
            f"{s['beats_naive_folds']}/{res['n_folds']} | "
            f"{('%.4f' % p) if p is not None else '—'} |")
    return "\n".join(rows)


def importance_table(res: dict, k: int = 10) -> str:
    pi = res["permutation_importance_best"]
    rows = [f"최상위 모델 **{pi['model']}** · 마지막 폴드 시험구간 순열중요도 (ΔRMSE ㎥/d, 클수록 중요)",
            "", "| 순위 | 피처 | ΔRMSE | ±sd |", "|:---:|---|---:|---:|"]
    for i, d in enumerate(pi["top"][:k], 1):
        rows.append(f"| {i} | `{d['feature']}` | {d['delta_RMSE']:+.1f} | {d['sd']:.1f} |")
    return "\n".join(rows)


def lead_table(res: dict) -> str:
    if "by_lead_time" not in res:
        return ""
    leads = ["1-3", "4-7", "8-14", "15-30", "31-90"]
    by = {}
    for r in res["by_lead_time"]:
        by.setdefault(r["model"], {})[r["lead"]] = r["RMSSE_pct"]
    order = [m for m in res["models"]]
    rows = ["| 모델 | " + " | ".join(f"h {l}일" for l in leads) + " |",
            "|---|" + "---:|" * len(leads)]
    for m in order:
        cells = []
        for l in leads:
            v = by.get(m, {}).get(l)
            cells.append("—" if v is None else (f"**{v:.1f}**" if v < 100 else f"{v:.1f}"))
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    parts = [
        "> `python -m src.benchmark` 출력. 원자료 `outputs/benchmark_meola2025.json`.",
        f"> rolling-origin — 초기 학습 {d['protocol']['initial_train_days']}일 · "
        f"지평 {d['protocol']['horizon_days']}일 · 원점 {d['protocol']['step_days']}일 전진 · "
        f"모델 선택은 각 폴드 학습셋 내부(fit→val)에서만.",
        f"> V = {d['V_digester_m3']:.0f} ㎥ (외생 입력, `PROMPT §2.3`).",
        "",
        "**RMSSE < 100 % 만 채택 가능**하다. 100 % 는 naive(persistence)와 동률이다.",
        "",
    ]
    for reg in ("od1", "h90"):
        if reg not in d:
            continue
        res = d[reg]
        title, note = TITLE[reg]
        parts += [f"### {title}", "", note,
                  f"**{res['n_folds']}폴드 · naive CV-RMSE = {res['naive_CV_RMSE']} ㎥/d**", "",
                  model_table(res), "", importance_table(res), ""]
        lt = lead_table(res)
        if lt:
            parts += ["#### 리드타임 층화 (RMSSE %) — 평균은 지평 구조를 숨긴다",
                      "", lt, ""]
    body = "\n".join(parts)

    doc = open(DOC, encoding="utf-8").read()
    marker = "<!--RESULTS-->"
    if marker in doc:
        doc = doc.replace(marker, body)
    else:  # 이미 채워진 경우 7장 본문만 교체
        head, _, rest = doc.partition("## 7. 벤치마크 실측 결과\n")
        _, _, tail = rest.partition("\n---\n\n## 8.")
        doc = head + "## 7. 벤치마크 실측 결과\n\n" + body + "\n---\n\n## 8." + tail
    open(DOC, "w", encoding="utf-8").write(doc)
    print(f"갱신 → {DOC}")


if __name__ == "__main__":
    main()
