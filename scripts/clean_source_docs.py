"""
업로드 문서에서 **기각된 내용을 실제로 삭제**한다 (요구사항 1)

원본은 `docs/source/` 에 그대로 보존하고, 기각분을 제거한 판을 `docs/cleaned/` 에 쓴다.
삭제는 손으로 하지 않고 이 스크립트가 한다 — 무엇을 왜 지웠는지가 코드로 남아야
나중에 검증할 수 있기 때문이다. 각 삭제는 `docs/REJECTED_REGISTRY.md` 의 R-번호를 단다.

삭제 자리에는 빈 공간이 아니라 **삭제 표지**를 남긴다. 문서를 읽는 사람이
"여기 무언가 있었고 왜 없어졌는지"를 알 수 있어야 한다.

실행 : python scripts/clean_source_docs.py
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "source")
DST = os.path.join(ROOT, "docs", "cleaned")

SECTION = "section"
LINES = "lines"

# (파일, 종류, 대상, 등록번호, 사유)
RULES: list[tuple[str, str, str, str, str]] = [
    ("PROMPT_통합앙상블모델_구축.md", SECTION,
     "### 2.1 기질은 처음부터 통합되지 않는다. 여액저장조에서 만난다.", "R-01",
     "기질별 전단 커널로 소화조 투입을 재구성하는 설계. 재구성값과 실측 투입의 상관이 "
     "r=0.548 에 그치고, 일요일 반입 22.3 t/d 에도 투입은 196.5 t/d 로 유지된다"
     "(반입 CV 46.1% → 투입 CV 17.3%). 지수 RTD 는 '일정하게 유지되는 출력'을 만들 수 "
     "없다 — 여액저장조는 수동 CSTR 이 아니라 조작되는 완충조다. 구동변수를 실측 투입"
     "(feed_AB_tpd)으로 교체했다."),
    ("PROMPT_통합앙상블모델_구축.md", SECTION,
     "### 5.4 전단 지연은 데이터가 유의하게 식별한다 — 유일한 예외", "R-01",
     "R-01 과 같은 전제(반입 구동) 위에서 얻은 결과다. 구동변수를 실측 투입으로 바꾸면 "
     "전단 커널 자체가 사라지므로 이 비교는 성립하지 않는다."),
    ("PROMPT_통합앙상블모델_구축.md", SECTION,
     "## 1. 데이터", "R-05",
     "존재하지 않는 파일 목록(01_일별_소화조상태.csv ~ 06_모델계수.csv, "
     "영천BGP_소화조_가중치_SRT_데이터셋.xlsx). 정본 자료는 "
     "`data/master_bgp_2018_2023.csv` (2,086일 × 61열) 하나다."),
    ("PROMPT_통합앙상블모델_구축.md", LINES,
     r"k \(음폐수\)", "R-04",
     "k(음폐수)=0.08 /d 는 음폐수:하수슬러지 5:5 혼합조건 값이고 1:9 는 0.32 /d 다."),
    ("PROMPT_통합앙상블모델_구축.md", LINES,
     r"\*\*V 또는 소화액 VS% 오류\*\*", "R-03",
     "유입·유출·소비 VS 가 전부 V 와 무관하다(유입 VS=부하 원자료, 유출 VS=Q_out×소화액 VS%). "
     "V 는 조내 VS 와 SRT 에만 들어가므로 이탈 원인 후보에서 배제된다."),
    ("MODELING_OVERVIEW.md", SECTION,
     "### 4.2 타깃 재정의", "R-09",
     "주 타깃을 Δ(1일 차분)으로 재정의하는 설계. h=1 은 persistence 가 R²=0.932 로 이미 "
     "이기는 구간이고 차분은 ACF(1)=−0.059 로 예측 가능성 자체가 없다. 지평 구간별 "
     "수준 예측(1–3/3–5/5–7/7–14/14–30/30–60/60–90일)으로 교체했다."),
    ("MODELING_OVERVIEW.md", SECTION,
     "### 1.5 VFA/알칼리도 신호등의 함정 (v1 관제 설계 폐기 사유)", "R-07",
     "폐기 사유 자체는 옳지만 이미 반영이 끝나 문서에 남길 필요가 없다. 경보는 시설 자체 "
     "90일 기준선 대비 상대편차로 내고(주의 15%, 위험 25%), 문헌 절대밴드는 경보가 아니라 "
     "'설계·문헌 대비 위치' 진단으로만 병기한다 — `src/bgp/config.py` 의 "
     "RELATIVE_ALARM / GUIDELINE."),
    ("영천BGP_모델링_최종명세_AI인계용.md", LINES,
     r"0\.999", "R-06",
     "연 단위 r=−0.999 (n=4). 반입톤당·투입톤당 어느 정의로도 재현되지 않는다(최대 −0.847)."),
]


def _section_span(text: str, heading: str):
    lines = text.split("\n")
    norm = [unicodedata.normalize("NFC", l).strip() for l in lines]
    target = unicodedata.normalize("NFC", heading).strip()
    try:
        i = norm.index(target)
    except ValueError:
        return None
    depth = len(heading) - len(heading.lstrip("#"))
    for j in range(i + 1, len(lines)):
        s = norm[j]
        if s.startswith("#"):
            if len(s) - len(s.lstrip("#")) <= depth:
                return i, j
    return i, len(lines)


def clean_file(path: str, rules) -> tuple[str, list[dict]]:
    text = open(path, encoding="utf-8").read()
    applied = []
    for _, kind, target, rid, reason in rules:
        if kind == SECTION:
            span = _section_span(text, target)
            if span is None:
                continue
            lines = text.split("\n")
            i, j = span
            stub = [
                f"> **[{rid}] 삭제됨** — `{target.lstrip('# ').strip()}`",
                ">",
                "> " + reason,
                ">",
                "> 원문은 `docs/source/` 에 보존. 판정 근거는 `docs/REJECTED_REGISTRY.md`.",
                "",
            ]
            text = "\n".join(lines[:i] + stub + lines[j:])
            applied.append({"id": rid, "kind": kind, "target": target,
                            "n_lines": j - i, "reason": reason})
        else:
            pat = re.compile(target)
            lines = text.split("\n")
            hit = [l for l in lines if pat.search(l)]
            if not hit:
                continue
            kept, done = [], False
            for l in lines:
                if pat.search(l):
                    if not done:
                        kept.append(f"> **[{rid}] 삭제됨** — {len(hit)}개 줄. {reason}")
                        done = True
                    continue
                kept.append(l)
            text = "\n".join(kept)
            applied.append({"id": rid, "kind": kind, "target": target,
                            "n_lines": len(hit), "reason": reason})
    return text, applied


def main():
    os.makedirs(DST, exist_ok=True)
    by_file: dict[str, list] = {}
    for r in RULES:
        by_file.setdefault(unicodedata.normalize("NFC", r[0]), []).append(r)

    report = {}
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".md"):
            continue
        key = unicodedata.normalize("NFC", fname)
        text, applied = clean_file(os.path.join(SRC, fname), by_file.get(key, []))
        header = (f"<!-- docs/source/{key} 에서 기각분을 삭제한 판. "
                  f"규칙: scripts/clean_source_docs.py · 근거: docs/REJECTED_REGISTRY.md -->\n\n")
        if applied:
            header += ("> **정리 알림** — 이 문서에서 "
                       + ", ".join(sorted({a["id"] for a in applied}))
                       + " 항목이 삭제되었다.\n\n---\n\n")
        open(os.path.join(DST, key), "w", encoding="utf-8").write(header + text)
        report[key] = applied
        tag = ", ".join(sorted({a["id"] for a in applied})) or "(삭제 없음)"
        print(f"{key:45s} → {tag}")

    with open(os.path.join(ROOT, "docs", "cleaning_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    main()
