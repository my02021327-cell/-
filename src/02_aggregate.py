"""STEP 2 runner — 3개 분류계급 병렬 집계 + p/n 보고."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from adpipe.config import load_raw
from adpipe.aggregate import aggregate_level

d = load_raw()
X, taxonomy = d["X"], d["taxonomy"]
os.makedirs("data/processed", exist_ok=True)
n = X.shape[0]; nsite = d["meta"]["site"].nunique()

rows = []
levels = {"species": "L_species(ASV대용)", "genus": "L_genus",
          "family": "L_family", "phylum": "L_phylum"}
for level, name in levels.items():
    Xl, tl = aggregate_level(X, taxonomy, level)
    # 조성 폐쇄 확인(집계 후 행합 보존)
    close_ok = np.allclose(Xl.sum(axis=1).values, X.sum(axis=1).values, atol=1e-6)
    Xl.round(4).to_csv(f"data/processed/{name}.csv")
    tl.to_csv(f"data/processed/{name}_taxonomy.csv")
    p = Xl.shape[1]
    rows.append((name, p, f"{p/n:.2f}", "✅" if close_ok else "❌"))

rep = ["# 02 · 분류계급 집계 (Aggregation)", "",
       "RA 합산 집계. 미분류는 `<rank>__unclassified_<상위>` 로 보존(조성 폐쇄 유지).", "",
       "| 테이블 | p(특징) | p/n | 폐쇄보존 |", "|---|---|---|---|"]
for name, p, pn, ok in rows:
    rep.append(f"| {name} | {p} | {pn} | {ok} |")
rep += ["",
        f"- n={n}, 독립단위(site)={nsite}.",
        "- **권장 기본 = genus(L_genus)**. species(ASV 대용)는 p/n 이 커 검증 부담이 크며, "
        "비교 대상으로만 유지(PROMPT 2). phylum 은 거시 조성 블록.",
        "- p/n>5 이면 경고: 필터링/상위계급 집계로 축소 필요."]
open("reports/02b_aggregation.md", "w").write("\n".join(rep))
print("\n".join(rep[4:]))
print("\n저장: data/processed/L_*.csv (+ _taxonomy.csv)")
