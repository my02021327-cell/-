# 02 · 분류계급 집계 (Aggregation)

RA 합산 집계. 미분류는 `<rank>__unclassified_<상위>` 로 보존(조성 폐쇄 유지).

| 테이블 | p(특징) | p/n | 폐쇄보존 |
|---|---|---|---|
| L_species(ASV대용) | 101 | 2.52 | ✅ |
| L_genus | 96 | 2.40 | ✅ |
| L_family | 69 | 1.73 | ✅ |
| L_phylum | 25 | 0.62 | ✅ |

- n=40, 독립단위(site)=10.
- **권장 기본 = genus(L_genus)**. species(ASV 대용)는 p/n 이 커 검증 부담이 크며, 비교 대상으로만 유지(PROMPT 2). phylum 은 거시 조성 블록.
- p/n>5 이면 경고: 필터링/상위계급 집계로 축소 필요.