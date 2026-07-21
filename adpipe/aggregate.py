"""STEP 2 — 분류계급 집계 (PROMPT 1 §STEP 2).

RA 는 합산으로 집계(재정규화만 주의). 미분류는 버리지 않고
`<rank>__unclassified_<상위계급>` 로 보존한다(조성 폐쇄 유지).
"""
import numpy as np
import pandas as pd

RANK_ORDER = ["domain", "phylum", "family", "genus", "species"]


def _label(taxonomy, col, level):
    if col not in taxonomy.index:
        return f"{level[0]}__unclassified"
    v = taxonomy.loc[col, level] if level in taxonomy.columns else None
    if pd.isna(v) or v is None or str(v).startswith("f__unclassified") or v == "unclassified":
        # 상위계급으로 라벨 보강
        higher = None
        for up in RANK_ORDER[:RANK_ORDER.index(level)][::-1]:
            hv = taxonomy.loc[col, up] if up in taxonomy.columns else None
            if pd.notna(hv) and hv not in (None, "unclassified"):
                higher = hv; break
        return f"{level[0]}__unclassified_{higher or 'root'}"
    return str(v)


def aggregate_level(X, taxonomy, level):
    """X(시료×species) -> (X_level, taxonomy_level).

    반환 taxonomy_level: index=집계 taxon, 컬럼=domain..level 상위계급 라벨.
    """
    if level == "species":
        tl = taxonomy.reindex(X.columns).copy()
        return X.copy(), tl

    labels = pd.Index([_label(taxonomy, c, level) for c in X.columns], name=level)
    Xg = X.T.groupby(labels).sum().T
    Xg.index = X.index

    # 집계 taxon 별 상위계급 (첫 구성원 기준, 불일치 시 대표값)
    upto = RANK_ORDER[:RANK_ORDER.index(level) + 1]
    rows = {}
    member_labels = pd.Series(labels.values, index=X.columns)
    for tax in Xg.columns:
        members = member_labels.index[member_labels == tax]
        rec = {}
        for up in upto:
            if up == level:
                rec[up] = tax
            else:
                vals = [taxonomy.loc[m, up] for m in members
                        if m in taxonomy.index and up in taxonomy.columns]
                vals = [v for v in vals if pd.notna(v)]
                rec[up] = pd.Series(vals).mode().iloc[0] if vals else "unclassified"
        rows[tax] = rec
    tl = pd.DataFrame.from_dict(rows, orient="index")[upto]
    tl.index.name = "taxon"
    return Xg, tl


def genus_domain_taxa(domain_taxa):
    """genus 수준 테이블용 domain_taxa: 멤버(genus)가 곧 taxon 이므로 그대로 사용 가능.

    genus 집계 후 taxonomy 의 genus 컬럼 == index 이므로 DomainRatios 매칭이 직접 성립.
    """
    return domain_taxa
