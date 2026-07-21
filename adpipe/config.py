"""CONFIG 및 원시 데이터 로딩 (PROMPT 1 §0)."""
import os
import numpy as np
import pandas as pd
import yaml

# --- CONFIG (실행 전 채우는 부분 — 합성 데모용으로 확정) ----------------------
CONFIG = {
    "INPUT": {
        "ra_table":     "data/raw/ra_table.csv",
        "metadata":     "data/raw/metadata.csv",
        "taxonomy":     "data/raw/taxonomy.csv",
        "counts_table": None,          # 없음 → CoDA 경로
        "domain_taxa":  "config/domain_taxa.yaml",
    },
    "DESIGN": {
        "n_samples": 40, "n_sites": 10, "n_seasons": 4,
        "structure": "repeated_measures",
        "group_key": "site",           # CV 분할 단위. 절대 변경 금지
        "ra_scale":  "auto",
        "domain":    "both",           # 세균 + 고균(메탄생성균) 모두 존재
    },
    "TARGET": {
        "name": "methane",             # 실측(targets.xlsx 2022) · 지역 공통 가정 → 계절 결정
        "task": "regression",
    },
    "SEASON_ORDER": ["spring", "summer", "autumn", "winter"],
}


def _find_root(start=None):
    """repo 루트(data/raw 존재하는 곳) 탐색."""
    d = os.path.abspath(start or os.getcwd())
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "data", "raw")):
            return d
        d = os.path.dirname(d)
    return os.path.abspath(os.getcwd())


def load_raw(root=None):
    """ra_table / metadata / taxonomy / domain_taxa 로드.

    반환: dict(X=DataFrame[sample×taxa], meta=DataFrame, taxonomy=DataFrame,
              domain_taxa=dict, y=Series, groups=Series, season=Series)
    """
    root = root or _find_root()
    c = CONFIG["INPUT"]
    X = pd.read_csv(os.path.join(root, c["ra_table"]), index_col=0)
    meta = pd.read_csv(os.path.join(root, c["metadata"]), index_col=0)
    taxonomy = pd.read_csv(os.path.join(root, c["taxonomy"]), index_col=0)
    with open(os.path.join(root, c["domain_taxa"])) as f:
        domain_taxa = yaml.safe_load(f)

    # 시료 정렬 일치
    meta = meta.loc[X.index]
    tname = CONFIG["TARGET"]["name"]
    y = meta[tname].astype(float) if tname in meta.columns else None
    groups = meta[CONFIG["DESIGN"]["group_key"]]
    season = meta["season"]
    return dict(X=X, meta=meta, taxonomy=taxonomy, domain_taxa=domain_taxa,
                y=y, groups=groups, season=season, root=root)


def get_modeling_data(level="genus", root=None):
    """모델링용 데이터: 지정 분류계급으로 집계한 X + 정렬된 참조 테이블."""
    from .aggregate import aggregate_level
    d = load_raw(root)
    Xl, tl = aggregate_level(d["X"], d["taxonomy"], level)
    out = dict(d)
    out["X"] = Xl
    out["taxonomy"] = tl
    out["level"] = level
    return out


def detect_ra_scale(X):
    """행 합 기준 RA 스케일 판별: 'unit'(≈1) / 'percent'(≈100) / 'unknown'."""
    rs = X.sum(axis=1)
    if np.allclose(rs, 1.0, atol=0.01):
        return "unit"
    if np.allclose(rs, 100.0, atol=1.0):
        return "percent"
    med = float(np.median(rs))
    if abs(med - 1.0) < 0.05:
        return "unit"
    if abs(med - 100.0) < 5:
        return "percent"
    return "unknown"


def to_proportion(X):
    """어떤 스케일이든 행합=1 비율로 정규화(조성 폐쇄)."""
    Xv = np.asarray(X, dtype=float)
    rs = Xv.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = Xv / rs
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(P, index=X.index, columns=X.columns)
    return P
