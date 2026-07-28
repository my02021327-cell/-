"""원자료(연도별 BGP 운영일지 xlsx) → 이름 기반 통합 추출.

연도마다 컬럼 위치가 다르므로(2018·19·20 vs 21·22) 위치가 아니라
3단 계층 헤더(그룹|계열|항목)를 이름으로 결합해 파싱한다.

master.xlsx 에 없던 변수를 살린다
  · 개별 VFA 6종(아세트산·프로피온산·부틸산·이소부틸산·발레르산·이소발레르산)
    → 프로피온산은 분해가 가장 느려 스트레스 시 먼저 축적되는 조기경보 지표
  · 온도 3점(상·중·하) → 성층화·교반 진단
  · 소화조 #A / #B 분리 → 한 조가 이상할 때 다른 조가 대조군이 된다
실행: `python -m src.raw_extract`
"""
from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_GLOB = "/root/.claude/uploads/*/[0-9a-f]*-1[89]*.xlsx"
UPLOAD_DIR = "/root/.claude/uploads"
OUT = "data/raw_merged.csv"
HDR_ROWS = (1, 2, 3)      # 그룹 / 계열 / 항목
DATA_ROW = 5


def _clean(x) -> str:
    s = "" if pd.isna(x) else str(x)
    return re.sub(r"\s+", " ", s).strip()


def read_year(path: str) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Data", header=None)
    g = d.iloc[HDR_ROWS[0]].ffill().map(_clean)
    s = d.iloc[HDR_ROWS[1]].ffill().map(_clean)
    f = d.iloc[HDR_ROWS[2]].map(_clean)
    names = []
    for j in range(d.shape[1]):
        parts = [p for p in (g[j], s[j], f[j]) if p and p != "nan"]
        names.append("|".join(parts) if parts else f"_c{j}")
    body = d.iloc[DATA_ROW:].copy()
    body.columns = names
    date_col = names[0]
    body[date_col] = pd.to_datetime(body[date_col], errors="coerce")
    body = body.dropna(subset=[date_col]).rename(columns={date_col: "date"})
    body = body.set_index("date").apply(pd.to_numeric, errors="coerce")
    return body.loc[:, ~body.columns.duplicated()]


def load_all(paths: list[str] | None = None) -> pd.DataFrame:
    paths = paths or sorted(glob.glob(os.path.join(UPLOAD_DIR, "*", "*.xlsx")))
    frames = []
    for p in paths:
        try:
            frames.append(read_year(p))
        except Exception as e:                       # noqa: BLE001
            print(f"  ! {os.path.basename(p)[:24]} 실패: {e}")
    d = pd.concat(frames).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    # 전 구간 결측이거나 상수(0 포함)인 열 제거
    keep = [c for c in d.columns
            if d[c].replace(0, np.nan).notna().sum() > 20]
    return d[keep]


def pick(d: pd.DataFrame, *terms: str) -> list[str]:
    """헤더 이름에 모든 term 이 들어간 컬럼."""
    return [c for c in d.columns if all(t in c for t in terms)]


def main():
    d = load_all()
    print(f"통합 {d.shape[0]}일 × {d.shape[1]}열  ({d.index.min().date()} ~ {d.index.max().date()})")
    print(f"연도별 행수: {d.groupby(d.index.year).size().to_dict()}\n")
    for tag, terms in [("소화조 #A", ("혐기성소화조", "#A")), ("소화조 #B", ("혐기성소화조", "#B")),
                       ("유기산화조", ("유기산화조",)), ("반입", ("반입",))]:
        cs = pick(d, *terms)
        print(f"[{tag}] {len(cs)}열")
        for c in cs:
            s = d[c].replace(0, np.nan)
            print(f"   {c.split('|')[-1]:14s} n={s.notna().sum():4d} "
                  f"평균={s.mean():10.2f} 범위 {s.min():8.2f}~{s.max():10.2f}")
        print()
    os.makedirs("data", exist_ok=True)
    d.to_csv(OUT, encoding="utf-8-sig")
    print(f"[저장] {OUT}")
    return d


if __name__ == "__main__":
    main()
