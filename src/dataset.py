"""원자료 → 모델링 데이터셋 재구축 (A/B 건강계열 선택 포함).

기존 master.xlsx 는 소화조 #A·#B 를 **평균**했다(온도 97%, Alk 100% 일치 확인).
그 결과 #A 온도계 고장(2022 Q4~)과 #B 알칼리도 고장(2023 Q1~)이 그대로 섞여
"2022 Q4 공정 단절"처럼 보였다. 실제로는 각각 쌍둥이 계열이 멀쩡했다.

건강계열 선택 규칙
  두 계열 상대차 중앙값이 pH 0.1 %, Alk 0.8 %, 메탄함량 0.2 % 로 거의 일치한다.
  따라서 (1) 10 % 이내로 일치하면 평균을 쓰고,
        (2) 벌어지면 '일치하던 날들의 60일 이동중앙값'에 가까운 쪽을 택한다.
  한쪽 계열이 서서히 표류해도 자기 이력이 아니라 **쌍둥이 기준**으로 잡힌다.

타깃 둘을 모두 만든다
  · biogas  : 매일 계측 (2,084일)
  · methane : biogas × CH4% (1,257일)
실행: `python -m src.dataset`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.raw_extract as rx
from src.bio_lag import substrate_pool

OUT = "data/dataset.csv"
AGREE = 0.10          # 상대차 10 % 이내면 일치로 본다
REF_WIN = 60          # 기준 이동중앙값 창(일)
TAU_FAST, TAU_SLOW = 1, 8
V_DIGESTER = 8000.0


def _col(d, *t):
    c = rx.pick(d, *t)
    return d[c[0]].replace(0, np.nan) if c else None


def healthy(a: pd.Series, b: pd.Series) -> pd.Series:
    """A/B 중 건강한 값을 고른다. 일치하면 평균, 벌어지면 기준에 가까운 쪽."""
    a, b = a.astype(float), b.astype(float)
    mean = pd.concat([a, b], axis=1).mean(axis=1)
    rel = (a - b).abs() / mean.abs()
    agree = rel <= AGREE
    ref = mean.where(agree).rolling(REF_WIN, min_periods=5).median().ffill()
    pick_a = (a - ref).abs() <= (b - ref).abs()
    out = mean.where(agree)
    out = out.fillna(a.where(pick_a)).fillna(b.where(~pick_a))
    return out.fillna(a).fillna(b)


def build() -> pd.DataFrame:
    d = rx.load_all()
    g = pd.DataFrame(index=d.index)

    # ── 소화조 : A/B 건강계열 ────────────────────────────────────────────
    pairs = {"pH": "pH", "온도": "온도(하)", "TS": "TS(%)", "VS": "VS(%)",
             "VFA": "VFA", "Alk": "Alk", "CODcr": "COD(cr)", "NH3N": "NH3-N",
             "CH4pct": "메탄함량"}
    for name, item in pairs.items():
        a, b = _col(d, "혐기성소화조", "#A", item), _col(d, "혐기성소화조", "#B", item)
        if a is None or b is None:
            continue
        g[f"dig_{name}"] = healthy(a, b)

    # ── 가스·투입 : 두 계열 합계 ────────────────────────────────────────
    ga, gb = _col(d, "혐기성소화조", "#A", "바이오가스발생량"), _col(d, "혐기성소화조", "#B", "바이오가스발생량")
    g["biogas"] = ga.fillna(0) + gb.fillna(0)
    g.loc[g["biogas"] == 0, "biogas"] = np.nan
    qa, qb = _col(d, "혐기성소화조", "#A", "투입량"), _col(d, "혐기성소화조", "#B", "투입량")
    g["feed"] = (qa.fillna(0) + qb.fillna(0)).replace(0, np.nan)

    # ── 유기산화조(소화조 실제 투입 성상) ────────────────────────────────
    for name, item in [("pH", "pH"), ("TS", "TS(%)"), ("VS", "VS(%)"), ("CODcr", "COD(cr)")]:
        s = _col(d, "유기산화조", item)
        if s is not None:
            g[f"in_{name}"] = s

    # ── 반입 스트림(계획 가능한 제어입력) ────────────────────────────────
    for name in ["음폐수", "가축분뇨", "음식물"]:
        s = _col(d, "반입", name)
        if s is not None:
            g[f"str_{name}"] = s

    # ── 타깃 ────────────────────────────────────────────────────────────
    ch4 = g["dig_CH4pct"].where(g["dig_CH4pct"].between(45, 80))
    g["CH4pct"] = ch4
    g["methane"] = g["biogas"] * ch4 / 100

    # ── 부하·지연 ───────────────────────────────────────────────────────
    g["VS_in"] = g["feed"] * g["in_VS"] / 100 * 1000          # kg VS/d
    g["COD_in"] = g["feed"] * g["in_CODcr"] / 1000            # kg COD/d
    g["OLR"] = g["VS_in"] / V_DIGESTER
    g["S_fast"] = substrate_pool(g["feed"].ffill(), TAU_FAST)
    g["S_slow"] = substrate_pool(g["feed"].ffill(), TAU_SLOW)
    # 농도 반영 부하 pool (충격은 유량이 아니라 농도에서 온다)
    g["L_fast"] = substrate_pool(g["VS_in"].ffill(), TAU_FAST)
    g["L_slow"] = substrate_pool(g["VS_in"].ffill(), TAU_SLOW)
    g["VFA_ALK"] = g["dig_VFA"] / g["dig_Alk"]
    return g


def main():
    g = build()
    print(f"데이터셋 {g.shape[0]}일 × {g.shape[1]}열  ({g.index.min().date()} ~ {g.index.max().date()})")
    print("\n유효 관측일(연도별)")
    key = ["biogas", "methane", "CH4pct", "feed", "VS_in", "dig_pH", "dig_온도",
           "dig_VFA", "dig_Alk", "in_VS"]
    print(g[key].notna().groupby(g.index.year).sum().to_string())
    print("\n건강계열 선택 효과 — 분기 중앙값")
    q = g[["dig_온도", "dig_Alk", "dig_pH"]].resample("QE").median()
    print(q.loc["2022-01":"2023-09"].round(1).to_string())
    g.to_csv(OUT, encoding="utf-8-sig")
    print(f"\n[저장] {OUT}")
    return g


if __name__ == "__main__":
    main()
