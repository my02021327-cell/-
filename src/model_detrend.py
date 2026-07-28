"""이화학을 수준이 아니라 자기 추세 대비 편차로 넣는다.

진단(계수 검사)
  dig_Alk 계수가 폴드별 −9,574 → −3,410 으로 단조 축소하고, 절댓값이 평균
  가스량(10,330)에 맞먹는다. dig_온도는 66 → 4,946 으로 75배 변하고 S_slow 는
  부호가 뒤집힌다. 천천히 표류하는 변수가 **시간 추세의 대리변수**로 적합되고
  있다는 뜻이고, 그런 계수는 다음 구간에서 반드시 틀린다.
  오라클이 동결보다 나빴던 것도 이 때문이다 — 틀린 계수 × 정확한 값.

처방
  수준 대신 **자기 60일 이동중앙값 대비 편차/비율**을 넣어 추세를 흡수하지
  못하게 만든다. 제안된 SURGE(단기/장기 비율)도 같은 원리이므로 함께 검정한다.
실행: `python -m src.model_detrend`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from src.model_both import run
from src.model_robust import prep
from src.splits import cv_folds

WIN = 60
SHORT, LONG = 14, 42


def detrend(g: pd.DataFrame) -> pd.DataFrame:
    d = g.copy()
    for c in ["dig_온도", "dig_VFA", "dig_Alk", "dig_pH", "VS_in", "L_slow"]:
        s = d[c].ffill()
        ref = s.rolling(WIN, min_periods=10).median()
        d[f"{c}_dev"] = (s - ref) / (ref.abs() + 1e-9)          # 상대 편차
    for c in ["VS_in", "feed"]:
        s = d[c].ffill()
        d[f"{c}_surge"] = (s.rolling(SHORT, min_periods=1).mean()
                           / (s.rolling(LONG, min_periods=1).mean() + 1e-9))
    return d


VARIANTS = {
    "유량pool (기준)":        (["S_fast", "S_slow"], []),
    "+이화학 편차":            (["S_fast", "S_slow"], ["dig_온도_dev", "dig_VFA_dev", "dig_Alk_dev"]),
    "+VFA/Alk 비율":         (["S_fast", "S_slow"], ["VFA_ALK"]),
    "+VSin 편차":            (["S_fast", "S_slow"], ["VS_in_dev"]),
    "+VSin SURGE":          (["S_fast", "S_slow"], ["VS_in_surge"]),
    "+투입 SURGE":            (["S_fast", "S_slow"], ["feed_surge"]),
    "+VSin SURGE+이화학 편차":  (["S_fast", "S_slow"],
                             ["VS_in_surge", "dig_온도_dev", "dig_VFA_dev", "dig_Alk_dev"]),
}


def main():
    g = detrend(prep())
    cols = sorted({c for k, h in VARIANTS.values() for c in k + h})
    X = g[cols].ffill().bfill()
    X = (X - X.mean()) / X.std()
    folds = cv_folds()
    rows = []
    for tgt in ["biogas", "methane"]:
        print("=" * 74)
        print(f"타깃: {tgt}")
        print("=" * 74)
        pers = None
        for name, (k, h) in VARIANTS.items():
            r = run(g, X, tgt, k, h, folds)
            if not r:
                print(f"  {name:22s} — 실패"); continue
            pers = pers or {hh: v[1] for hh, v in r.items()}
            print(f"  {name:22s} " + " ".join(f"h{hh}={v[0]:7.4f}" for hh, v in sorted(r.items())))
            for hh, v in r.items():
                rows.append(dict(target=tgt, variant=name, h=hh, R2=v[0], persistence=v[1]))
        print(f"  {'persistence':22s} " + " ".join(f"h{hh}={v:7.4f}" for hh, v in sorted(pers.items())))
        print()
    t = pd.DataFrame(rows)
    t.to_csv("outputs/model_detrend.csv", index=False, encoding="utf-8-sig")
    print("== persistence 대비 이득 ==")
    for tgt in ["biogas", "methane"]:
        s = t[t.target == tgt]
        piv = s.pivot_table(index="variant", columns="h", values="R2")
        per = s.groupby("h")["persistence"].first()
        print(f"\n[{tgt}]")
        print((piv - per).round(3).sort_values(by=7, ascending=False).to_string())
    print("\n[저장] outputs/model_detrend.csv")


if __name__ == "__main__":
    main()
