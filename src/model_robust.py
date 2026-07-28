"""부하 변수를 로버스트 변환해 재검정.

진단
  · 원 VS_in 은 왜도 9.7, 표준화 후 최대 z = 23.2 → 선형 exog(β·x)가 폭주한다.
  · 별개로 충격 시험에서 확인된 부호 편의(과부하 +557 / 저부하 −972 Nm3/d)는
    소화조가 고부하에서 포화한다는 뜻이다. 선형항으로는 표현할 수 없다.
  두 문제 모두 상단을 눌러주는 변환이 해법이다 — 윈저라이즈 + 로그.

실행: `python -m src.model_robust`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score

from src.dataset import build
from src.model_both import HORIZONS, run
from src.splits import DATA_START, cv_folds

TARGETS = ["biogas", "methane"]
WINSOR = (0.01, 0.99)


def robust(s: pd.Series, log: bool = True) -> pd.Series:
    """윈저라이즈 후 로그. 상단 이상치를 자르고 고부하 기울기를 눕힌다."""
    lo, hi = s.quantile(WINSOR[0]), s.quantile(WINSOR[1])
    w = s.clip(lo, hi)
    return np.log1p(w) if log else w


def prep():
    g = build().loc[DATA_START:]
    g["VS_w"] = robust(g["VS_in"], log=False)
    g["VS_log"] = robust(g["VS_in"])
    g["Lslow_w"] = robust(g["L_slow"], log=False)
    g["Lslow_log"] = robust(g["L_slow"])
    g["Lfast_log"] = robust(g["L_fast"])
    g["Sslow_log"] = robust(g["S_slow"])
    g["Sfast_log"] = robust(g["S_fast"])
    return g


VARIANTS = {
    "유량pool (기준)":            (["S_fast", "S_slow"], []),
    "유량pool+VSin 윈저":         (["S_fast", "S_slow"], ["VS_w"]),
    "유량pool+VSin 로그":         (["S_fast", "S_slow"], ["VS_log"]),
    "유량pool+부하slow 로그":      (["S_fast", "S_slow"], ["Lslow_log"]),
    "유량pool+부하pool 로그":      (["S_fast", "S_slow"], ["Lfast_log", "Lslow_log"]),
    "부하pool 로그 단독":          ([], ["Lfast_log", "Lslow_log"]),
    "유량pool 로그":              (["Sfast_log", "Sslow_log"], []),
}


def main():
    g = prep()
    cols = sorted({c for k, h in VARIANTS.values() for c in k + h})
    X = g[cols].ffill().bfill()
    X = (X - X.mean()) / X.std()
    print("변환 후 표준화 최대 z:")
    for c in cols:
        print(f"  {c:12s} max z = {X[c].max():5.2f}  min z = {X[c].min():6.2f}")
    folds = cv_folds()
    print(f"\n폴드 {len(folds)}개 | 2023 봉인 홀드아웃 미개봉\n")
    rows = []
    for tgt in TARGETS:
        print("=" * 76)
        print(f"타깃: {tgt}")
        print("=" * 76)
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
    t.to_csv("outputs/model_robust.csv", index=False, encoding="utf-8-sig")
    print("== persistence 대비 이득 ==")
    for tgt in TARGETS:
        s = t[t.target == tgt]
        piv = s.pivot_table(index="variant", columns="h", values="R2")
        per = s.groupby("h")["persistence"].first()
        print(f"\n[{tgt}]")
        print((piv - per).round(3).sort_values(by=7, ascending=False).to_string())
    print("\n[저장] outputs/model_robust.csv")


if __name__ == "__main__":
    main()
