"""이화학 충격부하 탐지 — 유량이 아니라 농도·성상에서 일어난 충격.

유량(반입·투입)은 6년간 평상시의 1.3배를 넘은 적이 없다. 그러나 소화조에
들어가는 것은 유량이 아니라 유량×농도이고, 농도 쪽에는 큰 충격이 있다.
원자료(#A/#B 분리, 유기산화조 계측)로 사건을 찾고 소화조 응답을 본다.
실행: `python -m src.chem_shock`
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import src.raw_extract as rx

Z_HI = 4.0            # 로버스트 z (MAD 기준) 임계


def col(d, *t):
    c = rx.pick(d, *t)
    return d[c[0]] if c else None


def robust_z(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan)
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-9)


def main():
    d = rx.load_all()
    print(f"원자료 {d.shape[0]}일 × {d.shape[1]}열\n")

    targets = {
        "유기산화조 VS(%)":   ("유기산화조", "VS"),
        "유기산화조 TS(%)":   ("유기산화조", "TS"),
        "유기산화조 COD":     ("유기산화조", "COD"),
        "유기산화조 pH":      ("유기산화조", "pH"),
        "#A VFA":           ("혐기성소화조", "#A", "VFA"),
        "#B VFA":           ("혐기성소화조", "#B", "VFA"),
        "#A Alk":           ("혐기성소화조", "#A", "Alk"),
        "#B Alk":           ("혐기성소화조", "#B", "Alk"),
        "#A pH":            ("혐기성소화조", "#A", "pH"),
        "#B pH":            ("혐기성소화조", "#B", "pH"),
        "#A 메탄함량":        ("혐기성소화조", "#A", "메탄함량"),
        "#B 메탄함량":        ("혐기성소화조", "#B", "메탄함량"),
    }
    print(f"{'변수':16s} {'n':>5s} {'중앙값':>10s} {'최대':>11s} {'최대/중앙':>9s} "
          f"{'|z|>4 일수':>9s}  최대 발생일")
    events = {}
    for name, terms in targets.items():
        s = col(d, *terms)
        if s is None:
            print(f"{name:16s} — 컬럼 없음"); continue
        s = s.replace(0, np.nan)
        z = robust_z(s)
        hit = z.abs() > Z_HI
        events[name] = d.index[hit.fillna(False)]
        print(f"{name:16s} {s.notna().sum():5d} {s.median():10.1f} {s.max():11.1f} "
              f"{s.max()/s.median():9.1f} {int(hit.sum()):9d}  {s.idxmax().date()}")

    # ── 충격 사건 목록 (유기산화조 VS 기준) ──
    print("\n" + "=" * 76)
    print("유기산화조 VS 충격 상위 12건 — 소화조에 실제로 들어간 유기물 농도")
    print("=" * 76)
    vs = col(d, "유기산화조", "VS").replace(0, np.nan)
    q = col(d, "혐기성소화조", "#A", "투입량")
    qb = col(d, "혐기성소화조", "#B", "투입량")
    load = (q.fillna(0) + qb.fillna(0)) * vs / 100 * 1000        # kg VS/d
    top = vs.nlargest(12)
    ga = col(d, "혐기성소화조", "#A", "바이오가스발생량")
    gb = col(d, "혐기성소화조", "#B", "바이오가스발생량")
    gas = ga.fillna(0) + gb.fillna(0)
    base = gas.rolling(30, min_periods=10).median()
    print(f"{'날짜':12s} {'VS%':>7s} {'중앙대비':>7s} {'VS부하kg':>10s} "
          f"{'가스Nm3':>9s} {'평시대비':>8s} {'#A pH':>6s} {'#B pH':>6s}")
    pa, pb = col(d, "혐기성소화조", "#A", "pH"), col(d, "혐기성소화조", "#B", "pH")
    for t, v in top.items():
        g = gas.get(t, np.nan); b = base.get(t, np.nan)
        print(f"{str(t.date()):12s} {v:7.2f} {v/vs.median():7.1f}x {load.get(t, np.nan):10.0f} "
              f"{g:9.0f} {g/b*100 if b else np.nan:7.0f}% "
              f"{pa.get(t, np.nan):6.2f} {pb.get(t, np.nan):6.2f}")

    # ── A/B 불일치 : 한 조만 이상하면 계측·공정 이상 판별 근거 ──
    print("\n" + "=" * 76)
    print("A/B 계열 불일치 — 병렬 조라 정상이면 서로 비슷해야 한다")
    print("=" * 76)
    for tag, item in [("pH", "pH"), ("VFA", "VFA"), ("Alk", "Alk"),
                      ("메탄함량", "메탄함량"), ("바이오가스발생량", "바이오가스발생량")]:
        a, b = col(d, "혐기성소화조", "#A", item), col(d, "혐기성소화조", "#B", item)
        if a is None or b is None:
            continue
        a, b = a.replace(0, np.nan), b.replace(0, np.nan)
        rel = (a - b).abs() / ((a + b) / 2)
        both = a.notna() & b.notna()
        print(f"  {tag:14s} n={both.sum():4d}  상대차 중앙 {rel[both].median()*100:5.1f}% "
              f"  >20% 인 날 {int((rel[both] > 0.2).sum()):4d}일  최대 {rel[both].max()*100:7.0f}%")


if __name__ == "__main__":
    main()
