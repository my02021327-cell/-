"""2018~2023 원자료 전수 감사 — 모델링 이전에 데이터 신뢰 구간을 확정한다.

검사 항목
  ① 결측 구조(연도별·요일별)      ② 물리적 불가능 값
  ③ 변수 간 정합성 위반            ④ 고정값(센서 정지) 런
  ⑤ 분기 평균 급변점(체제 단절)    ⑥ 물질수지(메탄 수율)
실행: `python -m src.data_audit`
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from src.final_ensemble import prepare, TARGET

pd.set_option("display.width", 200)

CHEM = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_TS", "소화조_VFA",
        "소화조_TAlk", "소화조_MAlk", "소화조_CODcr", "소화조_NH4N", "소화조_TN"]
IN = ["유입_pH", "유입_TS", "유입_VS", "유입_CODcr", "유입_TN"]
FEED = ["투입량합계", "feed_음폐수", "feed_가축분뇨", "feed_음식물"]

# 문헌·물리 기반 가능 범위 (하한, 상한)
PLAUS = {
    "소화조_pH": (6.0, 9.0), "유입_pH": (3.0, 9.0), "소화조_온도": (15.0, 60.0),
    "소화조_VS": (0.1, 10.0), "소화조_TS": (0.2, 15.0),
    "소화조_VFA": (50, 15000), "소화조_TAlk": (2000, 40000), "소화조_MAlk": (1000, 40000),
    "소화조_CODcr": (5000, 90000), "소화조_NH4N": (200, 8000), "소화조_TN": (200, 12000),
    "유입_TS": (0.5, 30.0), "유입_VS": (0.3, 30.0), "유입_CODcr": (5000, 250000),
    TARGET: (500, 20000),
}


def sec(t): print(f"\n{'='*78}\n{t}\n{'='*78}")


def main():
    d = prepare()
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    print(f"감사 대상 {len(d)}일  ({d.index.min().date()} ~ {d.index.max().date()})")

    # ── ① 결측 구조 ──────────────────────────────────────────────
    sec("① 결측률 (%) — 연도별")
    cols = [TARGET] + FEED[:1] + CHEM + IN
    miss = d.groupby(d.index.year)[cols].apply(lambda g: g.isna().mean() * 100).round(1)
    print(miss.T.to_string())

    sec("① -b 메탄 계측 요일 구조")
    wd = d.groupby(d.index.dayofweek)[TARGET].apply(lambda s: (1 - s.isna().mean()) * 100)
    print("  " + "  ".join(f"{n}={v:.0f}%" for n, v in zip("월화수목금토일", wd)))

    # ── ② 물리적 불가능 값 ───────────────────────────────────────
    sec("② 물리적 가능범위 이탈")
    any_bad = False
    for c, (lo, hi) in PLAUS.items():
        if c not in d: continue
        s = d[c].astype(float)
        bad = s.notna() & ((s < lo) | (s > hi))
        if bad.sum():
            any_bad = True
            ex = s[bad]
            yrs = ex.groupby(ex.index.year).size().to_dict()
            print(f"  {c:12s} 범위[{lo:g},{hi:g}] 이탈 {bad.sum():4d}건  "
                  f"관측 {ex.min():.4g}~{ex.max():.4g}  연도별{yrs}")
    if not any_bad: print("  없음")

    # ── ③ 변수 간 정합성 ─────────────────────────────────────────
    sec("③ 변수 간 정합성 위반")
    chk = [("소화조_VS > 소화조_TS", d["소화조_VS"] > d["소화조_TS"]),
           ("유입_VS > 유입_TS", d["유입_VS"] > d["유입_TS"]),
           ("소화조_MAlk > 소화조_TAlk", d["소화조_MAlk"] > d["소화조_TAlk"]),
           ("소화조_VFA == 소화조_TAlk (동일값)", d["소화조_VFA"] == d["소화조_TAlk"]),
           ("소화조_TAlk == 소화조_MAlk (동일값)", d["소화조_TAlk"] == d["소화조_MAlk"]),
           ("투입량합계 == 0", d["투입량합계"] == 0)]
    for name, m in chk:
        m = m.fillna(False)
        if m.sum():
            idx = d.index[m]
            print(f"  {name:34s} {m.sum():4d}건  {idx.min().date()} ~ {idx.max().date()}"
                  f"  연도별{idx.to_series().groupby(idx.year).size().to_dict()}")
        else:
            print(f"  {name:34s}    0건")

    # ── ④ 고정값 런 (센서 정지 / 전기 복사) ──────────────────────
    sec("④ 동일값 연속 반복 (최장 런)")
    for c in [TARGET] + CHEM[:8] + FEED[:1]:
        s = d[c].dropna()
        if len(s) < 20: continue
        grp = (s != s.shift()).cumsum()
        run = s.groupby(grp).size()
        top = run.max()
        if top >= 5:
            g = run.idxmax(); seg = s[grp == g]
            print(f"  {c:12s} 최장 {top:3d}회 연속  값={seg.iloc[0]:.4g}  "
                  f"{seg.index.min().date()} ~ {seg.index.max().date()}")

    # ── ⑤ 분기 평균 급변점 ───────────────────────────────────────
    sec("⑤ 체제 단절 — 분기평균이 직전 4분기 대비 몇 σ 벗어났나 (|z|>3)")
    q = d.resample("QE")[CHEM[:8] + [TARGET, "투입량합계"]].mean()
    for c in q.columns:
        s = q[c].dropna()
        if len(s) < 8: continue
        base_m, base_s = s.rolling(4).mean().shift(1), s.rolling(4).std().shift(1)
        z = (s - base_m) / base_s
        hit = z[z.abs() > 3]
        for t, v in hit.items():
            print(f"  {c:12s} {t.date()}  z={v:+6.1f}  값 {s[t]:.4g} "
                  f"(직전4분기 평균 {base_m[t]:.4g})")

    # ── ⑥ 물질수지 ───────────────────────────────────────────────
    sec("⑥ 물질수지 — 연도별 메탄수율")
    d2 = d.copy()
    d2["VS_load"] = d2["투입량합계"] * d2["유입_VS"] / 100 * 1000     # kg VS/d
    d2["yield"] = d2[TARGET] / d2["VS_load"]
    t = d2.groupby(d2.index.year).agg(
        투입량=("투입량합계", "mean"), 유입VS=("유입_VS", "mean"),
        VS부하=("VS_load", "mean"), 메탄=(TARGET, "mean"), 수율=("yield", "mean")).round(2)
    print(t.to_string())
    print("\n  * 문헌 통상 수율 0.30~0.55 Nm3 CH4/kg VS")

    # ── ⑦ 타깃 정의 검증 : 'methane' 은 CH4 인가 소화가스인가 ────
    sec("⑦ 타깃 정의 — CH4 함량 적용 전후 수율")
    e = pd.read_excel("data/monitor_extra.xlsx")
    e["date"] = pd.to_datetime(e["date"])
    x = d2.join(e.set_index("date")[["CH4_pct"]])
    ch4 = x["CH4_pct"].where(x["CH4_pct"].between(45, 75))
    y = x.groupby(x.index.year).apply(lambda g: pd.Series({
        "CH4함량%": ch4[g.index].mean(),
        "수율_원본": (g[TARGET] / g["VS_load"]).mean(),
        "수율_CH4적용": (g[TARGET] * ch4[g.index] / 100 / g["VS_load"]).mean()}))
    print(y.round(3).to_string())
    print("\n  원본 수율이 전 연도 문헌 상한(0.55)을 넘고 CH4 함량을 곱해야 범위에 들어오면,")
    print("  'methane' 컬럼은 CH4 발생량이 아니라 소화가스 발생량이다.")

    sec("⑦-b CH4 함량 이상값")
    s = x["CH4_pct"].dropna()
    for lo, hi, tag in [(0, 45, "45% 미만"), (45, 75, "정상 45~75%"),
                        (75, 100, "75~100%(의심)"), (100, 1e9, "100% 초과(불가능)")]:
        n = ((s >= lo) & (s < hi)).sum()
        print(f"  {tag:20s} {n:5d}건 ({n/len(s)*100:5.1f}%)")
    bad = s[s > 75]
    if len(bad):
        print(f"  이상 사례: {sorted(bad.round(1).tolist())}")


if __name__ == "__main__":
    main()
