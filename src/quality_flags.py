"""데이터 품질 이상 기록을 별도 파일로 분리 저장한다.

원자료는 그대로 두고(측정값을 참으로 가정한 모델링을 위해), 의심 구간만
플래그로 기록해 나중에 현장 확인·재분석 시 대조할 수 있게 한다.
출력: outputs/data_quality_flags.csv, outputs/data_quality_flags.md
실행: `python -m src.quality_flags`
"""
import numpy as np
import pandas as pd

from src.final_ensemble import prepare, TARGET

OUT = "outputs"
# 물리·문헌 기반 가능 범위
PLAUS = {
    "소화조_pH": (6.0, 9.0), "유입_pH": (3.0, 9.0), "소화조_온도": (15.0, 60.0),
    "소화조_VS": (0.1, 10.0), "소화조_TS": (0.2, 15.0), "소화조_VFA": (50, 15000),
    "소화조_TAlk": (2000, 40000), "소화조_MAlk": (1000, 40000),
    "소화조_CODcr": (5000, 90000), "소화조_NH4N": (200, 8000),
    "유입_TS": (0.5, 30.0), "유입_VS": (0.3, 30.0), "유입_CODcr": (5000, 250000),
    "CH4_pct": (45.0, 75.0),
}
FROZEN_MIN = 5          # 동일값 연속 n일 이상이면 센서 정지 의심
SEVERITY = {"치명": 3, "심각": 2, "주의": 1}


def rows_range(d):
    out = []
    for c, (lo, hi) in PLAUS.items():
        if c not in d:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        bad = s.notna() & ((s < lo) | (s > hi))
        for t in d.index[bad]:
            out.append(dict(date=t.date(), variable=c, flag="범위이탈", value=s[t],
                            detail=f"가능범위 [{lo:g}, {hi:g}]", severity="심각"))
    return out


def rows_frozen(d, cols):
    out = []
    for c in cols:
        s = pd.to_numeric(d[c], errors="coerce").dropna()
        if len(s) < 20:
            continue
        run = s.groupby((s != s.shift()).cumsum())
        for _, seg in run:
            if len(seg) >= FROZEN_MIN:
                sev = "치명" if len(seg) >= 20 else "심각" if len(seg) >= 10 else "주의"
                out.append(dict(date=seg.index.min().date(), variable=c, flag="고정값",
                                value=float(seg.iloc[0]),
                                detail=f"{len(seg)}일 연속 동일값 (~{seg.index.max().date()})",
                                severity=sev))
    return out


def rows_consistency(d):
    out = []
    checks = [
        ("유입_VS>유입_TS", d["유입_VS"] > d["유입_TS"], "유입_VS", "VS 는 TS 를 넘을 수 없음", "심각"),
        ("소화조_VS>소화조_TS", d["소화조_VS"] > d["소화조_TS"], "소화조_VS", "VS 는 TS 를 넘을 수 없음", "심각"),
        ("MAlk>TAlk", d["소화조_MAlk"] > d["소화조_TAlk"], "소화조_MAlk", "M-Alk 는 T-Alk 를 넘을 수 없음", "심각"),
        ("VFA==TAlk", d["소화조_VFA"] == d["소화조_TAlk"], "소화조_TAlk", "두 열 값이 동일 — 열 복사 의심", "치명"),
    ]
    for name, mask, var, detail, sev in checks:
        mask = mask.fillna(False)
        for t in d.index[mask]:
            out.append(dict(date=t.date(), variable=var, flag=name,
                            value=float(d.loc[t, var]), detail=detail, severity=sev))
    return out


def rows_regime(d, cols):
    """분기평균이 직전 4분기 대비 |z|>3 이면 체제 단절 후보."""
    out = []
    q = d.resample("QE")[cols].mean()
    for c in cols:
        s = q[c].dropna()
        if len(s) < 8:
            continue
        base_m = s.rolling(4).mean().shift(1)
        base_s = s.rolling(4).std().shift(1)
        z = (s - base_m) / base_s
        for t, v in z[z.abs() > 3].dropna().items():
            out.append(dict(date=t.date(), variable=c, flag="체제단절",
                            value=float(s[t]), severity="치명" if abs(v) > 5 else "심각",
                            detail=f"분기평균 z={v:+.1f} (직전4분기 {base_m[t]:.4g} → {s[t]:.4g})"))
    return out


def main():
    d = prepare()
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    e = pd.read_excel("data/monitor_extra.xlsx")
    e["date"] = pd.to_datetime(e["date"])
    d = d.join(e.set_index("date")[["CH4_pct", "NH4N"]])

    num = ["소화조_pH", "소화조_온도", "소화조_VS", "소화조_TS", "소화조_VFA",
           "소화조_TAlk", "소화조_MAlk", "소화조_CODcr", "CH4_pct", TARGET, "투입량합계"]
    recs = rows_range(d) + rows_frozen(d, num) + rows_consistency(d) + \
        rows_regime(d, ["소화조_온도", "소화조_TAlk", "소화조_MAlk", "소화조_TS",
                        "소화조_VS", "소화조_VFA", "소화조_CODcr", "소화조_pH"])

    f = pd.DataFrame(recs)
    f["rank"] = f["severity"].map(SEVERITY)
    f = f.sort_values(["rank", "date"], ascending=[False, True]).drop(columns="rank")
    f = f[["date", "variable", "flag", "severity", "value", "detail"]]
    f.to_csv(f"{OUT}/data_quality_flags.csv", index=False, encoding="utf-8-sig")

    lines = ["# 데이터 품질 플래그 (2018–2023)", "",
             "원자료는 수정하지 않는다. 측정값을 참으로 가정한 모델링과 별개로,",
             "현장 확인·재분석이 필요한 구간만 여기에 분리 기록한다.", "",
             f"총 {len(f)}건 — " + ", ".join(f"{k} {v}건" for k, v in
                                            f["severity"].value_counts().items()), ""]
    for sev in ["치명", "심각", "주의"]:
        sub = f[f["severity"] == sev]
        if not len(sub):
            continue
        lines += [f"## {sev} ({len(sub)}건)", "",
                  "| 날짜 | 변수 | 유형 | 값 | 내용 |", "|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['date']} | {r['variable']} | {r['flag']} | "
                         f"{r['value']:.4g} | {r['detail']} |")
        lines.append("")
    open(f"{OUT}/data_quality_flags.md", "w", encoding="utf-8").write("\n".join(lines))

    print(f"[저장] {OUT}/data_quality_flags.csv  ({len(f)}건)")
    print(f"[저장] {OUT}/data_quality_flags.md")
    print("\n등급별:", f["severity"].value_counts().to_dict())
    print("\n변수별 상위:")
    print(f.groupby(["variable", "flag"]).size().sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
