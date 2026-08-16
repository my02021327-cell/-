"""
영천BGP MASTER 데이터셋 수용 검증 (acceptance validation)

`data/영천BGP_MASTER_2018-2023.csv` 가 참조문서(`docs/reference/영천BGP_AI참조문서.md`),
설계문서(`docs/MODELING_OVERVIEW.md`), 전문가검토서(`docs/EXPERT_REVIEW.md`)에 기재된
정량 주장과 실제로 일치하는지 재계산해 대조한다. 문서를 믿지 않고 데이터로 확인하는 것이
목적이므로, 각 검사는 문서에 적힌 기대값(expected)을 코드에 박아두고 허용오차 안에서
재현되는지 판정한다.

    python scripts/validate_master.py

종료코드 0 = 전 항목 통과. 실패 항목이 있으면 1.

주의 — 검증 결과 확정된 재현 규약 두 가지 (문서에 명시되지 않아 재현이 갈렸던 부분):
  1) Y_COD 의 소화조 COD 는 **A·B 평균**을 쓴다 (A 단독이면 n=1,065·열역학위반 15.1%로
     검토서의 n=1,072·14.2% 와 어긋난다).
  2) 연도별 Y_COD 중앙값과 FAN 상관은 **열역학 필터(0 < Y_COD ≤ 0.35) 적용 후**에 낸다.
     필터 전에 집계하면 2018년이 0.423 으로 튄다(2018년 관측의 72%가 물리적으로 불가능).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path(__file__).resolve().parents[1] / "data" / "영천BGP_MASTER_2018-2023.csv"

_results: list[tuple[bool, str]] = []


def check(name: str, got, expected=None, tol: float = 0.0, note: str = "") -> None:
    """expected 가 주어지면 |got-expected| <= tol 로 판정, 없으면 got 를 참으로 판정."""
    if expected is None:
        ok = bool(got)
        detail = f"{got}"
    else:
        ok = abs(float(got) - float(expected)) <= tol
        detail = f"got={got:.4g} exp={expected:.4g} tol={tol:g}"
    _results.append((ok, name))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}{('  # ' + note) if note else ''}")


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _fit_metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[m], yhat[m]
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, float(np.sqrt(((y - yhat) ** 2).mean())), float(np.abs(y - yhat).mean()), int(m.sum())


def check_shape(df: pd.DataFrame) -> None:
    print("\n[1] 구조 — 참조문서 부록A: 2,086행 × 61열, 2018-01-01~2023-09-17 일별 연속")
    check("행수 2086", df.shape[0], 2086)
    check("열수 61", df.shape[1], 61)
    check("시작일 2018-01-01", df.date.min() == pd.Timestamp("2018-01-01"))
    check("종료일 2023-09-17 (DQ-01: 이후 105일 공백)", df.date.max() == pd.Timestamp("2023-09-17"))
    check("날짜 중복 없음", df.date.duplicated().sum() == 0)
    check("날짜 결손 없음(전일 연속)", (df.date.diff().dt.days.dropna() != 1).sum() == 0)


def check_persistence(df: pd.DataFrame) -> None:
    """검토서 A2: 지평별 persistence. h=1은 ML 여지 없음, h>=7에서 붕괴."""
    print("\n[2] persistence 베이스라인 — 검토서 A2 지평별 표 (검증구간 2022~2023, n=625)")
    s = df.set_index("date")["biogas_AB_m3d"]
    val = s.index >= pd.Timestamp("2022-01-01")
    for h, exp_r2, exp_rmse in [(1, 0.932, 460), (3, 0.800, 789), (7, 0.566, 1161),
                                (14, 0.252, 1524), (30, -0.564, 2204)]:
        r2, rmse, _, n = _fit_metrics(s[val], s.shift(h)[val])
        check(f"h={h:<2d} persistence R²", r2, exp_r2, 0.002)
        check(f"h={h:<2d} persistence RMSE", rmse, exp_rmse, 1.0)
        check(f"h={h:<2d} n=625", n, 625)

    print("\n[3] persistence + Δfeed — 설계문서 §4.1 (계수 33.26, 검증 R²=0.942)")
    d = df[["date", "biogas_AB_m3d", "feed_AB_tpd"]].copy()
    d["y1"] = d.biogas_AB_m3d.shift(1)
    d["dfeed"] = d.feed_AB_tpd.diff()
    d["resid"] = d.biogas_AB_m3d - d.y1
    tr = d[d.date < "2022-01-01"].dropna(subset=["resid", "dfeed"])          # 학습기간에서만 적합
    slope, intercept, r, _, _ = stats.linregress(tr.dfeed, tr.resid)
    check("Δfeed 계수 33.26 ㎥/t", slope, 33.26, 0.01)
    check("잔차회귀 R² (참조문서 0.418~0.441)", r ** 2, 0.44, 0.03)
    te = d[d.date >= "2022-01-01"].dropna(subset=["biogas_AB_m3d", "y1", "dfeed"])
    r2, rmse, mae, _ = _fit_metrics(te.biogas_AB_m3d, te.y1 + slope * te.dfeed + intercept)
    check("검증 R² 0.942", r2, 0.942, 0.002)
    check("검증 RMSE 425", rmse, 425, 1.0)
    check("검증 MAE 293", mae, 293, 1.0)

    print("\n[4] 자기상관 — 참조문서 §5.1 (확률보행 구조, 차분 후 백색잡음)")
    b = df["biogas_AB_m3d"]
    for lag, exp in [(1, 0.916), (7, 0.670), (30, 0.262), (365, 0.011)]:
        check(f"ACF(lag={lag})", b.autocorr(lag), exp, 0.002)
    check("차분 ACF(1) ≈ 0 (백색잡음 근접)", b.diff().autocorr(1), -0.059, 0.002)


def check_fan(df: pd.DataFrame) -> None:
    """검토서 A1: FAN>300 절대임계는 75.1%에서 발화 → 경보로 사용 불가."""
    print("\n[5] 유리암모니아(FAN) — 검토서 A1 발화율 / 참조문서 §5.3")
    f = df["FAN_A_mgL"].dropna()
    check("FAN_A 유효 n=229 (NH₃-N 측정일만)", len(f), 229)
    check("FAN_A 평균 410 mg/L", f.mean(), 410, 1.0)
    check("FAN>150 발화율 98.7%", (f > 150).mean() * 100, 98.7, 0.1)
    check("FAN>300 발화율 75.1% → 경보 불가", (f > 300).mean() * 100, 75.1, 0.1)
    check("2021년 FAN 중앙값 595 (정점)", df[df.year == 2021].FAN_A_mgL.median(), 595, 1.0)
    check("2023년 NH₃-N 측정 0건", df[df.year == 2023].NH3N_A_mgL.notna().sum(), 0)

    # FAN 열의 산출 규약 확인: 온도는 '센서결함 아닌 실측, 결함이면 38.0' 이다.
    c = df.dropna(subset=["NH3N_A_mgL", "dig_pH_A", "FAN_A_mgL"])
    t_used = c["dig_T_A_C"].where(c["flag_TA_sensor_fault"] == 0).fillna(38.0)
    pka = 0.09018 + 2729.92 / (t_used + 273.15)
    recomputed = c.NH3N_A_mgL / (1 + 10 ** (pka - c.dig_pH_A))
    check("FAN 열 = Anthonisen 식 재현(실측T, 결함시 38)", np.abs(recomputed - c.FAN_A_mgL).max(), 0.0, 0.01)


def check_ycod(df: pd.DataFrame) -> None:
    """검토서 B1/B2/B3: COD 기준 수율이 정규화 성능 KPI."""
    print("\n[6] Y_COD — 검토서 B1(수율)·B2(FAN 역상관)·B3(열역학 필터)")
    d = df.copy()
    dig_cod = d[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)     # ← A·B 평균이 규약
    removed = d.feed_AB_tpd * (d.acid_CODcr_mgL - dig_cod) / 1000.0      # kg COD/d
    d["Y_COD"] = d.CH4_m3d / removed

    v = d.Y_COD.dropna()
    check("Y_COD 유효 n=1,072", len(v), 1072)
    check("Y_COD 중앙값 0.298 (이론상한 0.35의 85%)", v.median(), 0.298, 0.001)
    check("COD 제거 중앙값 23,237 kg/d", removed.median(), 23237, 60)
    impossible = ((v > 0.35) | (v <= 0))
    check("열역학 위반 14.2% (B3 QC 규칙)", impossible.mean() * 100, 14.2, 0.1)

    # 필터 적용 후에 집계해야 검토서 표가 재현된다.
    f = d[(d.Y_COD > 0) & (d.Y_COD <= 0.35)]
    check("2021년 Y_COD 중앙값 0.250 (FAN 정점 ↔ 수율 최저)", f[f.year == 2021].Y_COD.median(), 0.250, 0.001)
    check("2018년 Y_COD 중앙값 0.281", f[f.year == 2018].Y_COD.median(), 0.281, 0.001)
    pair = f.dropna(subset=["FAN_A_mgL"])
    rho, pval = stats.spearmanr(pair.Y_COD, pair.FAN_A_mgL)
    check("Y_COD vs FAN Spearman ρ = −0.410", rho, -0.410, 0.002)
    check("  p < 1e-7", pval < 1e-7, note=f"p={pval:.2e}, n={len(pair)}")


def check_dq_flags(df: pd.DataFrame) -> None:
    """참조문서 §3 DQ 목록이 플래그 열로 실제 반영되어 있는지."""
    print("\n[7] DQ 플래그 — 참조문서 §3 / 설계문서 §2.2")
    for flag, col, dq in [("flag_TA_sensor_fault", "dig_T_A_C", "DQ-02"),
                          ("flag_TB_sensor_fault", "dig_T_B_C", "DQ-03"),
                          ("flag_ALKB_corrupted", "ALK_B_mgL", "DQ-05")]:
        masked = df.loc[df[flag] == 1, col]
        check(f"{dq}: {flag} 구간 {col} 전부 NaN", masked.notna().sum() == 0,
              note=f"{int((df[flag] == 1).sum())}일 마스킹됨")
    check("DQ-05: 2023년 ALK_B 사용 가능값 0건", df[df.year == 2023].ALK_B_mgL.notna().sum() == 0)
    check("DQ-04: feed 고정 48일 플래그", int((df.flag_feed_frozen == 1).sum()), 48)
    check("2018 고분산 국면 플래그 365일", int((df.flag_regime_2018 == 1).sum()), 365)

    print("\n[8] DQ-08 VS 측정법 계단변화 — 2020년 기점 VS/TS 0.80 → 0.70")
    ratio = df.groupby("year").acid_VS_pct.mean() / df.groupby("year").acid_TS_pct.mean()
    check("2019 VS/TS ≈ 0.80", ratio[2019], 0.796, 0.005)
    check("2020 VS/TS ≈ 0.70 (계단 하락)", ratio[2020], 0.705, 0.005)
    check("하락폭 ≥ 0.08 → in_VS 절대값 사용 금지", (ratio[2019] - ratio[2020]) > 0.08)


def check_structure(df: pd.DataFrame) -> None:
    """구조적 결측·저류조 완충·A/B 비독립성 — 검증 프로토콜의 근거."""
    print("\n[9] 구조적 결측 (NOT MCAR) — 참조문서 §2.1 / 설계문서 §2.4")
    lab = df.groupby("dow").dig_pH_A.apply(lambda s: s.notna().mean() * 100)
    flow = df.groupby("dow").feed_AB_tpd.apply(lambda s: s.notna().mean() * 100)
    check("랩 항목 주중(월~목) 측정률 ≥ 85%", lab.loc[0:3].min() > 85, note=f"{lab.loc[0:3].min():.1f}%")
    check("랩 항목 금요일 급감 (~67%)", lab[4], 67.1, 1.0)
    check("랩 항목 주말 ≈ 0% → 선형보간 금지", lab.loc[5:6].max() < 2.0, note=f"{lab.loc[5:6].max():.1f}%")
    check("유량 항목은 요일 무관 ~100%", flow.min() > 99, note=f"{flow.min():.1f}%")

    print("\n[10] 저류조 완충 — 설계문서 §1.3 (반입 ≠ 투입)")
    sun = df[df.dow == 6]
    check("일요일 반입 22.3 t/d (급감)", sun.intake_total_tpd.mean(), 22.3, 0.1)
    check("일요일 소화조 투입 196.5 t/d (평일과 동일)", sun.feed_AB_tpd.mean(), 196.5, 0.1)
    check("반입합계-가스 상관 r≈0.138 → 피처 제외", df.biogas_AB_m3d.corr(df.intake_total_tpd), 0.138, 0.005)
    check("투입(feed_AB)-가스 상관 r=0.540 → 부하변수는 이것만", df.biogas_AB_m3d.corr(df.feed_AB_tpd), 0.540, 0.005)

    print("\n[11] A/B 병렬 대칭 — 독립표본 취급 금지 (설계문서 §4.4)")
    check("A/B 바이오가스 상관 r=0.927", df.biogas_A_m3d.corr(df.biogas_B_m3d), 0.927, 0.002)
    share = 100 * df.biogas_A_m3d.sum() / (df.biogas_A_m3d.sum() + df.biogas_B_m3d.sum())
    check("A 분담률 ≈ 51%", share, 50.8, 0.5)

    print("\n[12] 설계 제원 — 부하·체류시간 여유 (설계문서 §1.1)")
    check("HRT 중앙값 40.5일", df.HRT_d.median(), 40.5, 0.1)
    check("OLR 평균 1.14 kgVS/㎥·d (문헌 1.5~4.0 하단)", df.OLR_kgVS_m3d.mean(), 1.143, 0.01)
    check("CH₄ 평균 66.0%", df.CH4_pct.mean(), 66.0, 0.1)


def main() -> int:
    if not CSV.exists():
        print(f"데이터 없음: {CSV}", file=sys.stderr)
        return 2
    print(f"검증 대상: {CSV}")
    df = load()
    check_shape(df)
    check_persistence(df)
    check_fan(df)
    check_ycod(df)
    check_dq_flags(df)
    check_structure(df)

    failed = [n for ok, n in _results if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_results) - len(failed)}/{len(_results)} 통과")
    for n in failed:
        print(f"  실패: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
