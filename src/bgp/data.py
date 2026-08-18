"""
자료 적재 · 품질 규칙 · 파생 · 네 각도의 피처 구성

원자료는 `data/master_bgp_2018_2023.csv` (2,086일 × 61열, 2018-01-01 ~ 2023-09-17).
저장소의 옛 `master.xlsx` 는 A/B 계열을 평균 낸 축약본이라 쓰지 않는다.

■ 타깃 구조 — 결측의 원인은 유량이 아니라 농도다
    CH4_m3d = biogas_AB_m3d × CH4_pct / 100
    biogas_AB_m3d 결측 0.0 %   ← 유량계는 매일 기록된다
    CH4_pct       결측 39.4 %   ← 가스분석을 매일 하지 않는다
  따라서 타깃 결측 1,265/2,086 은 전부 농도 미측정일이다. 이 구조를 이용해
  ① 유량은 완전 관측 계열로 시계열 모델에 그대로 쓰고
  ② 농도만 논문 방식 소프트센서로 보간한다(src/bgp/impute.py).

■ 네 각도 (요구사항 2)
    A. 기질 성상   substrate  — 반입 조성·산발효조 성상·여액 성상
    B. 내부 이화학 chemistry  — pH·VFA·ALK·TS/VS·온도·NH₃-N·FAN
    C. VS 물질수지 vsbalance  — 유입/유출/소비 VS, 분해율, OLR, HRT, Y_COD
    D. 시계열 구조 temporal   — 유량·타깃의 시차/이동통계, 달력
  각 각도는 단독으로도 평가한다(ablation). '어느 각도가 실제로 정보를 주는가'가
  예측 성능만큼 중요한 산출물이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bgp import config as C


# ══════════════════════════════════════════════════════════════════════════════
# 1. 적재 + 품질 규칙
# ══════════════════════════════════════════════════════════════════════════════
def load_master(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or C.MASTER_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 달력 연속성 확보 — 빠진 날짜는 전 열 결측 행으로 채운다(시차 연산의 전제)
    full = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    df = full.merge(df, on="date", how="left")
    for c in ("year", "month", "dow"):
        df[c] = getattr(df["date"].dt, {"year": "year", "month": "month", "dow": "dayofweek"}[c])
    return apply_dq(df)


def apply_dq(df: pd.DataFrame) -> pd.DataFrame:
    """물리 범위 밖 값을 결측 처리하고, 오염·고장 구간을 사용 금지 처리한다."""
    df = df.copy()

    for col, (lo, hi) in C.DQ["ranges"].items():
        if col in df.columns:
            bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            df.loc[bad, col] = np.nan

    # 2023년 ALK_B 는 VFA_B 로 오염됐다 → B 계열 알칼리도 계통 전부 폐기
    cut = pd.Timestamp(C.DQ["alkb_corrupted_from"])
    for col in ("ALK_B_mgL", "VFA_ALK_B"):
        if col in df.columns:
            df.loc[df["date"] >= cut, col] = np.nan

    # 온도 센서 고장 플래그 구간의 온도는 값이 아니라 잡음이다
    for flag, col in zip(C.DQ["temp_sensor_flags"], ("dig_T_A_C", "dig_T_B_C")):
        if flag in df.columns and col in df.columns:
            df.loc[df[flag].fillna(0) > 0, col] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. 파생 — 물질수지·화학양론·저해지표
# ══════════════════════════════════════════════════════════════════════════════
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    V = C.V_DIGESTER_M3

    # ── 기질 조성비 (반입 톤수 → 비율). 반입 ≠ 투입이지만 '조성'은 이전된다.
    tot = df["intake_total_tpd"].replace(0, np.nan)
    for src, name in (("intake_foodww_tpd", "mix_foodww"),
                      ("intake_manure_tpd", "mix_manure"),
                      ("intake_food_tpd", "mix_food")):
        df[name] = df[src] / tot

    # ── VS 물질수지 (밀도비 1, 조내농도 = 유출농도)
    #    유입 VS [kg/d] = 투입 t/d × 산발효조 TS% × (VS/TS)
    vs_ts = df["acid_VS_pct"] / df["acid_TS_pct"].replace(0, np.nan)
    df["vs_ts_ratio_feed"] = vs_ts
    df["VS_in_kgd"] = df["feed_AB_tpd"] * df["acid_TS_pct"] * 10.0 * vs_ts
    #    유출 VS [kg/d] = 유출유량 × 소화액 VS%. SRT=HRT·밀도1 가정에서 Q_out ≈ Q_in.
    df["VS_out_kgd"] = df["feed_AB_tpd"] * df["dig_VS_A_pct"] * 10.0
    df["VS_destroyed_kgd"] = df["VS_in_kgd"] - df["VS_out_kgd"]
    df["VS_destruction_pct"] = 100 * df["VS_destroyed_kgd"] / df["VS_in_kgd"].replace(0, np.nan)
    #    조내 VS [kg] = V × 소화액 VS%  (V 는 외생 상수)
    df["VS_reactor_kg"] = V * df["dig_VS_A_pct"] * 10.0

    # ── COD 물질수지와 열역학 검사 (EXPERT_REVIEW §B1·B3)
    #    Y_COD = CH₄ / 제거 COD. 0 < Y_COD ≤ 0.35 밖은 물리적으로 불가능하다.
    cod_removed = df["feed_AB_tpd"] * (df["acid_CODcr_mgL"] - df["dig_CODcr_A_mgL"]) / 1000.0
    df["COD_removed_kgd"] = cod_removed
    df["Y_COD"] = df[C.TARGET] / cod_removed.replace(0, np.nan)
    df["flag_thermo_violation"] = (
        df["Y_COD"].notna() & ((df["Y_COD"] <= 0) | (df["Y_COD"] > C.DQ["y_cod_max"]))
    ).astype(int)

    # ── 부하 지표. OLR·HRT 는 V 가 상수이므로 투입·부하의 함수 변환일 뿐이다.
    df["OLR_calc"] = df["VS_in_kgd"] / V
    df["HRT_calc"] = V / df["feed_AB_tpd"].replace(0, np.nan)

    # ── 저해 지표 : 유리 암모니아 (Anthonisen)
    T_K = df["dig_T_A_C"].fillna(C.DESIGN_TEMP_C) + 273.15
    pKa = 0.09018 + 2729.92 / T_K
    df["FAN_calc"] = df["NH3N_A_mgL"] / (1.0 + 10.0 ** (pKa - df["dig_pH_A"]))

    # ── 완충능
    df["VFA_ALK_A"] = df["VFA_A_mgL"] / df["ALK_A_mgL"].replace(0, np.nan)
    df["ALK_minus_VFA"] = df["ALK_A_mgL"] - df["VFA_A_mgL"]

    # ── 가스 품질·수율 (진단용. 분모가 금일 투입이라 예측 타깃으로는 순환적이다)
    df["CH4_per_VS_in"] = df[C.TARGET] / df["VS_in_kgd"].replace(0, np.nan)
    df["gas_per_feed"] = df[C.FLOW] / df["feed_AB_tpd"].replace(0, np.nan)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. 네 각도의 원천 변수 정의
# ══════════════════════════════════════════════════════════════════════════════
ANGLE_SUBSTRATE = [                       # A. 기질 성상
    "intake_total_tpd", "intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd",
    "mix_foodww", "mix_manure", "mix_food",
    "feed_AB_tpd", "feed_A_tpd", "feed_B_tpd",
    "acid_pH", "acid_TS_pct", "acid_VS_pct", "acid_CODcr_mgL", "vs_ts_ratio_feed",
    "leach_pH", "leach_TS_pct", "leach_VS_pct",
]
ANGLE_CHEMISTRY = [                       # B. 소화조 내부 이화학
    "dig_pH_A", "dig_pH_B", "dig_T_A_C", "dig_T_B_C",
    "VFA_A_mgL", "ALK_A_mgL", "VFA_ALK_A", "ALK_minus_VFA",
    "dig_TS_A_pct", "dig_VS_A_pct", "dig_TS_B_pct", "dig_VS_B_pct",
    "dig_CODcr_A_mgL", "NH3N_A_mgL", "FAN_calc",
]
ANGLE_VSBALANCE = [                       # C. VS 물질수지
    "VS_in_kgd", "VS_out_kgd", "VS_destroyed_kgd", "VS_destruction_pct",
    "VS_reactor_kg", "VS_load_tpd", "COD_removed_kgd", "Y_COD",
    "OLR_calc", "HRT_calc", "CH4_per_VS_in", "gas_per_feed",
]
ANGLE_TEMPORAL_BASE = [C.FLOW, C.TARGET, C.CONC]   # D. 시계열 구조의 원천

ANGLES = {
    "substrate": ANGLE_SUBSTRATE,
    "chemistry": ANGLE_CHEMISTRY,
    "vsbalance": ANGLE_VSBALANCE,
    "temporal": ANGLE_TEMPORAL_BASE,
}


def build_frame(path: str | None = None) -> pd.DataFrame:
    """적재 → DQ → 파생까지 끝난 일별 프레임."""
    return add_derived(load_master(path))


def quality_report(df: pd.DataFrame) -> dict:
    """자료 품질 요약 — 파이프라인 첫 산출물. 여기서 막히면 이후가 전부 무효다."""
    obs = df[C.TARGET].notna()
    rep = {
        "n_days": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "target_observed": int(obs.sum()),
        "target_missing_pct": round(100 * float(1 - obs.mean()), 1),
        "flow_missing_pct": round(100 * float(df[C.FLOW].isna().mean()), 1),
        "conc_missing_pct": round(100 * float(df[C.CONC].isna().mean()), 1),
        "thermo_violation_pct": round(
            100 * float(df.loc[df["Y_COD"].notna(), "flag_thermo_violation"].mean()), 1),
        "missing_by_angle": {
            k: round(100 * float(df[[c for c in v if c in df.columns]].isna().mean().mean()), 1)
            for k, v in ANGLES.items()},
    }
    # 열역학 위반은 연도별로 극단 편중돼 있다 — 기준선 산출에서 제외할 연도를 찾는다
    sub = df[df["Y_COD"].notna()]
    rep["thermo_violation_by_year"] = {
        int(y): round(100 * float(g["flag_thermo_violation"].mean()), 1)
        for y, g in sub.groupby("year")}
    return rep
