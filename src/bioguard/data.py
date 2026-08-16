"""
데이터 적재 — PROMPT §1 데이터 규약.

**중요**: PROMPT 는 `01_일별_소화조상태.csv` 등 7개 파일을 전제하지만 그 파일들은 제공되지
않았다. 대신 저장소 정본인 `data/영천BGP_MASTER_2018-2023.csv` 를 쓴다. 두 자료가 같은
원천임은 재현으로 확인했다 (반입 구성비 59.1/34.7/7.9%, §4 물질수지 3~5% 이내 일치).
상세는 `docs/INGESTION_REPORT.md` §10.

PROMPT 금지사항 6번에 따라 **파생된 `투입톤_*`·`유입비율_*` 을 쓰지 않는다.** 애초에 master
CSV 에는 그 열이 없으므로, 반입 원자료(`intake_total_tpd × 반입비율`)에서 출발하는 규약이
강제된다. 전단 커널은 `kernels.py` 에서 직접 설계한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SUBSTRATES, V_DIGESTER_M3

ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = ROOT / "data" / "영천BGP_MASTER_2018-2023.csv"

_INTAKE_COL = {"foodww": "intake_foodww_tpd", "manure": "intake_manure_tpd",
               "food": "intake_food_tpd"}


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_frame(df: pd.DataFrame | None = None, v_digester: float = V_DIGESTER_M3) -> pd.DataFrame:
    """모델 입력 프레임. 반입 원자료 → 기질별 반입 톤수, 타깃, 물질수지 항, 이화학 상태."""
    M = load_master() if df is None else df.copy()
    out = pd.DataFrame({"date": M.date, "year": M.year, "dow": M.dow})

    # --- 기질별 반입 톤수: 총량 × 구성비 (구성비는 완만히 변하므로 ffill 허용) ---
    total = M.intake_total_tpd.ffill(limit=2)
    ratios = {}
    for s in SUBSTRATES:
        ratios[s] = (M[_INTAKE_COL[s]] / M.intake_total_tpd).ffill(limit=7)
    rsum = sum(ratios.values())
    for s in SUBSTRATES:
        out[f"recv_{s}"] = (total * (ratios[s] / rsum)).fillna(0.0)   # 정규화 후 톤수
        out[f"ratio_{s}"] = (ratios[s] / rsum)

    # --- 타깃: 실측 메탄. 보간하지 않는다(라벨 생성 금지). ---
    out["ch4"] = M.CH4_m3d
    out["biogas"] = M.biogas_AB_m3d
    out["ch4_pct"] = M.CH4_pct

    # --- 물질수지 항 (T2) ---
    out["feed_AB"] = M.feed_AB_tpd
    out["acid_TS_pct"] = M.acid_TS_pct
    out["dig_VS_A_pct"] = M.dig_VS_A_pct
    out["dig_TS_A_pct"] = M.dig_TS_A_pct
    out["OLR"] = M.OLR_kgVS_m3d
    out["TS_load_tpd"] = M.feed_AB_tpd * M.acid_TS_pct / 100.0
    # 유입 VS 는 V 와 무관하다. master 의 OLR 열은 이미 V=8,000㎥ 로 나눈 파생값이므로
    # (OLR×8000 ≡ VS_load_tpd×1000, 중앙값 비 1.000 확인) OLR×V 로 쓰면 V≠8000 에서
    # 유입량을 잘못 스케일한다. 부하 원자료인 VS_load_tpd 를 직접 쓴다.
    out["VS_in_kgd"] = M.VS_load_tpd * 1000.0
    out["VS_out_kgd"] = M.feed_AB_tpd * M.dig_VS_A_pct / 100.0 * 1000  # 유출 VS (Q_out≈Q_in)
    out["VS_reactor_kg"] = v_digester * M.dig_VS_A_pct / 100.0 * 1000  # 조내 VS ← V 의존
    out["VS_consumed_kgd"] = out.VS_in_kgd - out.VS_out_kgd

    # --- 이화학 상태 (T3). A 계열만 사용: 2023 ALK_B 오염(DQ-05) ---
    out["dig_pH_A"] = M.dig_pH_A
    out["dig_T_A"] = M.dig_T_A_C
    out["dig_T_B"] = M.dig_T_B_C
    out["VFA_A"] = M.VFA_A_mgL
    out["ALK_A"] = M.ALK_A_mgL
    out["VFA_ALK_A"] = M.VFA_ALK_A
    out["NH3N_A"] = M.NH3N_A_mgL
    out["FAN_A"] = M.FAN_A_mgL
    out["dig_CODcr_A"] = M.dig_CODcr_A_mgL
    out["acid_CODcr"] = M.acid_CODcr_mgL
    out["HRT_d"] = M.HRT_d
    out["flag_TA"] = M.flag_TA_sensor_fault
    out["flag_TB"] = M.flag_TB_sensor_fault
    return out


def cod_yield(M: pd.DataFrame | None = None) -> pd.Series:
    """Y_COD — 소화조 COD 는 A·B 평균 (docs/INGESTION_REPORT.md §3.1 규약)."""
    M = load_master() if M is None else M
    dig = M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
    removed = M.feed_AB_tpd * (M.acid_CODcr_mgL - dig) / 1000.0
    return M.CH4_m3d / removed


def target_coverage(F: pd.DataFrame) -> dict:
    """타깃 결측 실태. PROMPT §1 은 63일 결측을 말하지만 우리 계열은 더 크다 — 그대로 보고한다."""
    n = len(F)
    obs = int(F.ch4.notna().sum())
    return {"n_days": n, "관측일": obs, "결측일": n - obs,
            "결측률_pct": round(100 * (n - obs) / n, 1),
            "note": "CH4 = biogas × CH4% 이며 CH4% 가 랩 측정(주말 미측정)이라 결측이 크다. "
                    "타깃은 보간하지 않는다 — 라벨을 만들어내면 성능이 착시가 된다."}
