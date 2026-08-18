"""
영천 BGP 메탄예측 — 전역 상수와 규약

모든 물리 상수·품질 규칙·지평 구간을 여기 한 곳에 노출한다.
값을 바꾸려면 이 파일만 고치면 되고, 코드 어디에도 매직넘버를 두지 않는다.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(OUT_DIR, "models")
MASTER_CSV = os.path.join(DATA_DIR, "master_bgp_2018_2023.csv")

RANDOM_STATE = 42

# ── 시설 상수 ────────────────────────────────────────────────────────────────
# 소화조 유효용적. 데이터로 식별되지 않는 외생 입력이다(여섯 경로 전부 실패).
# 운전 실적이 확보되면 이 값만 교체한다.
V_DIGESTER_M3 = 8000.0
N_DIGESTERS = 2                    # 병렬 2기. A/B 는 동일조건 병렬이므로 독립표본이 아니다.
DESIGN_TEMP_C = 38.0               # 중온 설정치

# ── 타깃 ────────────────────────────────────────────────────────────────────
# CH4_m3d = biogas_AB_m3d × CH4_pct / 100.
# 유량은 결측 0 %, 농도는 결측 39.4 % → 타깃 결측의 원인은 전부 '농도'다.
TARGET = "CH4_m3d"
FLOW = "biogas_AB_m3d"
CONC = "CH4_pct"

# ── 예측 지평 구간 (사용자 지정) ─────────────────────────────────────────────
# (이름, 시작일, 끝일). 각 구간마다 별도 모델을 적합하고 별도로 평가한다.
HORIZON_BANDS = [
    ("h1_3", 1, 3),
    ("h3_5", 3, 5),
    ("h5_7", 5, 7),
    ("h7_14", 7, 14),
    ("h14_30", 14, 30),
    ("h30_60", 30, 60),
    ("h60_90", 60, 90),
]

# ── 검증 프로토콜 ────────────────────────────────────────────────────────────
# 확장창 rolling-origin. 무작위 k-fold 금지(일별 자기상관으로 누출).
CV_INITIAL_TRAIN_DAYS = 730        # 초기 학습 2년 (계절 1주기 이상 확보)
CV_STEP_DAYS = 120                 # 원점 전진 간격
CV_VAL_DAYS = 90                   # 폴드 학습셋 말미 검증블록 (모델 선택은 이 안에서만)

# ── 데이터 품질 규칙 ─────────────────────────────────────────────────────────
# 참조문서 DQ 목록 + EXPERT_REVIEW §B3 열역학 검사
DQ = {
    # 2023년 ALK_B 열이 VFA_B 로 오염 → B 계열 알칼리도·VFA/ALK 사용 금지
    "alkb_corrupted_from": "2023-01-01",
    # 소화조 온도 센서 고장 구간 플래그 (master 에 flag 열로 존재)
    "temp_sensor_flags": ["flag_TA_sensor_fault", "flag_TB_sensor_fault"],
    # 2018년 가스 고분산 국면
    "regime_flag": "flag_regime_2018",
    # 열역학 상한 : COD 기준 메탄수율. 0 < Y_COD ≤ 0.35 밖은 물리적으로 불가능하다.
    "y_cod_max": 0.35,
    # 물리 범위 필터 (밖은 결측 처리)
    "ranges": {
        "dig_pH_A": (6.0, 9.0), "dig_pH_B": (6.0, 9.0), "acid_pH": (3.0, 9.0),
        "CH4_pct": (30.0, 80.0), "dig_T_A_C": (20.0, 45.0), "dig_T_B_C": (20.0, 45.0),
        "VFA_A_mgL": (0.0, 20000.0), "ALK_A_mgL": (0.0, 40000.0),
        "biogas_AB_m3d": (0.0, 40000.0), "feed_AB_tpd": (0.0, 500.0),
    },
}

# ── 운전 지침 밴드 (경보가 아니라 '설계·문헌 대비 위치' 진단용) ───────────────
# 문헌 절대임계를 그대로 경보로 쓰면 발화율이 pH 98.4 %, OLR 86.2 % 로 경보가 죽는다.
# 따라서 경보는 시설 자체 90일 기준선 대비 상대편차로 내고(RELATIVE_ALARM),
# 아래 표는 진단 문구에만 쓴다.
GUIDELINE = {
    "VFA_ALK_A": {"good": (0.0, 0.30), "caution": (0.30, 0.40), "danger": (0.40, 10.0)},
    "dig_pH_A": {"good": (6.8, 7.6), "caution": (6.6, 7.8), "danger": (0.0, 20.0)},
    "OLR_kgVS_m3d": {"good": (1.5, 4.0), "caution": (1.0, 5.0), "danger": (0.0, 99.0)},
    "dig_T_A_C": {"good": (37.0, 39.0), "caution": (36.0, 40.0), "danger": (0.0, 99.0)},
    "FAN_A_mgL": {"good": (0.0, 150.0), "caution": (150.0, 300.0), "danger": (300.0, 1e5)},
}
# 시설 자체 기준선 대비 상대편차 경보 (90일 이동 중앙값 대비)
RELATIVE_ALARM = {"baseline_days": 90, "caution_pct": 15.0, "danger_pct": 25.0}

# 경보를 낼 '나쁜 방향'. 편차의 절대값만 보면 개선을 경보로 띄운다 —
# 예컨대 VFA/알칼리도가 기준선보다 32 % 낮은 것은 완충능이 좋아진 것이지 이상이 아니다.
#   up   : 증가가 나쁨 (산 축적·저해·과부하)
#   down : 감소가 나쁨 (생산 저하)
#   both : 어느 쪽이든 이탈이 나쁨 (설정치 관리 대상)
ALARM_DIRECTION = {
    "VFA_ALK_A": "up",          # 산 축적
    "FAN_calc": "up",           # 암모니아 저해
    "OLR_calc": "up",           # 과부하
    "dig_pH_A": "both",         # 완충 붕괴는 양방향
    "dig_T_A_C": "both",        # 중온 설정치 관리
    "biogas_AB_m3d": "down",    # 생산 저하
    "CH4_pct": "down",          # 가스 품질 저하
    "feed_AB_tpd": "both",      # 투입 급변은 어느 쪽이든 주목
    "VS_destruction_pct": "down",
    "Y_COD": "both",
}

# ── 화학양론 상수 ────────────────────────────────────────────────────────────
CH4_PER_KG_COD = 0.35              # ㎥CH₄/kgCOD_제거 (이론 최대, 표준상태)
COD_PER_VS = 1.42                  # kgCOD/kgVS
CH4_PER_KG_VS_MAX = CH4_PER_KG_COD * COD_PER_VS   # ≈ 0.497 ㎥CH₄/kgVS_destroyed
