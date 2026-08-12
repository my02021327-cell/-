"""
소화조 상태 점검 프로그램 (health check)

■ 예측 모델(bio_model.py)과의 역할 분담
  예측 모델 : 가스 이력만 사용. 생물학 상태변수는 예측력이 없어 제외(검증됨).
  본 프로그램 : 생물학 상태를 전담. **암모니아 저해를 명시적으로 고려한다.**

  근거 — 생물학 상태의 일별 상대변화는 0.25~1.7%(pH·알칼리도)로 거의 정지 상태이고
  가스 일별 변화(2.7%)를 설명하지 못한다. 즉 생물학은 '수준과 연 추세'를 결정하므로
  단기 예측 피처가 아니라 **상태 진단 지표**로 쓰는 것이 옳다.

■ 급성 경보와 만성 상태를 분리한다 (설계의 핵심)
  이 둘을 한 지표로 묶으면 반드시 실패한다. 발화율의 의미가 서로 다르기 때문이다.

  [급성 경보] 오늘 조치가 필요한 이탈. **발화율이 낮아야** 경보로 기능한다.
    A1 성능저하   : Y_COD(COD 기준 메탄수율)가 자체 90일 기준선 대비 −15%
    A2 급성산성화 : VFA/알칼리도 3일 이동평균 > 0.30
    A3 설명불가   : 가스 예측 잔차 |z|>3 이 2일 연속
    A4 FAN 급등   : FAN이 자체 90일 기준선의 1.2배 초과 (측정일 한정)

  [만성 상태] 수개월~수년 지속되는 부하 국면. **발화율 = 그 상태의 실제 유병률**이므로
              높게 나오는 것이 정상이며, 경보가 아니라 국면 분류다.
    C1 암모니아 부하지수 : 알칼리도의 건강기준기간(2019) 대비 상승률
       정상 <5% / 경계 5~10% / 만성부하 10~15% / 고부하 ≥15%

■ v2 설계의 오류와 정정
  v2는 FAN>300 mg/L을 '최상위 경보'로 두었다. 발화율 75.1%로 경보 기능이 없었다.
  원인은 두 겹이다.
   (1) 이 시설은 TAN 3,832~4,913 mg/L에서 6년간 붕괴 없이 운전된 **순치(acclimated)된
       계**이므로 비순치 문헌 임계(150/300/600)를 그대로 쓸 수 없다.
   (2) 더 근본적으로, 만성 상태를 급성 경보로 표현하려 한 범주 오류였다.
       → 급성분(A4, 상대편차)과 만성분(C1, 부하지수)으로 분리해 해결한다.

  회전창 변화율만 쓰는 것도 실패한다(초기 구현에서 확인): 알칼리도가 2020년에 상승한 뒤
  고원에 머무르므로, 변화율 지표는 최악 연도(2021: FAN 595, Y_COD 0.250 최저)에서 오히려
  발화율이 최저(1.4%)가 된다. 만성 축적에는 **고정 기준기간 대비 수준**이 필요하다.

■ 알칼리도를 읽는 방향 (이 시설 고유 — 반드시 주의)
  **알칼리도 상승을 완충능 개선으로 읽으면 안 된다.** NH4+가 CO2와 결합해 NH4HCO3를
  형성하며 총알칼리도로 계측되므로(ALK = 14,089 + 0.640·TAN), 이 시설에서 알칼리도
  상승은 암모니아 축적 신호다. 알칼리도는 TAN과 rho=0.615(p=2.3e-25, n=1,257 일별)로
  가장 강하게 연동되는 **상시 측정** 변수이며, NH3-N이 2023년 0건인 상황에서 유일하게
  가용한 일별 암모니아 프록시다. 단 절대값 추정용이 아니라 국면 분류용으로만 쓴다.

실행 : python -m src.health_check
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.bio_model import Y_COD_MAX, load

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# ── 급성 경보 임계 (자체 기준선 대비 상대)
A1_DROP_PCT = -15.0      # Y_COD가 90일 기준선 대비 -15% 이하
A2_VFA_ALK = 0.30        # VFA/알칼리도 3일 이동평균 (문헌 절대임계 — 급성 전용)
A3_SIGMA, A3_RUN = 3.0, 2    # 잔차 |z|, 연속 일수
A4_FAN_RATIO = 1.20      # FAN이 자체 90일 기준선의 1.20배 초과

# ── 만성 상태 국면 (건강기준기간 대비 알칼리도 상승률 %)
REF_YEAR = 2019          # Y_COD 최고(0.310) = 건강 기준기간
C1_BANDS = [(-99, 5, "정상"), (5, 10, "경계"), (10, 15, "만성부하"), (15, 999, "고부하")]


def compute_indicators(d: pd.DataFrame) -> pd.DataFrame:
    f = d.copy()

    # ── 정규화 성능 (COD 기준 메탄수율). 열역학 상한으로 QC.
    f["Y_COD"] = f["CH4_m3d"] / f["COD_removed"]
    f.loc[(f["Y_COD"] <= 0) | (f["Y_COD"] > Y_COD_MAX), "Y_COD"] = np.nan

    # ══ 급성 경보 ══════════════════════════════════════════
    # A1 성능저하
    f["Y_COD_base"] = f["Y_COD"].rolling(90, min_periods=30).median()
    f["A1_dev_pct"] = 100 * (f["Y_COD"] / f["Y_COD_base"] - 1)
    f["A1"] = f["A1_dev_pct"] <= A1_DROP_PCT

    # A2 급성 산성화
    f["VFA_ALK_m3"] = f["VFA_ALK"].rolling(3, min_periods=2).mean()
    f["A2"] = f["VFA_ALK_m3"] > A2_VFA_ALK

    # A3 설명 불가 이탈 — 가스 예측 잔차 관리도
    #    예측은 평균회귀만 사용(생물학 상태변수는 예측 기여 없음이 검증됨)
    y = f["biogas_AB_m3d"]
    pred = y.shift(1) + 0.38 * (y.shift(1).rolling(30, min_periods=8).mean() - y.shift(1))
    res = y - pred
    z = (res - res.rolling(90, min_periods=30).mean()) / res.rolling(90, min_periods=30).std()
    f["A3_z"] = z
    hit = (z.abs() > A3_SIGMA).fillna(False)
    f["A3"] = hit & hit.shift(1).fillna(False)

    # A4 FAN 급등 (측정일 한정)
    f["FAN_base"] = f["FAN_A_mgL"].rolling(90, min_periods=8).median()
    f["A4_FAN_ratio"] = f["FAN_A_mgL"] / f["FAN_base"]
    f["A4"] = f["A4_FAN_ratio"] > A4_FAN_RATIO

    acute = ["A1", "A2", "A3", "A4"]
    f["n_acute"] = f[acute].fillna(False).astype(int).sum(axis=1)
    f["acute_grade"] = pd.cut(f["n_acute"], [-1, 0, 1, 9],
                              labels=["정상", "주의", "경보"])

    # ══ 만성 상태 ══════════════════════════════════════════
    # C1 암모니아 부하지수 : 건강기준기간 대비 알칼리도 상승률
    ref = f.loc[f["year"] == REF_YEAR, "ALK_A_mgL"].median()
    f["C1_burden_pct"] = 100 * (f["ALK_A_mgL"] / ref - 1)
    f["C1_burden_sm"] = f["C1_burden_pct"].rolling(30, min_periods=8).median()
    f["chronic_phase"] = pd.cut(
        f["C1_burden_sm"],
        [b[0] for b in C1_BANDS] + [C1_BANDS[-1][1]],
        labels=[b[2] for b in C1_BANDS],
    )
    return f


def summarize(f: pd.DataFrame) -> dict:
    n = len(f)
    out = {"n_days": n, "acute": {}, "chronic": {}}
    for k, label, note in [
        ("A1", "성능저하 (Y_COD 90일 기준선 −15%)", "정규화 성능 이탈"),
        ("A2", "급성 산성화 (VFA/ALK 3일 > 0.30)", "둔감 — 급성 전용"),
        ("A3", "설명불가 이탈 (잔차 |z|>3, 2일연속)", "일반 탐지기"),
        ("A4", "FAN 급등 (자체 기준선 1.2배)", "측정일 n=229 한정"),
    ]:
        s = f[k].fillna(False)
        out["acute"][k] = {"label": label, "days": int(s.sum()),
                           "rate_pct": round(100 * float(s.mean()), 1), "note": note}
    g = f["acute_grade"].value_counts().reindex(["정상", "주의", "경보"]).fillna(0)
    out["acute_grades"] = {k: int(v) for k, v in g.items()}

    ph = f["chronic_phase"].value_counts().reindex([b[2] for b in C1_BANDS]).fillna(0)
    out["chronic"]["phase_days"] = {k: int(v) for k, v in ph.items()}
    out["chronic"]["reference_year"] = REF_YEAR
    out["chronic"]["note"] = ("발화율=유병률이므로 높게 나오는 것이 정상. "
                              "경보가 아니라 국면 분류 — 기질 혼합비 등 중기 의사결정 입력")

    fan = f["FAN_A_mgL"].dropna()
    out["rejected_v2_rule"] = {
        "rule": "FAN > 300 mg/L 절대임계 (v2 §5 L1, 비순치 문헌임계)",
        "n_measured": int(len(fan)),
        "rate_pct": round(100 * float((fan > 300).mean()), 1),
        "verdict": "경보 기능 없음 → 급성분 A4(상대편차) + 만성분 C1(부하지수)으로 분리",
    }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    f = compute_indicators(d)
    s = summarize(f)

    print("=" * 74)
    print(" 소화조 상태 점검 — 급성 경보 / 만성 상태 분리 (암모니아 저해 명시적 고려)")
    print("=" * 74)
    print(f" 대상 {s['n_days']}일 (2018-01-01 ~ 2023-09-17)")

    print("\n【급성 경보】 발화율이 낮아야 경보로 기능한다")
    print("-" * 74)
    print(f"{'규칙':<38} {'발화일':>7} {'발화율':>8}")
    for k, r in s["acute"].items():
        print(f"{k} {r['label']:<35} {r['days']:>7} {r['rate_pct']:>7.1f}%")
    print("-" * 74)
    for k, v in s["acute_grades"].items():
        print(f"  {k:<5} {v:>5} 일  ({100*v/s['n_days']:>5.1f}%)")

    print("\n【만성 상태】 발화율 = 그 국면의 실제 유병률 (경보 아님)")
    print("-" * 74)
    print(f" C1 암모니아 부하지수 = 알칼리도의 {REF_YEAR}년(Y_COD 최고) 대비 상승률")
    for k, v in s["chronic"]["phase_days"].items():
        print(f"  {k:<8} {v:>5} 일  ({100*v/s['n_days']:>5.1f}%)")

    print("\n" + "-" * 74)
    r = s["rejected_v2_rule"]
    print(f" [폐기] {r['rule']}")
    print(f"   측정 {r['n_measured']}일 중 발화율 {r['rate_pct']}% → {r['verdict']}")

    print("\n" + "-" * 74)
    print(" 연도별 대조 — 만성 부하지수가 성능 저점(2021)을 포착하는가")
    print("-" * 74)
    yr = f.groupby("year").agg(
        FAN중위=("FAN_A_mgL", "median"),
        알칼리도=("ALK_A_mgL", "median"),
        부하지수=("C1_burden_sm", "median"),
        Y_COD=("Y_COD", "median"),
        급성발화율=("n_acute", lambda x: round(100 * (x > 0).mean(), 1)),
    )
    print(yr.round(2).to_string())

    cols = ["date", "year", "Y_COD", "A1_dev_pct", "A2", "A3_z", "FAN_A_mgL",
            "A4_FAN_ratio", "n_acute", "acute_grade", "C1_burden_sm", "chronic_phase"]
    f[cols].to_csv(os.path.join(OUT, "health_check.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "health_check_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장 → {OUT}/health_check.csv, health_check_summary.json")


if __name__ == "__main__":
    main()
