"""
소화조 내부 건강상태 점검 프로그램 — PROMPT §6 / §12 산출물 4.

예측력이 없어 예측 모델에서 배제된 변수를 폐기하지 않고 여기로 재배치한다.
메탄 예측 성능과 무관하게 독립적으로 평가·보고한다.

산출: 일별 지표 + 이동추세 + 경보 플래그 + 이상구간 타임라인.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEALTH_THRESHOLDS as TH

# 지표 정의: (열, 표시명, 진단 의미, 판정 함수)
def _flag_vfa_alk(v):
    return np.where(np.isnan(v), "", np.where(v >= TH["VFA_ALK"]["crit"], "crit",
                    np.where(v >= TH["VFA_ALK"]["warn"], "warn", "ok")))


def _flag_ph(v):
    return np.where(np.isnan(v), "",
                    np.where((v < TH["pH"]["lo"]) | (v > TH["pH"]["hi"]), "crit", "ok"))


def _flag_olr(v):
    return np.where(np.isnan(v), "",
                    np.where(v > TH["OLR"]["hi"], "crit",
                    np.where(v < TH["OLR"]["lo"], "warn", "ok")))


def _flag_fan(v):
    return np.where(np.isnan(v), "", np.where(v >= TH["FAN"]["crit"], "crit",
                    np.where(v >= TH["FAN"]["warn"], "warn", "ok")))


def _flag_temp(v):
    d = np.abs(v - TH["TEMP"]["set"])
    return np.where(np.isnan(v), "", np.where(d > TH["TEMP"]["tol"], "warn", "ok"))


INDICATORS = [
    ("VFA_ALK_A", "VFA/알칼리도", "산·알칼리도 균형, 과부하 조기경보",
     f"주의 ≥{TH['VFA_ALK']['warn']} / 위험 ≥{TH['VFA_ALK']['crit']}", _flag_vfa_alk),
    ("VFA_A", "VFA 절대값", "산 축적 — 급상승 추세 자체가 신호", "추세 기준(고정임계 없음)", None),
    ("FAN_A", "유리암모니아 FAN", "암모니아 저해",
     f"주의 ≥{TH['FAN']['warn']} / 위험 ≥{TH['FAN']['crit']} mg/L", _flag_fan),
    ("NH3N_A", "NH₃-N", "질소 부하", "FAN 과 병행 해석", None),
    ("dig_pH_A", "소화조 pH", "완충 붕괴",
     f"{TH['pH']['lo']}~{TH['pH']['hi']} 이탈 시 위험", _flag_ph),
    ("dig_T_A", "소화조 온도", "중온 안정성",
     f"{TH['TEMP']['set']}±{TH['TEMP']['tol']}℃ 이탈 (센서 결함 플래그 병행)", _flag_temp),
    ("OLR", "유기물 부하율", "부하 과소/과대",
     f"지침 {TH['OLR']['lo']}~{TH['OLR']['hi']} kgVS/㎥·d", _flag_olr),
]


REL_WARN, REL_CRIT = 0.15, 0.25       # 시설 기준선(90일 중앙값) 대비 상대편차


def _relative_flag(v: np.ndarray) -> np.ndarray:
    """
    시설 자체 기준선 대비 상대편차 경보.

    문헌 절대임계를 이 시설에 그대로 적용하면 경보가 상시 발화한다(아래 표의 문헌 발화율
    참조 — pH 98%, OLR 86%). 저장소 문서가 `FAN>300`(75% 발화)에서 이미 지적한 실패이며,
    원인은 같다: **비순치·일반 설계 기준을 순치된 특정 시설에 적용**한 것.
    실제 경보는 시설의 90일 기준선 대비 이탈로 낸다.
    """
    s = pd.Series(v)
    base = s.rolling(90, min_periods=30).median().shift(1)   # 과거만 사용
    dev = (s - base).abs() / base.abs()
    out = np.where(~np.isfinite(dev.to_numpy()), "",
                   np.where(dev.to_numpy() >= REL_CRIT, "crit",
                            np.where(dev.to_numpy() >= REL_WARN, "warn", "ok")))
    return out


def build(F: pd.DataFrame) -> dict:
    H = pd.DataFrame({"date": F.date})
    summary, series = [], {"dates": F.date.dt.strftime("%Y-%m-%d").tolist()}

    for col, name, meaning, rule, fn in INDICATORS:
        v = F[col].to_numpy(float)
        trend = pd.Series(v).rolling(30, min_periods=5).mean()
        series[col] = [None if not np.isfinite(x) else round(float(x), 4) for x in v]
        series[f"{col}_ma30"] = [None if not np.isfinite(x) else round(float(x), 4) for x in trend]
        rec = {"열": col, "지표": name, "진단의미": meaning, "문헌기준": rule,
               "관측일": int(np.isfinite(v).sum()),
               "결측률_pct": round(100 * float(np.isnan(v).mean()), 1),
               "중앙값": None if not np.isfinite(np.nanmedian(v)) else round(float(np.nanmedian(v)), 3)}

        # ① 문헌 절대임계 — 진단(설계·문헌 대비 위치) 용도로만 보고
        if fn is not None:
            fl = fn(v)
            obs = max(int((fl != "").sum()), 1)
            rec["문헌_주의_pct"] = round(100 * float((fl == "warn").sum() / obs), 1)
            rec["문헌_위험_pct"] = round(100 * float((fl == "crit").sum() / obs), 1)
        else:
            rec["문헌_주의_pct"] = rec["문헌_위험_pct"] = None

        # ② 시설 기준선 대비 상대편차 — 실제 경보
        rf = _relative_flag(v)
        H[col] = rf
        obs = max(int((rf != "").sum()), 1)
        rec["상대_주의_pct"] = round(100 * float((rf == "warn").sum() / obs), 1)
        rec["상대_위험_pct"] = round(100 * float((rf == "crit").sum() / obs), 1)
        summary.append(rec)

    # 종합 판정: 상대편차 기준. 위험 2개 이상 → crit, 위험 1개 또는 주의 2개 이상 → warn
    cols = [c for c in H.columns if c != "date"]
    crit = (H[cols] == "crit").sum(axis=1)
    warn = (H[cols] == "warn").sum(axis=1)
    verdict = np.where(crit >= 2, "crit", np.where((crit >= 1) | (warn >= 2), "warn", "ok"))
    verdict = np.where((H[cols] != "").sum(axis=1) == 0, "", verdict)   # 전 지표 미측정일
    H["verdict"] = verdict
    series["verdict"] = list(verdict)

    return {"지표": summary, "series": series,
            "경보설계": {
                "채택": "시설 기준선(90일 중앙값, 과거만) 대비 상대편차 — 주의 ≥15%, 위험 ≥25%",
                "문헌임계_용도": "경보가 아니라 진단. 설계·문헌 대비 이 시설이 어디에 있는지 보여준다.",
                "근거": "문헌 절대임계를 그대로 쓰면 pH 98%·OLR 86%·FAN 75% 발화로 경보 기능을 "
                      "상실한다. 이 시설은 pH 7.95·OLR 1.14 로 6년간 안정 운전한 순치된 계이며, "
                      "일반 설계 기준의 밖에 있는 것이 곧 이상은 아니다.",
                "종합규칙": "위험 2개 이상 → 위험 / 위험 1개 또는 주의 2개 이상 → 주의",
            },
            "종합": {"위험일": int((verdict == "crit").sum()),
                   "주의일": int((verdict == "warn").sum()),
                   "정상일": int((verdict == "ok").sum()),
                   "미측정일": int((verdict == "").sum())},
            "이상구간": episodes(F.date, verdict),
            "주의사항": [
                "FAN>300 은 측정일의 75%에서 발화한다 — 이 시설은 순치된 계이고 해당 임계는 "
                "비순치 기준이다. 절대임계 경보로 쓰지 말고 추세·2차 판별로만 사용한다.",
                "VFA/알칼리도는 이 시설에서 함정 지표다. 암모니아가 분모(알칼리도)를 부풀려 "
                "성능이 하락하는 동안 지표는 '개선'으로 표시된다(2022년 지표 최저 = 메탄 최저).",
                "소화조 온도는 센서 표류·고착 구간이 있다(DQ-02/03). 결함 플래그와 함께 읽어야 한다.",
            ]}


def episodes(dates: pd.Series, verdict: np.ndarray, min_len: int = 3) -> list[dict]:
    """연속 이상구간 타임라인."""
    out, start, cur = [], None, None
    for i, v in enumerate(verdict):
        if v in ("warn", "crit"):
            if start is None:
                start, cur = i, v
            elif v == "crit":
                cur = "crit"
        else:
            if start is not None and i - start >= min_len:
                out.append({"시작": dates.iloc[start].strftime("%Y-%m-%d"),
                            "종료": dates.iloc[i - 1].strftime("%Y-%m-%d"),
                            "일수": i - start, "수준": cur})
            start, cur = None, None
    if start is not None and len(verdict) - start >= min_len:
        out.append({"시작": dates.iloc[start].strftime("%Y-%m-%d"),
                    "종료": dates.iloc[len(verdict) - 1].strftime("%Y-%m-%d"),
                    "일수": len(verdict) - start, "수준": cur})
    out.sort(key=lambda e: -e["일수"])
    return out[:25]
