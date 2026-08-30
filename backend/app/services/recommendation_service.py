"""Operator recommendation engine.

BioGuard is a decision-support system, not a controller. Nothing here issues a
physical command: every entry tells an operator what to check, what to hold, and
when to re-judge.

Two kinds of input are combined, and they are kept distinct:

  * PROJECT   -- the health conditions and command gates the project already
                 defines (project_rules.py). These set severity. No threshold is
                 invented, adjusted or restated here.
  * REFERENCE -- operating practice for the response to a condition, once the
                 project has already declared it. Grounded in the EPA AgSTAR
                 Anaerobic Digester/Biogas System Operator Guidebook (pH is a
                 lagging indicator; alkalinity is the buffer to watch; rapid
                 organic-loading change destabilises a digester) and in the
                 ammonia-inhibition literature (free ammonia rises with TAN, pH
                 and temperature together). These add only *actions*, never
                 numbers and never severity.

Trend detection reuses the project's own robust statistic -- the median/MAD
Hampel form already used by dq_service -- purely to describe whether a value has
moved against its own recent behaviour. It is descriptive, never a health rule,
and a trend on its own can never raise an ACTION.
"""

from __future__ import annotations

import math
from typing import Any

from . import project_rules as rules

PROJECT = "PROJECT"
REFERENCE = "ENGINEERING_REFERENCE"

# Same robust form dq_service uses for its Hampel test.
HAMPEL_K = 3 * 1.4826
TREND_WINDOW = 7
PRIORITY_ORDER = {"ACTION": 0, "WATCH": 1, "NORMAL": 2}

# Columns a trend is computed for, and the label an operator sees.
TREND_FIELDS: dict[str, str] = {
    "feed_AB_tpd": "총 투입량",
    "biogas_AB_m3d": "총 바이오가스",
    "CH4_m3d_observed": "메탄생성량",
    "ch4_purity_A_pct": "메탄순도 A",
    "ch4_purity_B_pct": "메탄순도 B",
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def build_trend(history: list[dict[str, Any]], selected_date: str) -> dict[str, Any]:
    """Describe how the selected day sits against its own recent behaviour.

    Uses the project's median/MAD form. `direction` is the plain sign of the move
    against the trailing window; `notable` is True only when the move exceeds the
    same robust envelope dq_service uses. Neither is a health verdict.
    """
    ordered = [row for row in history if isinstance(row.get("date"), str)]
    ordered.sort(key=lambda row: row["date"])
    index = next((position for position, row in enumerate(ordered)
                  if row["date"] == selected_date), None)
    if index is None:
        return {"available": False, "fields": {}}

    current = ordered[index]
    result: dict[str, Any] = {}
    for column, label in TREND_FIELDS.items():
        value = current.get(column)
        window = [row[column] for row in ordered[max(0, index - TREND_WINDOW):index]
                  if _finite(row.get(column))]
        if not _finite(value) or len(window) < 3:
            result[column] = {"label": label, "value": value if _finite(value) else None,
                              "baseline": None, "change": None, "direction": "UNKNOWN",
                              "notable": False}
            continue
        centre = _median(window)
        spread = _median([abs(item - centre) for item in window])
        change = float(value) - centre
        envelope = HAMPEL_K * spread
        result[column] = {
            "label": label,
            "value": float(value),
            "baseline": centre,
            "change": change,
            "change_pct": (change / centre * 100.0) if centre else None,
            "direction": "UP" if change > 0 else "DOWN" if change < 0 else "FLAT",
            # A move larger than the robust envelope of its own recent window.
            # When the window has no spread at all the envelope collapses to
            # zero, which is the limit of the same test: any real move is then
            # outside everything that window has ever shown.
            "notable": bool(abs(change) > envelope
                            if spread > 0
                            else abs(change) > 1e-9 * max(1.0, abs(centre))),
            "method": "TREND_ROBUST_MAD",
        }
    return {"available": True, "fields": result, "window": TREND_WINDOW}


def _reading(diagnosis: dict[str, Any], variable: str) -> dict[str, Any]:
    entry_a = diagnosis["classified"][variable]["A"]
    entry_b = diagnosis["classified"][variable]["B"]
    return {"A": entry_a, "B": entry_b}


def _side_text(diagnosis: dict[str, Any], variable: str, digits: int = 2) -> str:
    sides = _reading(diagnosis, variable)
    def show(entry):
        return f"{entry['value']:.{digits}f}" if _finite(entry["value"]) else "N/A"
    return f"A {show(sides['A'])} · B {show(sides['B'])}"


def _colour_set(diagnosis: dict[str, Any], variable: str) -> set[str]:
    sides = _reading(diagnosis, variable)
    return {sides["A"]["color"], sides["B"]["color"]}


def _tail_hit(diagnosis: dict[str, Any], variable: str, tail: str, colours: set[str]) -> bool:
    """True when either side is off-band in the given direction."""
    sides = _reading(diagnosis, variable)
    return any(entry["color"] in colours and entry["tail"] == tail
               for entry in (sides["A"], sides["B"]))


def _flag_codes(diagnosis: dict[str, Any]) -> set[str]:
    return {flag["code"] for flag in diagnosis["health"]["flags"]}


def _delta_direction(deltas: dict[str, Any] | None, variable: str) -> str:
    if not deltas or variable not in deltas:
        return "UNKNOWN"
    values = [item for item in deltas[variable].values() if _finite(item)]
    if not values:
        return "UNKNOWN"
    if all(item > 0 for item in values):
        return "UP"
    if all(item < 0 for item in values):
        return "DOWN"
    return "MIXED"


def build_recommendations(
    diagnosis: dict[str, Any],
    evidence: dict[str, Any],  # noqa: ARG001 - kept so the payload contract is stable
    trend: dict[str, Any],
    deltas: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Severity always comes from the project. Reference practice only supplies
    the response once the project has already declared the condition."""
    codes = _flag_codes(diagnosis)
    fields = trend.get("fields", {})
    items: list[dict[str, Any]] = []

    def add(priority, code, title, evidence_lines, actions, recheck, source):
        items.append({
            "priority": priority, "code": code, "title": title,
            "evidence": [line for line in evidence_lines if line],
            "actions": actions, "recheck": recheck, "source": source,
        })

    def moved(column, direction):
        item = fields.get(column) or {}
        return item.get("direction") == direction and item.get("notable")

    def trend_line(column):
        item = fields.get(column) or {}
        if not item.get("notable") or not _finite(item.get("change_pct")):
            return None
        arrow = "상승" if item["direction"] == "UP" else "하락"
        return f"{item['label']} 최근 {trend.get('window', TREND_WINDOW)}일 대비 {item['change_pct']:+.1f}% {arrow}"

    vfa_rising = _delta_direction(deltas, "VFA") == "UP"
    ratio_rising = _delta_direction(deltas, "VFA_TA") == "UP"

    # ---- acid / buffer -----------------------------------------------------
    if {"ACID_SEVERE", "ACID_EARLY"} & codes:
        severe = "ACID_SEVERE" in codes
        factor = rules.ACID["severeFeedFactor"] if severe else rules.ACID["earlyFeedFactor"]
        add("ACTION", "REC-ACID-002",
            "산 축적 — 공정 조치 필요",
            [f"VFA/ALK {_side_text(diagnosis, 'VFA_TA', 3)}",
             "산 생성이 메탄 생성을 앞지르고 있을 가능성",
             trend_line("CH4_m3d_observed")],
            ["신규 Feed 증가를 중단하고 현재 부하를 재평가하십시오.",
             f"Project 명령에 따라 Feed_target = {factor:g} × Q_ref 를 적용하십시오.",
             "VFA, ALK, pH를 우선 재측정하십시오.",
             "유입 원료 조성 변화 또는 고부하 원료 투입 여부를 확인하십시오.",
             "안정화 전까지 추가 고부하 투입을 피하십시오."],
            "재측정에서도 VFA/ALK가 하락하지 않으면 현장 SOP의 상위 대응 절차로 전환하십시오.",
            [PROJECT, REFERENCE])
    elif "ACID_WATCH" in codes:
        add("WATCH", "REC-ACID-001",
            "산발효 증가 징후",
            [f"VFA/ALK {_side_text(diagnosis, 'VFA_TA', 3)}",
             "VFA 상승 추세" if vfa_rising else None,
             trend_line("feed_AB_tpd")],
            ["추가 Feed 증가를 중지하고 현재 Feed를 유지하십시오.",
             "최근 24~48시간 Feed / 유기부하 증가 이력을 확인하십시오.",
             "VFA와 ALK를 재측정하십시오.",
             "pH 추세가 하락하는지 확인하십시오.",
             "Temperature와 Mixing 상태를 함께 확인하십시오."],
            "다음 측정에서 VFA/ALK가 계속 상승하면 ACTION 단계로 전환하십시오.",
            [PROJECT, REFERENCE])

    if "BUFFER_DEPLETION_CANDIDATE" in codes:
        add("WATCH", "REC-ALK-001",
            "완충능 저하 가능",
            [f"ALK {_side_text(diagnosis, 'ALK', 0)}",
             f"pH {_side_text(diagnosis, 'pH', 2)}"],
            ["ALK를 재측정하여 분석 오류를 배제하십시오.",
             "VFA/ALK와 pH를 동시에 확인하십시오.",
             "최근 유기부하 증가 여부를 확인하십시오.",
             "추가 Feed 증가는 중지하십시오.",
             "Project의 alkali gate 충족 여부를 확인하십시오. 투입량 계산식은 Project에 정의되어 있지 않습니다."],
            "다음 분석에서 ALK 회복과 VFA/ALK 하락을 함께 확인하십시오.",
            [PROJECT, REFERENCE])

    # ---- thermal -----------------------------------------------------------
    if "THERMAL_ACTION_CANDIDATE" in codes:
        low = _tail_hit(diagnosis, "temperature", "LOW", {"RED", "ORANGE"}) or all(
            _finite(e["value"]) and e["value"] < rules.THERMAL["lowC"]
            for e in _reading(diagnosis, "temperature").values())
        actions = ["A/B 온도센서를 교차 확인하여 계측 이상 여부를 먼저 배제하십시오."]
        actions += (["가열 계통과 순환 펌프의 운전 상태를 즉시 점검하십시오.",
                     "온도 복구 전까지 급격한 Feed 변경을 피하십시오."] if low else
                    ["가열 계통 과열 또는 제어 이상 여부를 점검하십시오.",
                     "불필요한 추가 가열을 중지할 수 있는지 SOP를 확인하십시오."])
        actions += ["Mixing / 순환 상태를 확인하십시오.",
                    "승인된 SOP에 따라 Project 정상 온도대역으로 복구하십시오."]
        add("ACTION", "REC-TEMP-001",
            "소화조 온도 이상 — 공정 조치 필요",
            [f"온도 {_side_text(diagnosis, 'temperature', 2)}",
             f"Project 정상 대역 {rules.THERMAL['lowC']:g}~{rules.THERMAL['highC']:g} ℃ 이탈",
             trend_line("biogas_AB_m3d")],
            actions,
            "온도 복구 후 Biogas와 VFA/ALK 추세가 회복되는지 확인하십시오.",
            [PROJECT, REFERENCE])
    elif {"THERMAL_ONE_SIDED_ABNORMAL", "THERMAL_ONE_SIDED_INVALID"} & codes:
        add("WATCH", "REC-TEMP-003",
            "온도 한쪽 계통 이상",
            [f"온도 {_side_text(diagnosis, 'temperature', 2)}"],
            ["T-A / T-B 센서를 교차 검증하십시오.",
             "해당 계통의 가열·순환 상태를 점검하십시오.",
             "Project 기준상 국부 점검 대상이며 전역 부하 조정은 하지 않습니다."],
            "센서 확인 후에도 한쪽만 이탈이 지속되면 계측 계통을 우선 정비하십시오.",
            [PROJECT])

    # ---- ammonia -----------------------------------------------------------
    if {"AMMONIA_WATCH", "AMMONIA_ACTION_CANDIDATE"} & codes or _colour_set(diagnosis, "TAN") & {"ORANGE", "RED"}:
        high_ph = _tail_hit(diagnosis, "pH", "HIGH", {"ORANGE", "RED"})
        high_temp = _tail_hit(diagnosis, "temperature", "HIGH", {"ORANGE", "RED"})
        priority = "WATCH"
        title = "암모니아 부하 확인"
        lines = [f"TAN {_side_text(diagnosis, 'TAN', 0)}"]
        if high_ph or high_temp:
            title = "암모니아 저해 가능 — 우선 확인"
            lines.append("pH 또는 온도가 동시에 상승 (유리암모니아 비율 상승 조건)")
        add(priority, "REC-TAN-001", title, lines,
            ["TAN 직접 측정값과 측정일을 확인하십시오.",
             "최근 고질소·단백질성 원료 투입 변화를 확인하십시오.",
             "pH와 Temperature를 함께 확인하십시오.",
             "추가 고질소 Feedstock 투입을 보류하십시오.",
             "현장 ammonia inhibition 대응 SOP를 확인하십시오."],
            "신규 TAN 측정값으로 재판정하십시오. Project에 FAN 계산식이 없어 유리암모니아는 산출하지 않습니다.",
            [PROJECT, REFERENCE])

    # ---- solids / COD ------------------------------------------------------
    if _tail_hit(diagnosis, "TS", "HIGH", {"ORANGE", "RED"}):
        add("WATCH", "REC-TS-001", "고형물 농도 상승",
            [f"TS {_side_text(diagnosis, 'TS', 2)}"],
            ["Feed 고형물 농도 변화 여부를 확인하십시오.",
             "Mixing 상태를 점검하십시오.",
             "Pump / 배관 폐색 징후를 확인하십시오."],
            "다음 분석에서 TS가 정상 대역으로 복귀하는지 확인하십시오.",
            [PROJECT, REFERENCE])
    if _tail_hit(diagnosis, "CODcr", "HIGH", {"ORANGE", "RED"}) or _tail_hit(diagnosis, "VS", "HIGH", {"ORANGE", "RED"}):
        add("WATCH", "REC-LOAD-001", "유기부하 상승 가능",
            [f"CODcr {_side_text(diagnosis, 'CODcr', 0)}",
             f"VS {_side_text(diagnosis, 'VS', 2)}",
             trend_line("feed_AB_tpd")],
            ["최근 유입 유기부하 증가 여부를 확인하십시오.",
             "원료 조성 변화를 확인하십시오.",
             "VFA 상승이 동반되는지 확인하십시오.",
             "Biogas와 CH4가 부하 증가에 정상적으로 반응하는지 확인하십시오."],
            "부하 증가가 확인되면 안정화까지 추가 증량을 보류하십시오.",
            [PROJECT, REFERENCE])

    # ---- multi-variable patterns (observational; never raise ACTION alone) --
    ph_falling = _tail_hit(diagnosis, "pH", "LOW", {"ORANGE", "RED"})
    alk_low = _tail_hit(diagnosis, "ALK", "LOW", {"ORANGE", "RED"})
    if (ratio_rising or "ACID_WATCH" in codes) and alk_low and ph_falling:
        add("WATCH", "REC-PATTERN-001", "산 축적 및 완충능 저하 동반",
            [f"VFA/ALK {_side_text(diagnosis, 'VFA_TA', 3)}",
             f"ALK {_side_text(diagnosis, 'ALK', 0)} · pH {_side_text(diagnosis, 'pH', 2)}"],
            ["추가 Feed 증가를 중지하십시오.",
             "VFA / ALK / pH를 함께 재측정하십시오.",
             "최근 유기부하 변화를 확인하십시오.",
             "안정화 전까지 고부하 원료 추가 투입을 피하십시오."],
            "세 지표가 동시에 개선되는지 다음 분석에서 확인하십시오.",
            [PROJECT, REFERENCE])

    purity_down = moved("ch4_purity_A_pct", "DOWN") or moved("ch4_purity_B_pct", "DOWN")
    if (vfa_rising or "ACID_WATCH" in codes) and purity_down and moved("CH4_m3d_observed", "DOWN"):
        add("WATCH", "REC-PATTERN-002", "소화 성능 저하 가능성",
            [trend_line("CH4_m3d_observed"), trend_line("ch4_purity_A_pct"),
             f"VFA/ALK {_side_text(diagnosis, 'VFA_TA', 3)}"],
            ["최근 Feed 변경 이력을 확인하십시오.",
             "VFA / ALK / pH를 확인하십시오.",
             "Temperature와 Gas analyzer 상태를 확인하십시오.",
             "추가 부하 증가는 보류하십시오."],
            "다음 측정에서 순도와 생성량이 회복되는지 확인하십시오.",
            [PROJECT, REFERENCE])

    thermal_off = bool({"THERMAL_ACTION_CANDIDATE", "THERMAL_ONE_SIDED_ABNORMAL"} & codes)
    if thermal_off and moved("biogas_AB_m3d", "DOWN"):
        add("WATCH", "REC-PATTERN-003", "온도 조건에 따른 반응 저하 가능",
            [f"온도 {_side_text(diagnosis, 'temperature', 2)}", trend_line("biogas_AB_m3d")],
            ["온도 계측을 재확인하십시오.",
             "가열·순환 계통 상태를 확인하십시오.",
             "온도 복구 전까지 Feed 급변을 피하십시오."],
            "온도 복구 후 Biogas 추세가 회복되는지 확인하십시오.",
            [PROJECT, REFERENCE])

    # ---- observational gas / feed trends (no project condition required) ----
    if not any(item["priority"] == "ACTION" for item in items):
        if moved("feed_AB_tpd", "UP") and (vfa_rising or moved("biogas_AB_m3d", "DOWN")):
            add("WATCH", "REC-FEED-001", "투입량 급증 후 공정 반응 확인",
                [trend_line("feed_AB_tpd"), trend_line("biogas_AB_m3d")],
                ["추가 Feed 증가를 중지하십시오.",
                 "Feed 변경 시점과 이상 발생 시점을 비교하십시오.",
                 "VFA/ALK와 pH를 재확인하십시오."],
                "안정화가 확인될 때까지 부하를 유지하십시오.",
                [PROJECT, REFERENCE])
        elif moved("feed_AB_tpd", "DOWN") and moved("biogas_AB_m3d", "DOWN"):
            add("WATCH", "REC-FEED-002", "투입량 감소와 가스 생산 동반 감소",
                [trend_line("feed_AB_tpd"), trend_line("biogas_AB_m3d")],
                ["Feed 공급 중단 여부를 확인하십시오.",
                 "Pump / feeder 상태를 점검하십시오.",
                 "의도된 감량인지 운전 기록을 확인하십시오.",
                 "복구 시 급격한 부하 증가를 피하십시오."],
                "Feed 복구 후 Biogas가 회복되는지 확인하십시오.",
                [PROJECT, REFERENCE])
        elif moved("biogas_AB_m3d", "DOWN"):
            add("WATCH", "REC-GAS-001", "바이오가스 생산량 감소",
                [trend_line("biogas_AB_m3d")],
                ["Feed 투입량과 공급 중단 여부를 확인하십시오.",
                 "Gas flow meter 상태를 확인하십시오.",
                 "Mixing과 Temperature를 확인하십시오.",
                 "VFA/ALK와 pH를 확인하십시오."],
                "SOP에 따라 gas handling 계통 누설 여부도 함께 점검하십시오.",
                [PROJECT, REFERENCE])
        elif purity_down:
            add("WATCH", "REC-GAS-002", "메탄순도 감소",
                [trend_line("ch4_purity_A_pct") or trend_line("ch4_purity_B_pct")],
                ["Gas analyzer 교정 상태를 확인하십시오.",
                 "VFA/ALK, pH, Temperature를 확인하십시오.",
                 "최근 유기부하 급변 여부를 확인하십시오."],
                "교정 확인 후에도 순도 감소가 지속되면 공정 지표를 함께 재판정하십시오.",
                [PROJECT, REFERENCE])

    # A degraded channel is a measurement problem, not an operating instruction,
    # so it produces no card here. The data-quality state still reaches the API
    # through health.evidence and health.data_confidence for anything that needs
    # it; it simply is not something to hand an operator as an action.

    if not items:
        add("NORMAL", "REC-NORMAL-001", "현재 운전조건 유지",
            ["주요 공정지표가 Project 정상범위 내에 있습니다."],
            ["현재 Feed와 운전조건을 유지하십시오.",
             "주요 계측 추세를 지속 확인하십시오."],
            "다음 정기 분석에서 재확인하십시오.",
            [PROJECT])

    items.sort(key=lambda item: PRIORITY_ORDER[item["priority"]])
    return items
