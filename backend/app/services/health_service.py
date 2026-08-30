"""Adapter between an uploaded BioGuard dataset and the project health engine.

This module holds NO thresholds. Every band, gate, concordance limit and command
factor lives in project_rules.py, which is a verbatim migration of the project's
own specification. The job here is only to:

  * map the BIOGUARD_INPUT schema onto the project's ten variables,
  * convert DQ verdicts into the project's valid/invalid input contract,
  * present the engine's output for the console.
"""

from __future__ import annotations

import math
from typing import Any

from . import project_rules as rules
from .project_rules import RULE_SOURCE, compute_diagnosis


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


# --- schema mapping ---------------------------------------------------------
# project variable -> (A column, B column). VFA_TA is the derived ratio the
# project itself defines as VFA/ALK; FAN has no formula anywhere in the project.
FIELD_MAP: dict[str, tuple[str, str]] = {
    "temperature": ("temperature_A_C", "temperature_B_C"),
    "pH": ("pH_A", "pH_B"),
    "VFA": ("VFA_A_mgL", "VFA_B_mgL"),
    "ALK": ("ALK_A_mgL", "ALK_B_mgL"),
    "VFA_TA": ("VFA_TA_A", "VFA_TA_B"),
    "TAN": ("TAN_A_mgL", "TAN_B_mgL"),
    "TS": ("TS_A_pct", "TS_B_pct"),
    "VS": ("VS_A_pct", "VS_B_pct"),
    "CODcr": ("CODcr_A_mgL", "CODcr_B_mgL"),
}
# A derived variable inherits the DQ verdict of the fields it is computed from.
DQ_SOURCE_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "VFA_TA": (("VFA_A_mgL", "ALK_A_mgL"), ("VFA_B_mgL", "ALK_B_mgL")),
}

# Project display labels and status words, kept verbatim. The "(통계)" suffix is
# the project's own wording and matters: specification section 2 states that a
# single variable's colour is a statistical rarity signal, not a process action.
VARIABLE_LABELS: dict[str, str] = {
    "temperature": "혐기성 소화조 내부 온도", "pH": "소화조 내부 pH",
    "VFA": "휘발성지방산(VFA)", "ALK": "알칼리도(ALK)", "VFA_TA": "VFA / Alk 비율",
    "TAN": "총암모니아성질소(TAN)", "FAN": "유리암모니아(FAN)",
    "TS": "총고형물(TS)", "VS": "휘발성고형물(VS)", "CODcr": "화학적산소요구량(CODcr)",
}
PROJECT_STATUS_WORD = {"GREEN": "정상", "ORANGE": "주의(통계)", "RED": "이상(통계)", "DATA": "데이터확인"}
COLOR_TO_STATUS = {"GREEN": "NORMAL", "ORANGE": "WATCH", "RED": "ACTION", "DATA": "DATA"}
COLOR_TO_LAMP = {"GREEN": "GREEN", "ORANGE": "AMBER", "RED": "RED", "DATA": "GRAY"}

STATE_TEXT = {
    "NORMAL": "정상", "NORMAL_DEGRADED_DATA": "주의", "DATA_SUSPECT": "주의",
    "PROCESS_WATCH": "주의", "PROCESS_ACTION": "조치필요", "RECOVERY": "주의",
}

# Variables the project explicitly records as informational: CH4 purity, feed and
# biogas are "not one of the 10 HEALTH_RULES variables (no threshold defined for
# it anywhere)" and are "never health-colored" (EXCEL_INFORMATIONAL in the
# shipped HTML). They therefore report data state only, never a process signal.
INFORMATIONAL_SPECS: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, str] | None]] = [
    ("ch4_purity", "메탄가스 순도", "%", "MEAN",
     ("ch4_purity_A_pct", "ch4_purity_B_pct"), ("ch4_purity_A_pct", "ch4_purity_B_pct")),
    ("biogas_total", "총 바이오가스 발생량", "m³/d", "DIRECT",
     ("biogas_AB_m3d",), ("biogas_A_m3d", "biogas_B_m3d")),
    ("ch4_observed", "실제 메탄생성량", "m³/d", "DIRECT",
     ("CH4_m3d_observed",), None),
]
INFORMATIONAL_DQ_FIELDS: dict[str, tuple[str, ...]] = {
    "ch4_purity": ("ch4_purity_A_pct", "ch4_purity_B_pct"),
    "biogas_total": ("biogas_A_m3d", "biogas_B_m3d"),
    "ch4_observed": ("biogas_A_m3d", "biogas_B_m3d", "ch4_purity_A_pct", "ch4_purity_B_pct"),
}
INFORMATIONAL_STATUS = {
    "DATA_NORMAL": ("GRAY", "데이터정상"),
    "DATA_WATCH": ("GRAY", "데이터점검"),
    "DATA": ("GRAY", "데이터없음"),
}
UNUSABLE_DQ_PREFIXES = ("MISSING", "INVALID", "STALE")


def _side_input(value: Any, dq_status: str) -> dict[str, Any]:
    """Convert a DQ verdict into the project's valid/invalid input contract.

    Specification section 3 withholds "corrupted, faulted, or stale" readings from
    the numeric bands -- that is MISSING / INVALID / STALE. A statistical SUSPECT
    is a different thing: run_data_quality keeps those values in the clean frame
    on purpose ("SUSPECT values remain clean/model-usable", suspect_values_retained),
    so they are classified and carried with a dq_status of SUSPECT for the operator
    rather than being blanked out of the matrix.
    """
    if dq_status.startswith(UNUSABLE_DQ_PREFIXES):
        return {"value": None, "valid": False, "reason": dq_status}
    if not _finite(value):
        return {"value": None, "valid": False, "reason": "MISSING"}
    suspect = dq_status.startswith("SUSPECT")
    return {"value": float(value), "valid": True, "sampleAgeDays": 0, "dq_suspect": suspect}


def _dq_for(variable: str, side_index: int, dq_statuses: dict[str, str]) -> str:
    """Worst verdict across the fields a variable is built from: unusable wins
    over suspect, suspect wins over valid."""
    sources = DQ_SOURCE_FIELDS.get(variable)
    fields = sources[side_index] if sources else (FIELD_MAP[variable][side_index],)
    worst = "VALID"
    for field in fields:
        status = dq_statuses.get(field, "MISSING")
        if status.startswith(UNUSABLE_DQ_PREFIXES):
            return status
        if status != "VALID":
            worst = status
    return worst


def build_raw_variables(values: dict[str, Any], dq_statuses: dict[str, str]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for variable, (column_a, column_b) in FIELD_MAP.items():
        raw[variable] = {
            "A": _side_input(values.get(column_a), _dq_for(variable, 0, dq_statuses)),
            "B": _side_input(values.get(column_b), _dq_for(variable, 1, dq_statuses)),
        }
    raw["FAN"] = {
        "A": {"value": None, "valid": False, "reason": rules.FAN_UNAVAILABLE_REASON},
        "B": {"value": None, "valid": False, "reason": rules.FAN_UNAVAILABLE_REASON},
    }
    return raw


def build_deltas(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, dict[str, float | None]] | None:
    """Change against the previous *direct measurement*, as ACID EARLY requires.
    Returns None when there is no previous direct measurement, so the rule fails
    safe rather than firing on an assumption."""
    if not previous:
        return None
    deltas: dict[str, dict[str, float | None]] = {}
    for variable in ("VFA_TA", "VFA"):
        column_a, column_b = FIELD_MAP[variable]
        side: dict[str, float | None] = {}
        for key, column in (("A", column_a), ("B", column_b)):
            now, before = current.get(column), previous.get(column)
            side[key] = float(now) - float(before) if _finite(now) and _finite(before) else None
        deltas[variable] = side
    return deltas


def diagnose(
    values: dict[str, Any],
    dq_statuses: dict[str, str],
    deltas: dict[str, dict[str, float | None]] | None = None,
    previous_state: str | None = None,
) -> dict[str, Any]:
    return compute_diagnosis(build_raw_variables(values, dq_statuses), deltas, previous_state)


def evaluate_health(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """The signal tower: the engine's final state, never a single-variable copy."""
    health = diagnosis["health"]
    state = health["state"]
    return {
        # The tower shows process severity. Every project state keeps the colour
        # the specification's section 5 table gives it, with exactly one
        # exception: NORMAL_DEGRADED_DATA. That state exists to say "no
        # process-action evidence, data confidence below HIGH", so escalating the
        # process lamp for it would report a process problem that the engine
        # explicitly did not find. RECOVERY and DATA_SUSPECT stay AMBER because
        # the engine asks for operator verification in both.
        "lamp": "GREEN" if state == "NORMAL_DEGRADED_DATA" else rules.STATE_TO_LAMP[state],
        "combined_lamp": rules.STATE_TO_LAMP[state],
        "data_review": diagnosis["data_confidence"] != "HIGH",
        "state": state,
        "state_text": STATE_TEXT[state],
        "code": health["code"],
        "reasons": health["notes"] or [health["code"]],
        "flags": health["flags"],
        "data_confidence": diagnosis["data_confidence"],
        # Process health and data confidence are reported separately: the state
        # name alone does not tell an operator which of the two is degraded.
        "process_state": "PROCESS_ACTION" if state == "PROCESS_ACTION"
        else "PROCESS_WATCH" if state == "PROCESS_WATCH"
        else "DATA_SUSPECT" if state == "DATA_SUSPECT" else "NORMAL",
        "process_text": "조치필요" if state == "PROCESS_ACTION"
        else "주의" if state in ("PROCESS_WATCH", "DATA_SUSPECT") else "정상",
        "rule_source": RULE_SOURCE,
        "rule_id": "evaluateProcessHealth",
        "policy": dict(rules.STATE_TO_LAMP),
    }


# Which project variables back each condition the engine can raise, and the
# project condition each one is being read against. Both come from the rules
# registry -- no threshold is restated here.
FLAG_VARIABLES: dict[str, tuple[str, ...]] = {
    "ACID_SEVERE": ("VFA_TA",),
    "ACID_EARLY": ("VFA_TA", "VFA"),
    "ACID_WATCH": ("VFA_TA",),
    "BUFFER_DEPLETION_CANDIDATE": ("ALK", "pH", "VFA_TA"),
    "AMMONIA_WATCH": ("FAN", "VFA_TA"),
    "AMMONIA_ACTION_CANDIDATE": ("FAN", "VFA_TA"),
    "THERMAL_ACTION_CANDIDATE": ("temperature",),
    "THERMAL_ONE_SIDED_ABNORMAL": ("temperature",),
    "THERMAL_ONE_SIDED_INVALID": ("temperature",),
}


def _condition_text(code: str, variable: str) -> str:
    acid, buffer_, ammonia, thermal = rules.ACID, rules.BUFFER, rules.AMMONIA, rules.THERMAL
    if code == "ACID_SEVERE":
        return f"A/B VFA/TA > {acid['severeVFA_TA']:g}"
    if code == "ACID_EARLY":
        if variable == "VFA":
            return f"A/B ΔVFA >= {acid['earlyDeltaVFA']:g} mg/L"
        return (f"A/B VFA/TA > P95 {acid['p95VFA_TA']:g} 이고 "
                f"ΔVFA/TA >= {acid['earlyDeltaVFA_TA']:g}")
    if code == "ACID_WATCH":
        return f"A/B VFA/TA >= {acid['watchVFA_TA']:g}"
    if code == "BUFFER_DEPLETION_CANDIDATE":
        return {"ALK": f"A/B ALK <= {buffer_['alkLow']:g} mg/L",
                "pH": f"pH <= {buffer_['phLow']:g}",
                "VFA_TA": f"VFA/TA > {buffer_['vfaTaTrigger']:g}"}[variable]
    if code == "AMMONIA_WATCH":
        return (f"FAN {ammonia['watchFanLow']:g}~{ammonia['watchFanHigh']:g}"
                if variable == "FAN" else f"VFA/TA > {ammonia['vfaTaTrigger']:g}")
    if code == "AMMONIA_ACTION_CANDIDATE":
        return (f"A/B FAN > {ammonia['actionFan']:g}"
                if variable == "FAN" else f"VFA/TA > {ammonia['vfaTaTrigger']:g}")
    if code.startswith("THERMAL"):
        return f"A/B 유효 온도 {thermal['lowC']:g}~{thermal['highC']:g} ℃ 이내"
    if code.endswith("_ONE_SIDED_RISK_RED"):
        return "한쪽만 위험방향 RED, 반대쪽 정상"
    return ""


def _variable_reading(diagnosis: dict[str, Any], variable: str, code: str) -> dict[str, Any]:
    entry_a = diagnosis["classified"][variable]["A"]
    entry_b = diagnosis["classified"][variable]["B"]
    threshold = rules.THRESHOLDS[variable]
    return {
        "variable": variable,
        "label": VARIABLE_LABELS[variable],
        "A": entry_a["value"], "B": entry_b["value"],
        "unit": threshold["unit"],
        "green_range": [threshold["lowGreen"], threshold["highGreen"]],
        "A_color": entry_a["color"], "B_color": entry_b["color"],
        "condition": _condition_text(code, variable),
        "rule_id": variable,
        "rule_source": RULE_SOURCE,
    }


def build_evidence(diagnosis: dict[str, Any], dq_statuses: dict[str, str]) -> dict[str, Any]:
    """Process evidence and data-quality evidence, kept strictly apart.

    A process condition and a degraded channel are different problems with
    different responses, so they are never merged into one reason string.
    """
    health = diagnosis["health"]
    process: list[dict[str, Any]] = []
    for flag in health["flags"]:
        code = flag["code"]
        variables = FLAG_VARIABLES.get(code)
        if variables is None and code.endswith("_ONE_SIDED_RISK_RED"):
            variables = (code.removesuffix("_ONE_SIDED_RISK_RED"),)
        if not variables:
            continue
        process.append({
            "code": code,
            "level": flag["level"],
            "note": flag.get("note", ""),
            "readings": [_variable_reading(diagnosis, variable, code)
                         for variable in variables if variable in rules.THRESHOLDS],
        })

    data: list[dict[str, Any]] = []
    for variable in rules.VARIABLE_IDS:
        sides = []
        for key, index in (("A", 0), ("B", 1)):
            entry = diagnosis["classified"][variable][key]
            if entry["status"] not in ("INVALID", "STALE"):
                continue
            reason = entry.get("reason") or "INVALID"
            sides.append({"side": key, "reason": reason})
        if sides:
            data.append({
                "variable": variable,
                "label": VARIABLE_LABELS[variable],
                "sides": sides,
                "rule_source": RULE_SOURCE,
            })
    suspect = sorted({
        field for field, status in dq_statuses.items() if status.startswith("SUSPECT")
    })
    return {
        "process": process,
        "data_quality": data,
        "suspect_fields": suspect,
        "data_confidence": diagnosis["data_confidence"],
        "process_normal": not process,
    }


def _row_colour(side_a: dict[str, Any], side_b: dict[str, Any]) -> str:
    """The project's own per-variable row rollup (worstOf in the shipped HTML)."""
    order = {"RED": 3, "ORANGE": 2, "DATA": 1, "GREEN": 0}
    colour_a = "DATA" if side_a["status"] in ("INVALID", "STALE") else side_a["color"]
    colour_b = "DATA" if side_b["status"] in ("INVALID", "STALE") else side_b["color"]
    return colour_a if order[colour_a] >= order[colour_b] else colour_b


def _band_reason(entry: dict[str, Any]) -> str:
    if entry["status"] in ("INVALID", "STALE"):
        return entry.get("reason") or "INVALID"
    if entry["color"] == "GREEN":
        return "WITHIN_GREEN_RANGE"
    return f"{entry['color']}_{entry['tail']}_TAIL_STATISTICAL_RARITY"


def evaluate_telemetry_signals(
    diagnosis: dict[str, Any], values: dict[str, Any], dq_statuses: dict[str, str]
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for variable in rules.VARIABLE_IDS:
        entry_a = diagnosis["classified"][variable]["A"]
        entry_b = diagnosis["classified"][variable]["B"]
        colour = _row_colour(entry_a, entry_b)
        # The reason quotes whichever side set the row colour.
        driver = entry_a if _row_colour(entry_a, entry_a) == colour else entry_b
        raw_sides = diagnosis["raw_variables"].get(variable, {})
        suspect = any(
            raw_sides.get(side, {}).get("dq_suspect") for side in ("A", "B")
        )
        usable = [entry["value"] for entry in (entry_a, entry_b) if _finite(entry["value"])]
        consistency = diagnosis["consistency"][variable]
        threshold = rules.THRESHOLDS[variable]
        signals.append({
            "variable": variable,
            "label": VARIABLE_LABELS[variable],
            "value": sum(usable) / len(usable) if usable else None,
            "unit": threshold["unit"],
            "process_status": colour,
            "status": COLOR_TO_STATUS[colour],
            "status_text": PROJECT_STATUS_WORD[colour],
            "signal_color": COLOR_TO_LAMP[colour],
            "dq_status": "UNUSABLE" if colour == "DATA" else "SUSPECT" if suspect else "VALID",
            "A": entry_a["value"], "B": entry_b["value"],
            "side_values": {"A": entry_a["value"], "B": entry_b["value"]},
            "tail": {"A": entry_a["tail"], "B": entry_b["tail"]},
            "consistency": consistency["consistency"],
            "ab_gap": consistency["diff"],
            "ab_limit": rules.CONCORDANCE.get(variable, {}).get("verify"),
            "rule_source": RULE_SOURCE,
            "rule_id": variable,
            "reason": _band_reason(driver),
            "green_range": [threshold["lowGreen"], threshold["highGreen"]],
            "semantics": "STATISTICAL_RARITY_NOT_PROCESS_ACTION",
        })

    for variable, label, unit, aggregate, value_fields, side_fields in INFORMATIONAL_SPECS:
        statuses = [dq_statuses.get(field, "MISSING") for field in INFORMATIONAL_DQ_FIELDS[variable]]
        unusable = any(status.startswith(UNUSABLE_DQ_PREFIXES) for status in statuses)
        suspect = any(status.startswith("SUSPECT") for status in statuses)
        raw = [values.get(field) for field in value_fields]
        if aggregate == "DIRECT":
            value = float(raw[0]) if raw and _finite(raw[0]) else None
        else:
            usable = [float(item) for item in raw if _finite(item)]
            value = sum(usable) / len(usable) if usable else None
        status = "DATA" if unusable or value is None else "DATA_WATCH" if suspect else "DATA_NORMAL"
        colour, text = INFORMATIONAL_STATUS[status]
        sides = None
        if side_fields:
            sides = {
                "A": float(values[side_fields[0]]) if _finite(values.get(side_fields[0])) else None,
                "B": float(values[side_fields[1]]) if _finite(values.get(side_fields[1])) else None,
            }
        signals.append({
            "variable": variable, "label": label, "value": value, "unit": unit,
            "process_status": "NOT_A_HEALTH_RULES_VARIABLE",
            "status": status, "status_text": text, "signal_color": colour,
            "dq_status": "UNUSABLE" if unusable else "SUSPECT" if suspect else "VALID",
            "A": sides["A"] if sides else None, "B": sides["B"] if sides else None,
            "side_values": sides, "tail": None,
            "consistency": "NOT_APPLICABLE", "ab_gap": None, "ab_limit": None,
            "rule_source": "PROJECT_EXCEL_INFORMATIONAL",
            "rule_id": None,
            "reason": "NO_PROJECT_HEALTH_RULE_DEFINED_FOR_THIS_VARIABLE",
            "green_range": None,
            "semantics": "INFORMATIONAL_NEVER_HEALTH_COLORED",
        })
    return signals


# --- operator actions -------------------------------------------------------
# Every entry is keyed by a condition code the project's own engine emits. The
# instructions restate the project's notes, gates and advisories; no setpoint,
# dose or percentage that the project does not define appears anywhere.
# Every entry is keyed by a condition code the project's own engine emits, and
# every number in the operator text is interpolated from the registry rather than
# typed here, so the instructions can never drift from the rules they cite.
_ACID, _BUFFER, _AMMONIA, _THERMAL = rules.ACID, rules.BUFFER, rules.AMMONIA, rules.THERMAL
# Short operator-facing cards. Each is one title plus at most three lines, all
# restating conditions and gates the project already defines -- no new strategy,
# threshold, feed change, dose or setpoint is introduced here.
_ACID, _BUFFER, _AMMONIA, _THERMAL = rules.ACID, rules.BUFFER, rules.AMMONIA, rules.THERMAL
ACTION_LIBRARY: dict[str, dict[str, Any]] = {
    "ACID_SEVERE": {
        "code": "OP-ACID-001", "priority": "ACTION", "title": "산발효 심각",
        "recommendation": [
            f"Project 명령에 따라 Feed_target = {_ACID['severeFeedFactor']:g} × Q_ref 로 조정하십시오.",
            "VFA/ALK를 직접 재확인하기 전까지 Feed를 늘리지 마십시오.",
        ],
    },
    "ACID_EARLY": {
        "code": "OP-ACID-002", "priority": "ACTION", "title": "산발효 조기 징후",
        "recommendation": [
            f"Project 명령에 따라 Feed_target = {_ACID['earlyFeedFactor']:g} × Q_ref 로 조정하십시오.",
            "다음 직접 VFA/TA 확인 전까지 Feed를 늘리지 마십시오.",
        ],
    },
    "ACID_WATCH": {
        "code": "OP-ACID-003", "priority": "WATCH", "title": "VFA/ALK 주의",
        "recommendation": [
            "VFA/ALK와 pH를 재확인하십시오.",
            "최근 투입량·유기부하 추세를 확인하십시오.",
        ],
    },
    "BUFFER_DEPLETION_CANDIDATE": {
        "code": "OP-BUFFER-001", "priority": "WATCH", "title": "완충능 저하 확인",
        "recommendation": [
            "ALK-B를 직접 재분석하고 pH 추세를 확인하십시오.",
            "Project에 투입량 계산식이 없으므로 약품 투입 명령은 발령하지 않습니다.",
        ],
    },
    "AMMONIA_WATCH": {
        "code": "OP-AMMONIA-001", "priority": "WATCH", "title": "암모니아 주의",
        "recommendation": [
            "신선한 TAN 직접 측정값을 확보하십시오.",
            "pH·온도를 함께 확인하십시오.",
        ],
    },
    "AMMONIA_ACTION_CANDIDATE": {
        "code": "OP-AMMONIA-002", "priority": "ACTION", "title": "암모니아 축적 확인",
        "recommendation": [
            "운전자 확인이 필요합니다. Project에 FAN 투입 계산식이 없습니다.",
            "TAN을 신규 측정하고 pH·온도를 함께 확인하십시오.",
        ],
    },
    "THERMAL_ACTION_CANDIDATE": {
        "code": "OP-THERMAL-001", "priority": "ACTION", "title": "온도 이상 확인",
        "recommendation": [
            "온도 센서와 가열·순환 계통 상태를 확인하십시오.",
            "기존 SOP에 따라 대응하십시오. Project에 승온 setpoint는 정의되어 있지 않습니다.",
        ],
    },
    "THERMAL_ONE_SIDED_ABNORMAL": {
        "code": "OP-DATA-001", "priority": "WATCH", "title": "온도 한쪽 이상",
        "recommendation": [
            "T-A / T-B 센서를 교차 검증하십시오.",
            "Project 기준상 국부 점검 대상이며 전역 명령은 발령하지 않습니다.",
        ],
    },
    "THERMAL_ONE_SIDED_INVALID": {
        "code": "OP-DATA-001", "priority": "WATCH", "title": "온도 채널 한쪽 무효",
        "recommendation": [
            "무효 채널의 센서 상태를 확인하십시오.",
            "Project 기준상 국부 점검 대상이며 전역 명령은 발령하지 않습니다.",
        ],
    },
    "DEGRADED_DATA_ONLY": {
        "code": "OP-DATA-003", "priority": "WATCH", "title": "계측 데이터 확인",
        "recommendation": [
            "무효·결측 채널을 확인하고 가능하면 신규 측정값으로 갱신하십시오.",
            "해당 값에 의존하는 조치는 신규 측정 확인 후 수행하십시오.",
        ],
    },
    "RECOVERED": {
        "code": "OP-RECOVERY-001", "priority": "WATCH", "title": "이전 경보 해제",
        "recommendation": [
            "이전 상승 상태가 해소되었습니다. 주요 계측 추세를 계속 확인하십시오.",
        ],
    },
    "NORMAL": {
        "code": "OP-NORMAL-001", "priority": "NORMAL", "title": "정상 운전 유지",
        "recommendation": [
            "현재 운전조건을 유지하고 주요 계측 추세를 지속 확인하십시오.",
        ],
    },
}
ONE_SIDED_RISK_ACTION = {
    "code": "OP-DATA-002", "priority": "WATCH", "title": "한쪽 계통만 이상",
    "recommendation": [
        "해당 항목을 A/B 동시 채취로 재측정하십시오.",
        "Project 기준상 전역 Feed 조치보다 데이터 확인을 우선합니다.",
    ],
}
PRIORITY_ORDER = {"ACTION": 0, "WATCH": 1, "NORMAL": 2}


VERDICT_TEXT = {
    "ACID_SEVERE": "산발효 심각", "ACID_EARLY": "산발효 조기 징후",
    "ACID_WATCH": "산발효 상태 주의", "BUFFER_DEPLETION_CANDIDATE": "완충능 저하 후보",
    "AMMONIA_WATCH": "암모니아 주의", "AMMONIA_ACTION_CANDIDATE": "암모니아 축적 후보",
    "THERMAL_ACTION_CANDIDATE": "소화조 온도 이상", "THERMAL_ONE_SIDED_ABNORMAL": "온도 한쪽 이상",
    "THERMAL_ONE_SIDED_INVALID": "온도 채널 한쪽 무효", "DEGRADED_DATA_ONLY": "계측 데이터 신뢰도 저하",
    "RECOVERED": "이전 경보 해제",
}


def evaluate_operator_actions(diagnosis: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derived only from project conditions and the project's own command gates."""
    health = diagnosis["health"]
    commands = diagnosis["commands"]
    flags = health["flags"]
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()

    for code in ([flag["code"] for flag in flags] or [health["code"]]):
        template = ACTION_LIBRARY.get(code)
        if template is None and code.endswith("_ONE_SIDED_RISK_RED"):
            template = ONE_SIDED_RISK_ACTION
        if template is None or code in seen:
            continue
        seen.add(code)
        note = next((flag.get("note") for flag in flags if flag["code"] == code), None)
        actions.append({
            **template,
            "recommendation": list(template["recommendation"]),
            "reason": code,
            "verdict": VERDICT_TEXT.get(code, template["title"]),
            "evidence": note or (health["notes"][0] if health["notes"] else health["code"]),
            "evidence_readings": next(
                (item["readings"] for item in evidence["process"] if item["code"] == code), []
            ) if evidence else [],
            "rule_source": RULE_SOURCE,
            "rule_id": "evaluateProcessHealth+deriveOperatorCommands",
        })

    if not actions:
        template = ACTION_LIBRARY["NORMAL"]
        actions.append({
            **template, "recommendation": list(template["recommendation"]),
            "reason": health["code"], "verdict": "주요 Project Health Rule 조건 내 운전 중",
            "evidence": "PROCESS_NORMAL", "evidence_readings": [],
            "rule_source": RULE_SOURCE,
            "rule_id": "evaluateProcessHealth+deriveOperatorCommands",
        })
    actions.sort(key=lambda item: PRIORITY_ORDER[item["priority"]])

    state = health["state"]
    return {
        "state": state,
        "lamp": "GREEN" if state == "NORMAL_DEGRADED_DATA" else rules.STATE_TO_LAMP[state],
        "combined_lamp": rules.STATE_TO_LAMP[state],
        "actions": actions,
        "feed": {
            "target_tpd": commands["feedTarget"],
            "q_ref_tpd": commands["qRefTonPerDay"],
            "note": commands["feedNote"],
            "evidence": (
                "Project ACID 조건이 성립하여 Feed_target이 지정되었습니다."
                if commands["feedTarget"] is not None
                else "증감을 정당화하는 Project process-action 조건이 현재 없습니다."
            ),
        },
        "gates": {"thermal": commands["thermalGate"], "alkali": commands["alkaliGate"]},
        "advisories": commands["advisories"],
        # The project's own CONTROL_ADAPTER is LOCAL_PRESENTATION and never
        # transmits; this prototype has no physical link either.
        "control_mode": "NO_LIVE_CONTROL",
        "rule_source": RULE_SOURCE,
    }
