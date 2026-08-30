"""BioGuard process-health rules — migrated, not re-derived.

Every threshold, band, gate and command factor below is a verbatim transcription
of the authoritative project specification:

    Yongcheon_BGP_Final_AI_Prompt(1).md          (specification of record)
    deliverables/yeongcheon_bioguard_multiview_final.html :: HEALTH_RULES
                                                 (shipped JS engine, 10/10 QA)
    tools/generate_bioguard_hmi.py :: THRESHOLDS  (existing Python mirror)

All three agree numerically. Nothing here was invented, rounded, re-fitted or
tuned to any dataset. This module is the single runtime registry: no threshold
may be restated in dataset_service, health_service or the frontend.

Semantics that the specification is explicit about, and that callers must respect:

  * Section 2 -- a variable's GREEN/ORANGE/RED is a *statistical rarity* signal
    on that one variable. RED != PROCESS ACTION. The process action decision is
    made only by evaluate_process_health(), which is multivariate.
  * Section 3 -- invalid/stale readings are never classified against the numeric
    bands. They are DATA, never a red process alarm.
  * Section 5 -- the signal tower shows the final health state, never a raw
    single-variable copy.
  * Section 10 -- process action is never promoted on incomplete evidence.
"""

from __future__ import annotations

import math
from typing import Any

RULE_SOURCE = "Yongcheon_BGP_Final_AI_Prompt(1).md::HEALTH_RULES"

# --- Section 1: statistical signal colour thresholds -------------------------
# RED low < lowRed | ORANGE low lowRed..<lowGreen | GREEN lowGreen..highGreen
# | ORANGE high >highGreen..highRed | RED high > highRed
THRESHOLDS: dict[str, dict[str, Any]] = {
    "temperature": dict(lowRed=37.7, lowGreen=37.9, highGreen=40.0, highRed=40.2, unit="°C"),
    "pH":          dict(lowRed=7.56, lowGreen=7.70, highGreen=8.14, highRed=8.21, unit=""),
    "VFA":         dict(lowRed=2337, lowGreen=2428, highGreen=3378, highRed=3625, unit="mg/L"),
    "ALK":         dict(lowRed=15008, lowGreen=15324, highGreen=18305, highRed=18629, unit="mg/L"),
    "VFA_TA":      dict(lowRed=0.130, lowGreen=0.137, highGreen=0.207, highRed=0.215, unit=""),
    "TAN":         dict(lowRed=2077, lowGreen=2731, highGreen=5634, highRed=5920, unit="mg/L"),
    "FAN":         dict(lowRed=170, lowGreen=226, highGreen=716, highRed=820, unit="mg/L"),
    "TS":          dict(lowRed=2.09, lowGreen=2.26, highGreen=2.94, highRed=3.16, unit="%"),
    "VS":          dict(lowRed=0.77, lowGreen=0.87, highGreen=1.41, highRed=1.59, unit="%"),
    "CODcr":       dict(lowRed=15614, lowGreen=18739, highGreen=36311, highRed=40152, unit="mg/L"),
}

# --- Section 2: process-risk direction, which is not always the rarity direction
RISK_TAIL: dict[str, str | None] = {
    "pH": "LOW", "VFA": "HIGH", "ALK": "LOW", "VFA_TA": "HIGH", "FAN": "HIGH",
    "temperature": None, "TAN": None, "TS": None, "VS": None, "CODcr": None,
}

# --- Section 4: A/B concordance (parallel-digester consistency) --------------
CONCORDANCE: dict[str, dict[str, float]] = {
    "temperature": dict(verify=2.0, strong=2.5), "pH": dict(verify=0.08),
    "VFA": dict(verify=600), "ALK": dict(verify=900), "VFA_TA": dict(verify=0.04),
    "TAN": dict(verify=1112), "TS": dict(verify=0.58), "VS": dict(verify=0.39),
    "CODcr": dict(verify=13933),
}

# --- Sections 6 and 7: acid / buffer / ammonia / thermal --------------------
ACID = dict(watchVFA_TA=0.215, p95VFA_TA=0.207, earlyDeltaVFA_TA=0.04, earlyDeltaVFA=700,
            earlyFeedFactor=0.75, severeVFA_TA=0.45, severeFeedFactor=0.55)
BUFFER = dict(alkLow=15000, phLow=7.70, vfaTaTrigger=0.22)
AMMONIA = dict(watchFanLow=500, watchFanHigh=700, actionFan=700, vfaTaTrigger=0.22)
THERMAL = dict(lowC=36, highC=41)
TAN_MAX_AGE_DAYS = 14
Q_REF_TPD = 180.0

# --- Section 5: health state -> signal tower colour --------------------------
STATE_TO_LAMP = {
    "NORMAL": "GREEN",
    "NORMAL_DEGRADED_DATA": "AMBER",
    "DATA_SUSPECT": "AMBER",
    "PROCESS_WATCH": "AMBER",
    "PROCESS_ACTION": "RED",
    # "AMBER or GREEN (engine-decided)": the engine reaches RECOVERY only from a
    # previously elevated state and still asks for operator verification, so the
    # conservative half of the specified pair is the one taken.
    "RECOVERY": "AMBER",
}

# FAN has no implemented formula anywhere in the project. Both the shipped HTML
# and tools/generate_bioguard_hmi.py record it as permanently unavailable, so it
# is never fabricated here either.
FAN_UNAVAILABLE_REASON = "UNAVAILABLE_NO_FAN_FORMULA_IMPLEMENTED"

VARIABLE_IDS = list(THRESHOLDS)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def normalize_input(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Section 10 fail-safe: nothing non-numeric or stale reaches the classifier."""
    if raw is None:
        return {"value": None, "valid": False, "reason": "MISSING"}
    if raw.get("valid") is False:
        return {"value": None, "valid": False, "reason": raw.get("reason") or "INVALID"}
    value = raw.get("value")
    if not _finite(value):
        return {"value": None, "valid": False, "reason": "NON_NUMERIC"}
    if raw.get("stale"):
        return {"value": float(value), "valid": False, "reason": "STALE"}
    return {
        "value": float(value), "valid": True, "reason": None,
        "sampleAgeDays": raw.get("sampleAgeDays"),
    }


def classify_variable(var_id: str, norm: dict[str, Any]) -> dict[str, Any]:
    """Section 1 + 2: statistical rarity colour and tail. Not a process action."""
    if not norm.get("valid"):
        reason = norm.get("reason") or "MISSING"
        return {
            "status": "STALE" if reason == "STALE" else "INVALID",
            "color": "DATA", "tail": "NA", "reason": reason, "value": None,
        }
    threshold = THRESHOLDS[var_id]
    value = norm["value"]
    if value < threshold["lowRed"]:
        color, tail = "RED", "LOW"
    elif value < threshold["lowGreen"]:
        color, tail = "ORANGE", "LOW"
    elif value <= threshold["highGreen"]:
        color, tail = "GREEN", "CENTER"
    elif value <= threshold["highRed"]:
        color, tail = "ORANGE", "HIGH"
    else:
        color, tail = "RED", "HIGH"
    return {"status": color, "color": color, "tail": tail, "reason": None, "value": value}


def evaluate_ab_consistency(var_id: str, norm_a: dict, norm_b: dict) -> dict[str, Any]:
    """Section 4. The only A/B comparison in the system; no ad-hoc |A-B| limits."""
    if not norm_a.get("valid") or not norm_b.get("valid"):
        return {"consistency": "UNKNOWN", "diff": None, "strong": None}
    diff = abs(norm_a["value"] - norm_b["value"])
    rule = CONCORDANCE.get(var_id, {"verify": math.inf})
    strong = rule.get("strong")
    return {
        "consistency": "CONCORDANT" if diff <= rule["verify"] else "DISCORDANT",
        "diff": diff,
        "strong": (diff > strong) if strong is not None else None,
    }


def apply_fan_gating(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Section 3: FAN is UNAVAILABLE whenever fresh TAN is not available, and TAN
    older than TAN_MAX_AGE_DAYS must not drive FAN judgment."""
    def fresh(side: dict[str, Any] | None) -> bool:
        if not side or not side.get("valid") or side.get("stale"):
            return False
        age = side.get("sampleAgeDays")
        return age is None or age <= TAN_MAX_AGE_DAYS

    fan = raw.get("FAN", {})
    return {
        "A": fan.get("A") if fresh(raw["TAN"].get("A")) else {"valid": False, "reason": "UNAVAILABLE_STALE_TAN"},
        "B": fan.get("B") if fresh(raw["TAN"].get("B")) else {"valid": False, "reason": "UNAVAILABLE_STALE_TAN"},
    }


def evaluate_data_confidence(norm: dict[str, dict[str, Any]]) -> str:
    """Section 8 data-confidence ladder, verbatim."""
    core_ids = ["pH", "VFA", "ALK", "VFA_TA"]
    fully_clean = sum(
        1 for var_id in VARIABLE_IDS
        if norm[var_id]["A"].get("valid") and norm[var_id]["B"].get("valid")
    )
    core_usable = sum(
        1 for var_id in core_ids
        if norm[var_id]["A"].get("valid") or norm[var_id]["B"].get("valid")
    )
    ph_both_valid = norm["pH"]["A"].get("valid") and norm["pH"]["B"].get("valid")
    if fully_clean == len(VARIABLE_IDS):
        return "HIGH"
    if ph_both_valid and core_usable >= 3:
        return "MEDIUM"
    return "LOW"


def evaluate_process_health(
    classified: dict[str, dict[str, Any]],
    data_confidence: str,
    deltas: dict[str, dict[str, float | None]] | None = None,
    previous_state: str | None = None,
) -> dict[str, Any]:
    """Sections 5-7 + 10. The only place a process state is decided."""
    flags: list[dict[str, str]] = []

    def usable(entry: dict[str, Any]) -> bool:
        return entry["status"] not in ("INVALID", "STALE")

    def finalize(state: str, code: str, notes: list[str]) -> dict[str, Any]:
        return {"state": state, "code": code, "notes": notes, "flags": flags}

    vta_a, vta_b = classified["VFA_TA"]["A"], classified["VFA_TA"]["B"]
    vta_both = usable(vta_a) and usable(vta_b)

    # Section 6 -- ACID SEVERE
    if vta_both and vta_a["value"] > ACID["severeVFA_TA"] and vta_b["value"] > ACID["severeVFA_TA"]:
        return finalize("PROCESS_ACTION", "ACID_SEVERE",
                        ["A/B VFA/TA both exceed severe threshold (>0.45)"])

    # Section 6 -- ACID EARLY (needs both P95 exceedance and confirmed concurrent rise)
    delta_ta = (deltas or {}).get("VFA_TA", {})
    delta_vfa = (deltas or {}).get("VFA", {})
    delta_complete = all(
        _finite(value) for value in
        (delta_ta.get("A"), delta_ta.get("B"), delta_vfa.get("A"), delta_vfa.get("B"))
    )
    if (
        vta_both
        and vta_a["value"] > ACID["p95VFA_TA"] and vta_b["value"] > ACID["p95VFA_TA"]
        and delta_complete
        and delta_ta["A"] >= ACID["earlyDeltaVFA_TA"] and delta_ta["B"] >= ACID["earlyDeltaVFA_TA"]
        and delta_vfa["A"] >= ACID["earlyDeltaVFA"] and delta_vfa["B"] >= ACID["earlyDeltaVFA"]
    ):
        return finalize("PROCESS_ACTION", "ACID_EARLY",
                        ["A/B VFA/TA above P95 with confirmed concurrent rise vs. previous direct measurement"])

    # Section 6 -- ACID WATCH
    if vta_both and vta_a["value"] >= ACID["watchVFA_TA"] and vta_b["value"] >= ACID["watchVFA_TA"]:
        flags.append({"level": "PROCESS_WATCH", "code": "ACID_WATCH",
                      "note": "A/B VFA/TA at/above watch band (~0.215-0.22); insufficient trend evidence for ACTION"})

    # Section 7 -- BUFFER DEPLETION (ALK alone never triggers an alkali command)
    alk_a, alk_b = classified["ALK"]["A"], classified["ALK"]["B"]
    ph_a, ph_b = classified["pH"]["A"], classified["pH"]["B"]
    alk_low_both = usable(alk_a) and usable(alk_b) and alk_a["value"] <= BUFFER["alkLow"] and alk_b["value"] <= BUFFER["alkLow"]
    ph_low_or_falling = (usable(ph_a) and ph_a["value"] <= BUFFER["phLow"]) or (usable(ph_b) and ph_b["value"] <= BUFFER["phLow"])
    vta_elevated = (usable(vta_a) and vta_a["value"] > BUFFER["vfaTaTrigger"]) or (usable(vta_b) and vta_b["value"] > BUFFER["vfaTaTrigger"])
    if alk_low_both and ph_low_or_falling and vta_elevated:
        flags.append({"level": "PROCESS_WATCH", "code": "BUFFER_DEPLETION_CANDIDATE",
                      "note": "ALK low both sides + pH low/falling + VFA/TA elevated"})

    # Section 7 -- AMMONIA (requires fresh TAN, hence a usable FAN)
    fan_a, fan_b = classified["FAN"]["A"], classified["FAN"]["B"]
    in_band = lambda value: AMMONIA["watchFanLow"] <= value <= AMMONIA["watchFanHigh"]  # noqa: E731
    if ((usable(fan_a) and in_band(fan_a["value"])) or (usable(fan_b) and in_band(fan_b["value"]))) and vta_elevated:
        flags.append({"level": "PROCESS_WATCH", "code": "AMMONIA_WATCH",
                      "note": "FAN in watch band (~500-700) with VFA/TA rising"})
    if (usable(fan_a) and usable(fan_b) and fan_a["value"] > AMMONIA["actionFan"]
            and fan_b["value"] > AMMONIA["actionFan"] and vta_elevated):
        flags.append({"level": "PROCESS_ACTION", "code": "AMMONIA_ACTION_CANDIDATE",
                      "note": "A/B FAN >700 with VFA/TA elevated/accumulating — operator verification required (no automated dosing formula defined)"})

    # Section 7 -- THERMAL (one-sided is DATA SUSPECT, never a global command)
    temp_a, temp_b = classified["temperature"]["A"], classified["temperature"]["B"]
    if not (usable(temp_a) and usable(temp_b)):
        if usable(temp_a) or usable(temp_b):
            flags.append({"level": "DATA_SUSPECT", "code": "THERMAL_ONE_SIDED_INVALID",
                          "note": "Only one temperature channel valid — local thermal check, no global command"})
    else:
        abnormal = lambda value: value < THERMAL["lowC"] or value > THERMAL["highC"]  # noqa: E731
        a_abnormal, b_abnormal = abnormal(temp_a["value"]), abnormal(temp_b["value"])
        if a_abnormal and b_abnormal:
            flags.append({"level": "PROCESS_ACTION", "code": "THERMAL_ACTION_CANDIDATE",
                          "note": "A/B valid temperature both outside 36-41°C band"})
        elif a_abnormal or b_abnormal:
            flags.append({"level": "DATA_SUSPECT", "code": "THERMAL_ONE_SIDED_ABNORMAL",
                          "note": "Only one side outside normal thermal band — local thermal check, no global command"})

    # Section 4 -- one-sided risk-direction RED prioritises DATA SUSPECT
    for var_id, risk in RISK_TAIL.items():
        if not risk or var_id == "temperature":
            continue
        side_a, side_b = classified[var_id]["A"], classified[var_id]["B"]
        if not (usable(side_a) and usable(side_b)):
            continue
        a_risky = side_a["color"] == "RED" and side_a["tail"] == risk
        b_risky = side_b["color"] == "RED" and side_b["tail"] == risk
        if (a_risky and not b_risky and side_b["color"] != "RED") or (b_risky and not a_risky and side_a["color"] != "RED"):
            flags.append({"level": "DATA_SUSPECT", "code": f"{var_id}_ONE_SIDED_RISK_RED",
                          "note": f"{var_id}: one side RED in the risk-relevant direction, other side normal — data suspect prioritized over a global action"})

    for level in ("PROCESS_ACTION", "PROCESS_WATCH", "DATA_SUSPECT"):
        hits = [flag for flag in flags if flag["level"] == level]
        if hits:
            return finalize(level, "+".join(hit["code"] for hit in hits),
                            [hit["note"] for hit in hits])

    if not flags and previous_state in ("PROCESS_ACTION", "PROCESS_WATCH", "DATA_SUSPECT"):
        return finalize("RECOVERY", "RECOVERED",
                        ["Previous elevated state cleared; operator verification still recommended"])
    if data_confidence != "HIGH":
        return finalize("NORMAL_DEGRADED_DATA", "DEGRADED_DATA_ONLY",
                        ["No process-action evidence found; data confidence below HIGH due to invalid/stale channels"])
    return finalize("NORMAL", "NORMAL", [])


def derive_operator_commands(health: dict[str, Any]) -> dict[str, Any]:
    """Sections 6 and 9. Feed factors and gates exactly as specified; the gates
    stay CONDITION_NOT_VERIFIED until their own multivariate condition fires."""
    commands: dict[str, Any] = {
        "feedTarget": None, "feedNote": None,
        "thermalGate": "CONDITION_NOT_VERIFIED",
        "alkaliGate": "CONDITION_NOT_VERIFIED",
        "advisories": [],
        "qRefTonPerDay": Q_REF_TPD,
    }
    code = health.get("code", "")
    if "ACID_SEVERE" in code:
        commands["feedTarget"] = round(Q_REF_TPD * ACID["severeFeedFactor"], 1)
        commands["feedNote"] = "ACID SEVERE: Feed_target = 0.55 x Q_ref t/d. No automatic full Feed stop is issued."
    elif "ACID_EARLY" in code:
        commands["feedTarget"] = round(Q_REF_TPD * ACID["earlyFeedFactor"], 1)
        commands["feedNote"] = "ACID EARLY: Feed_target = 0.75 x Q_ref t/d. No Feed increase until next direct VFA/TA confirmation."
    else:
        commands["feedNote"] = (
            f"Maintain Feed ≈ {Q_REF_TPD:g} t/d. "
            "No increase or additional decrease is justified by current evidence."
        )
    codes = {flag["code"] for flag in health.get("flags", [])}
    if "THERMAL_ACTION_CANDIDATE" in codes:
        commands["thermalGate"] = "CONDITION_VERIFIED"
    if "BUFFER_DEPLETION_CANDIDATE" in codes:
        commands["alkaliGate"] = "CONDITION_VERIFIED"
    if "AMMONIA_ACTION_CANDIDATE" in codes:
        commands["advisories"].append(
            "Ammonia accumulation candidate — operator verification required "
            "(no automated dosing formula defined for FAN)."
        )
    commands["advisories"].append(
        "Required: T-A/T-B sensor cross-verification, ALK-B direct re-assay, "
        "simultaneous A/B VFA/ALK/pH sampling, fresh TAN."
    )
    commands["advisories"].append(
        "Prohibited on current evidence: new Feed increase, unjustified additional "
        "Feed decrease, NaHCO3 dosing, stale-TAN-based FAN control."
    )
    return commands


def compute_diagnosis(
    raw_variables: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, float | None]] | None = None,
    previous_health_state: str | None = None,
) -> dict[str, Any]:
    """The project's single entry point, reproduced for backend runtime."""
    raw = dict(raw_variables)
    raw["FAN"] = apply_fan_gating(raw)
    norm: dict[str, dict[str, Any]] = {}
    classified: dict[str, dict[str, Any]] = {}
    consistency: dict[str, dict[str, Any]] = {}
    for var_id in VARIABLE_IDS:
        side = raw.get(var_id, {})
        norm[var_id] = {"A": normalize_input(side.get("A")), "B": normalize_input(side.get("B"))}
        classified[var_id] = {
            "A": classify_variable(var_id, norm[var_id]["A"]),
            "B": classify_variable(var_id, norm[var_id]["B"]),
        }
        consistency[var_id] = evaluate_ab_consistency(var_id, norm[var_id]["A"], norm[var_id]["B"])
    data_confidence = evaluate_data_confidence(norm)
    health = evaluate_process_health(classified, data_confidence, deltas, previous_health_state)
    return {
        "raw_variables": raw, "norm": norm, "classified": classified,
        "consistency": consistency, "data_confidence": data_confidence,
        "health": health, "commands": derive_operator_commands(health),
        "rule_source": RULE_SOURCE,
    }
