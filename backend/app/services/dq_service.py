from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .excel_service import HEADERS


MISSING = {"MISSING", "MISSING_ZERO_SENTINEL"}
INVALID = {"INVALID_PARSE", "INVALID_RANGE"}
SUSPECT = {"SUSPECT_TEMPORAL_OUTLIER", "SUSPECT_SPIKE", "SUSPECT_FLATLINE", "SUSPECT_CROSS_FIELD"}


@dataclass(frozen=True)
class Rule:
    minimum: float | None = None
    maximum: float | None = None
    zero_missing: bool = False
    flatline: bool = False


RULES: dict[str, Rule] = {
    "feed_A_tpd": Rule(0), "feed_B_tpd": Rule(0),
    "biogas_A_m3d": Rule(0), "biogas_B_m3d": Rule(0),
    "ch4_purity_A_pct": Rule(0, 100, True, True), "ch4_purity_B_pct": Rule(0, 100, True, True),
    "temperature_A_C": Rule(-20, 100, True, True), "temperature_B_C": Rule(-20, 100, True, True),
    "pH_A": Rule(0, 14, True, True), "pH_B": Rule(0, 14, True, True),
    "VFA_A_mgL": Rule(0, None, True, True), "VFA_B_mgL": Rule(0, None, True, True),
    "ALK_A_mgL": Rule(0, None, True, True), "ALK_B_mgL": Rule(0, None, True, True),
    "TAN_A_mgL": Rule(0, None, True, True), "TAN_B_mgL": Rule(0, None, True, True),
    "TS_A_pct": Rule(0, 100, True, True), "TS_B_pct": Rule(0, 100, True, True),
    "VS_A_pct": Rule(0, 100, True, True), "VS_B_pct": Rule(0, 100, True, True),
    "CODcr_A_mgL": Rule(0, None, True, True), "CODcr_B_mgL": Rule(0, None, True, True),
}


def _mad(values: list[float]) -> tuple[float, float]:
    center = float(np.median(values))
    return center, float(np.median(np.abs(np.asarray(values) - center)))


def run_data_quality(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clean = raw[["date", "source_row"]].copy()
    records: list[dict[str, Any]] = []
    for column in HEADERS[1:]:
        rule = RULES[column]
        parsed: list[float | None] = []
        statuses: list[str] = []
        reasons: list[str] = []
        for value in raw[column]:
            if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
                parsed.append(None); statuses.append("MISSING"); reasons.append("EMPTY_CELL"); continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                parsed.append(None); statuses.append("INVALID_PARSE"); reasons.append("NON_NUMERIC"); continue
            if not np.isfinite(number):
                parsed.append(None); statuses.append("INVALID_PARSE"); reasons.append("NON_FINITE"); continue
            if rule.zero_missing and number == 0:
                parsed.append(None); statuses.append("MISSING_ZERO_SENTINEL"); reasons.append("ZERO_SENTINEL"); continue
            if (rule.minimum is not None and number < rule.minimum) or (rule.maximum is not None and number > rule.maximum):
                parsed.append(None); statuses.append("INVALID_RANGE"); reasons.append("PHYSICAL_RANGE"); continue
            parsed.append(number); statuses.append("VALID"); reasons.append("")

        history: list[float] = []
        deltas: list[float] = []
        previous: float | None = None
        flat_start = 0
        for index, value in enumerate(parsed):
            if value is None or statuses[index] != "VALID":
                previous = None if value is None else value
                flat_start = index + 1
                continue
            window = history[-15:]
            if len(window) >= 7:
                center, mad = _mad(window)
                if mad > 0 and abs(value - center) > 3 * 1.4826 * mad:
                    statuses[index] = "SUSPECT_TEMPORAL_OUTLIER"; reasons[index] = "HAMPEL_WINDOW_15"
            if previous is not None:
                delta = value - previous
                delta_window = deltas[-15:]
                if len(delta_window) >= 7:
                    center, mad = _mad(delta_window)
                    if mad > 0 and abs(delta - center) > 3 * 1.4826 * mad:
                        statuses[index] = "SUSPECT_SPIKE"; reasons[index] = "ROBUST_DELTA_WINDOW_15"
                deltas.append(delta)
            if rule.flatline and index > 0 and parsed[index - 1] == value:
                if index - flat_start + 1 >= 7:
                    for pos in range(flat_start, index + 1):
                        if statuses[pos] == "VALID":
                            statuses[pos] = "SUSPECT_FLATLINE"; reasons[pos] = "EXACT_REPEAT_7_PLUS"
            else:
                flat_start = index
            history.append(value)
            previous = value

        clean[column] = parsed
        for index, row in raw.iterrows():
            records.append({
                "date": row["date"], "source_row": int(row["source_row"]), "field": column,
                "raw_value": row[column], "clean_value": parsed[index],
                "dq_status": statuses[index], "dq_reason": reasons[index],
            })

    dq = pd.DataFrame(records)
    # Cross-field corruption: exact VFA=ALK on the same side is suspect, not deleted.
    for side in ("A", "B"):
        vfa, alk = f"VFA_{side}_mgL", f"ALK_{side}_mgL"
        same = clean[vfa].notna() & clean[alk].notna() & clean[vfa].eq(clean[alk])
        for date in clean.loc[same, "date"]:
            mask = dq["date"].eq(date) & dq["field"].eq(alk)
            dq.loc[mask, ["dq_status", "dq_reason"]] = ["SUSPECT_CROSS_FIELD", "ALK_IDENTICAL_TO_VFA"]
    # SUSPECT values remain clean/model-usable; only missing/invalid are unavailable.
    for column in HEADERS[1:]:
        field = dq.loc[dq["field"].eq(column)].sort_values("date")
        unusable = field["dq_status"].isin(MISSING | INVALID).to_numpy()
        clean.loc[unusable, column] = np.nan
    counts = dq["dq_status"].value_counts().to_dict()
    summary = {
        "valid": int(counts.get("VALID", 0)),
        "missing": int(sum(counts.get(key, 0) for key in MISSING)),
        "suspect": int(sum(counts.get(key, 0) for key in SUSPECT)),
        "invalid": int(sum(counts.get(key, 0) for key in INVALID)),
        "by_status": {str(key): int(value) for key, value in counts.items()},
        "raw_preserved": True,
        "suspect_values_retained": True,
    }
    return clean, dq, summary
