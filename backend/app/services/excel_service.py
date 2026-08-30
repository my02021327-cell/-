from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from openpyxl import load_workbook


SHEET_NAME = "BIOGUARD_INPUT"
IMPUTATION_SHEET = "IMPUTATION_LOG"
HEADERS = [
    "date", "feed_A_tpd", "feed_B_tpd", "biogas_A_m3d", "biogas_B_m3d",
    "ch4_purity_A_pct", "ch4_purity_B_pct", "temperature_A_C", "temperature_B_C",
    "pH_A", "pH_B", "VFA_A_mgL", "VFA_B_mgL", "ALK_A_mgL", "ALK_B_mgL",
    "TAN_A_mgL", "TAN_B_mgL", "TS_A_pct", "TS_B_pct", "VS_A_pct", "VS_B_pct",
    "CODcr_A_mgL", "CODcr_B_mgL",
]


class WorkbookValidationError(ValueError):
    pass


def parse_workbook(payload: bytes) -> pd.DataFrame:
    try:
        workbook = load_workbook(BytesIO(payload), read_only=True, data_only=True)
    except Exception as exc:
        raise WorkbookValidationError(f"INVALID_WORKBOOK: {exc}") from exc
    if SHEET_NAME not in workbook.sheetnames:
        raise WorkbookValidationError(f"MISSING_SHEET: {SHEET_NAME}")
    sheet = workbook[SHEET_NAME]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise WorkbookValidationError("EMPTY_SHEET")
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    missing = [header for header in HEADERS if header not in headers]
    if missing:
        raise WorkbookValidationError("MISSING_HEADERS: " + ",".join(missing))
    indices = {header: headers.index(header) for header in HEADERS}
    records: list[dict[str, Any]] = []
    for source_row, values in enumerate(rows[1:], start=2):
        if all(values[indices[header]] in (None, "") for header in HEADERS):
            continue
        record = {header: values[indices[header]] for header in HEADERS}
        record["source_row"] = source_row
        records.append(record)
    if not records:
        raise WorkbookValidationError("NO_DATA_ROWS")
    frame = pd.DataFrame(records)
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if parsed_dates.isna().any():
        bad = frame.loc[parsed_dates.isna(), "source_row"].astype(str).tolist()
        raise WorkbookValidationError("INVALID_DATE_ROWS: " + ",".join(bad))
    if parsed_dates.duplicated().any():
        dup = parsed_dates[parsed_dates.duplicated(keep=False)].dt.strftime("%Y-%m-%d").unique().tolist()
        raise WorkbookValidationError("DUPLICATE_DATES: " + ",".join(dup))
    frame["date"] = parsed_dates
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


# Demo workbooks record which cells were synthesised. The log uses either the
# schema column name or a short alias, so both are accepted; anything else is
# ignored rather than guessed at.
IMPUTATION_FIELD_ALIASES = {
    "temp_A": "temperature_A_C", "temp_B": "temperature_B_C",
    "purity_A": "ch4_purity_A_pct", "purity_B": "ch4_purity_B_pct",
    "VFA_A": "VFA_A_mgL", "VFA_B": "VFA_B_mgL",
    "ALK_A": "ALK_A_mgL", "ALK_B": "ALK_B_mgL",
    "TAN_A": "TAN_A_mgL", "TAN_B": "TAN_B_mgL",
    "TS_A": "TS_A_pct", "TS_B": "TS_B_pct",
    "VS_A": "VS_A_pct", "VS_B": "VS_B_pct",
    "CODcr_A": "CODcr_A_mgL", "CODcr_B": "CODcr_B_mgL",
    "biogas_A": "biogas_A_m3d", "biogas_B": "biogas_B_m3d",
    "feed_A": "feed_A_tpd", "feed_B": "feed_B_tpd",
}


def parse_imputation_log(payload: bytes) -> pd.DataFrame:
    """Return the (date, field) cells the workbook declares as synthesised.

    An empty frame means the workbook makes no such declaration, in which case
    every value is treated as an original reading.
    """
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "field": pd.Series(dtype=str)})
    try:
        workbook = load_workbook(BytesIO(payload), read_only=True, data_only=True)
    except Exception:
        return empty
    if IMPUTATION_SHEET not in workbook.sheetnames:
        return empty
    rows = list(workbook[IMPUTATION_SHEET].iter_rows(values_only=True))
    if len(rows) < 2:
        return empty
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    if "date" not in headers or "field" not in headers:
        return empty
    date_index, field_index = headers.index("date"), headers.index("field")
    records: list[dict[str, Any]] = []
    for values in rows[1:]:
        if date_index >= len(values) or field_index >= len(values):
            continue
        raw_field = values[field_index]
        if raw_field is None:
            continue
        name = str(raw_field).strip()
        column = IMPUTATION_FIELD_ALIASES.get(name, name)
        if column not in HEADERS:
            continue
        records.append({"date": values[date_index], "field": column})
    if not records:
        return empty
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).drop_duplicates()
    return frame.reset_index(drop=True) if len(frame) else empty
