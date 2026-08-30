from __future__ import annotations

from typing import Any

import pandas as pd


def _finite(value: Any) -> bool:
    try:
        return bool(pd.notna(value)) and float(value) == float(value)
    except (TypeError, ValueError):
        return False


class FeatureService:
    """Build the deployment frame exclusively from one uploaded dataset."""

    REQUIRED_ORIGIN_FEATURES = ("feed_A_tpd", "feed_B_tpd", "feed_AB_tpd")

    def prepare_candidate(
        self,
        frame: pd.DataFrame,
        candidate: pd.Timestamp,
        *,
        minimum_history_rows: int,
    ) -> tuple[pd.DataFrame, list[str]]:
        prefix = frame.loc[frame["date"].le(candidate)].copy()
        reasons: list[str] = []
        if len(prefix) < minimum_history_rows:
            reasons.append(f"MINIMUM_HISTORY_{minimum_history_rows}_ROWS_REQUIRED")
        elif (candidate - prefix["date"].min()).days < minimum_history_rows - 1:
            reasons.append("MINIMUM_CALENDAR_HISTORY_NOT_MET")
        latest = prefix.loc[prefix["date"].eq(candidate)].tail(1)
        for feature in self.REQUIRED_ORIGIN_FEATURES:
            if latest.empty or not _finite(latest.iloc[0].get(feature)):
                reasons.append(f"REQUIRED_FEATURE_UNAVAILABLE:{feature}")
        return prefix, reasons
