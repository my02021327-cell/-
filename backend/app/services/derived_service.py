from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["feed_A"] = result["feed_A_tpd"]
    result["feed_B"] = result["feed_B_tpd"]
    result["feed_AB_tpd"] = result[["feed_A_tpd", "feed_B_tpd"]].sum(axis=1, min_count=2)
    result["biogas_AB_m3d"] = result[["biogas_A_m3d", "biogas_B_m3d"]].sum(axis=1, min_count=2)
    result["VFA_TA_A"] = np.where(result["ALK_A_mgL"] > 0, result["VFA_A_mgL"] / result["ALK_A_mgL"], np.nan)
    result["VFA_TA_B"] = np.where(result["ALK_B_mgL"] > 0, result["VFA_B_mgL"] / result["ALK_B_mgL"], np.nan)
    required = ["biogas_A_m3d", "biogas_B_m3d", "ch4_purity_A_pct", "ch4_purity_B_pct"]
    valid = result[required].notna().all(axis=1)
    result["CH4_m3d_observed"] = np.where(
        valid,
        result["biogas_A_m3d"] * result["ch4_purity_A_pct"] / 100.0
        + result["biogas_B_m3d"] * result["ch4_purity_B_pct"] / 100.0,
        np.nan,
    )
    return result
