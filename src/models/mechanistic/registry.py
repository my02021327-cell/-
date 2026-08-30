"""Closed PHASE 05 mechanistic-candidate registry.

The registry is deliberately descriptive.  It does not rank candidates and it
does not turn unavailable hydraulic or COD quantities into fitted inputs.
"""

from __future__ import annotations

import json

import pandas as pd

from .kernels import K_CANDIDATES, KERNEL_TAIL_TOL
from .kinetic_state import MIN_KERNEL_COVERAGE


TARGETS = ("CH4_m3d_observed", "biogas_AB_m3d")
HORIZONS = (1, 3, 7, 14, 30)
TAU_RES_CANDIDATES = (1, 2, 3, 5, 7, 10, 14, 21, 30)


def _parameters(**values: object) -> str:
    """Return stable JSON suitable for a CSV registry cell."""

    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def mechanistic_registry() -> pd.DataFrame:
    """Return the locked R5 registry, independently for both gas targets."""

    grid = list(K_CANDIDATES)
    rows: list[dict[str, object]] = []
    definitions = (
        {
            "model_id": "M1F",
            "family": "EXOGENOUS_PROCESS_KINETIC",
            "load_basis": "FEED_WET_MASS",
            "kernel_type": "HRT_FREE_FIRST_ORDER_DAILY_BIN",
            "parameters": _parameters(
                k_candidates_d_inverse=grid,
                tail_tolerance=KERNEL_TAIL_TOL,
                min_kernel_coverage=MIN_KERNEL_COVERAGE,
                beta0_nonnegative=True,
                beta1_nonnegative=True,
            ),
            "target_history_use": "FALSE",
            "uses_target_history": False,
            "physical_status": "PROCESS_INFORMED_WET_FEED_PROXY",
            "availability": "AVAILABLE",
            "development_status": "LOCK_FOR_R7",
            "notes": "feed_AB_tpd is t wet mass/d; coefficient is not a biochemical yield",
        },
        {
            "model_id": "M1TS",
            "family": "EXOGENOUS_PROCESS_KINETIC",
            "load_basis": "TS_LOAD_PROXY",
            "kernel_type": "HRT_FREE_FIRST_ORDER_DAILY_BIN",
            "parameters": _parameters(
                k_candidates_d_inverse=grid,
                tail_tolerance=KERNEL_TAIL_TOL,
                min_kernel_coverage=MIN_KERNEL_COVERAGE,
                beta0_nonnegative=True,
                beta1_nonnegative=True,
            ),
            "target_history_use": "FALSE",
            "uses_target_history": False,
            "physical_status": "PROVISIONAL_PROCESS_PROXY",
            "availability": "AVAILABLE_WITH_DATE_ONLY_LAB_SUPPORT",
            "development_status": "LOCK_FOR_R7",
            "notes": "TS_load_tpd=feed_AB_tpd*acid_TS/100; lab reporting time remains provisional",
        },
        {
            "model_id": "M1VS",
            "family": "EXOGENOUS_PROCESS_KINETIC",
            "load_basis": "VS_LOAD_PROXY",
            "kernel_type": "HRT_FREE_FIRST_ORDER_DAILY_BIN",
            "parameters": _parameters(
                k_candidates_d_inverse=grid,
                tail_tolerance=KERNEL_TAIL_TOL,
                min_kernel_coverage=MIN_KERNEL_COVERAGE,
                beta0_nonnegative=True,
                beta1_nonnegative=True,
            ),
            "target_history_use": "FALSE",
            "uses_target_history": False,
            "physical_status": "REGIME_SENSITIVE",
            "availability": "AVAILABLE_WITH_DATE_ONLY_LAB_SUPPORT",
            "development_status": "RETAIN_AS_SENSITIVITY",
            "notes": "VS_2020_STRUCTURAL_BREAK=TRUE; never promoted as the primary core model in R5",
        },
        {
            "model_id": "M1TS2",
            "family": "OPTIONAL_MULTI_POOL",
            "load_basis": "TS_LOAD_PROXY",
            "kernel_type": "HRT_FREE_TWO_POOL_FIRST_ORDER_DAILY_BIN",
            "parameters": _parameters(
                k_candidates_d_inverse=grid,
                k_fast_gt_k_slow=True,
                minimum_rate_ratio=2.0,
                tail_tolerance=KERNEL_TAIL_TOL,
                min_kernel_coverage=MIN_KERNEL_COVERAGE,
                coefficients_nonnegative=True,
            ),
            "target_history_use": "FALSE",
            "uses_target_history": False,
            "physical_status": "OPTIONAL_COMPLEXITY",
            "availability": "AVAILABLE_WITH_DATE_ONLY_LAB_SUPPORT",
            "development_status": "CONDITIONAL_ON_NONDEGENERACY",
            "notes": "OOF gain, coefficient degeneracy, kernel separation, and fold stability required",
        },
        {
            "model_id": "M2",
            "family": "CONSTRAINED_TARGET_RESIDUAL",
            "load_basis": "BEST_ELIGIBLE_M1F_OR_M1TS",
            "kernel_type": "LOCKED_M1_PLUS_EXPONENTIAL_RESIDUAL_INERTIA",
            "parameters": _parameters(
                tau_res_candidates_d=list(TAU_RES_CANDIDATES),
                correction="e_t*exp(-h/tau_res)",
                free_gamma=False,
            ),
            "target_history_use": "TRUE_CONSTRAINED",
            "uses_target_history": True,
            "physical_status": "TARGET_HISTORY_DEPENDENT_SEPARATE_FAMILY",
            "availability": "REQUIRES_OBSERVED_TARGET_AT_ORIGIN",
            "development_status": "LOCK_FOR_R7_SEPARATE_FAMILY",
            "notes": "no free AR coefficient; unavailable when y(t) is missing",
        },
        {
            "model_id": "M3_COD",
            "family": "BLOCKED_PHYSICAL_BALANCE",
            "load_basis": "COD_MASS_LOAD_UNAVAILABLE",
            "kernel_type": "NOT_IMPLEMENTED",
            "parameters": _parameters(Q_feed_m3d="UNKNOWN"),
            "target_history_use": "FALSE",
            "uses_target_history": False,
            "physical_status": "BLOCKED_CURRENTLY",
            "availability": "BLOCKED",
            "development_status": "BLOCKED",
            "notes": "acid_CODcr is concentration; Q_feed is UNKNOWN so COD mass load cannot be formed",
        },
        {
            "model_id": "M3_HYDRAULIC",
            "family": "BLOCKED_HYDRAULIC",
            "load_basis": "HYDRAULIC_INPUTS_UNAVAILABLE",
            "kernel_type": "NOT_IMPLEMENTED",
            "parameters": _parameters(
                rho_feed="UNKNOWN",
                Q_feed="UNKNOWN",
                HRT_nominal="UNKNOWN",
                HRT_RTD="UNKNOWN",
                SRT_actual="UNKNOWN",
                V_active="UNKNOWN",
            ),
            "target_history_use": "FALSE",
            "uses_target_history": False,
            "physical_status": "BLOCKED_CURRENTLY",
            "availability": "BLOCKED",
            "development_status": "BLOCKED",
            "notes": "HRT_mass_proxy_d is DIAGNOSTIC_ONLY and is prohibited from the kinetic kernel",
        },
    )
    for definition in definitions:
        for target in TARGETS:
            rows.append({**definition, "target": target})
    columns = [
        "model_id",
        "family",
        "target",
        "load_basis",
        "kernel_type",
        "parameters",
        "target_history_use",
        "uses_target_history",
        "physical_status",
        "availability",
        "development_status",
        "notes",
    ]
    return pd.DataFrame(rows).loc[:, columns]

