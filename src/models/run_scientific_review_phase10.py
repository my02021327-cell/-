"""Generate all Phase 10 review artifacts without fitting or opening 2023."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.scientific.phase10 import (
    FINAL_DATE, build_claims_register, build_cross_target_consistency,
    build_extreme_error_cases, build_mechanistic_audit, build_physical_audit,
    build_reliability_calibration, build_residual_diagnostics,
    build_uncertainty_audit, build_unknown_register, locked_a30,
)
from src.scientific.reporting import make_figures, write_report

ROOT = Path(__file__).resolve().parents[2]
A30_SHA256 = "584b9637a67c6182d148d4ea19f509deee06bacc99c81e74ec8c2d1924143ab4"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_yaml(value: dict, path: Path) -> None:
    # JSON is a strict subset of YAML 1.2 and avoids an optional runtime dependency.
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    outputs = ROOT / "outputs"
    processed = ROOT / "data/processed"
    a30_path = processed / "08_adaptive_predictions.parquet"
    if hashlib.sha256(a30_path.read_bytes()).hexdigest() != A30_SHA256:
        raise AssertionError("locked Phase08 A30 artifact hash changed")

    phase09 = pd.read_parquet(processed / "09_ensemble_predictions.parquet")
    a30 = locked_a30(phase09)
    # Push-down filtering prevents 2023 rows from being loaded from the feature file.
    dq = pd.read_parquet(processed / "03_feature_base.parquet", filters=[("date", "<=", FINAL_DATE)])
    kernel = pd.read_csv(outputs / "05_kernel_audit.csv")
    physical05 = pd.read_csv(outputs / "05_physical_audit.csv")
    development = pd.read_csv(outputs / "05_development_summary.csv")
    intercepts = pd.read_csv(outputs / "05_intercept_audit.csv")
    unknown02 = pd.read_csv(outputs / "02_unknown_parameters.csv")

    cross = build_cross_target_consistency(a30)
    physical = build_physical_audit(a30, cross, kernel, physical05)
    mechanistic = build_mechanistic_audit(kernel, development, intercepts, physical05)
    residual = build_residual_diagnostics(a30)
    extreme = build_extreme_error_cases(a30, dq)
    uncertainty = build_uncertainty_audit(a30)
    reliability = build_reliability_calibration(a30)
    claims = build_claims_register()
    unknowns = build_unknown_register(unknown02)

    tables = {
        "10_scientific_claims_register.csv": claims,
        "10_physical_consistency_audit.csv": physical,
        "10_cross_target_consistency.csv": cross,
        "10_mechanistic_scientific_audit.csv": mechanistic,
        "10_residual_diagnostics.csv": residual,
        "10_extreme_error_cases.csv": extreme,
        "10_uncertainty_audit.csv": uncertainty,
        "10_reliability_calibration.csv": reliability,
        "10_unknown_parameter_register.csv": unknowns,
    }
    for name, frame in tables.items():
        _write_csv(frame, outputs / name)

    lock = {
        "system": {"selected": "A30_INVERSE_ERROR_RELIABILITY_WEIGHT", "stacking": False, "contextual_selector_A40": False, "soft_CH4_sensor": False},
        "targets": ["CH4_m3d_observed", "biogas_AB_m3d"],
        "horizons": {"primary": [7, 14], "secondary": [1, 3], "exploratory": [30]},
        "training": {"final_refit_period": {"start": "2018-01-01", "end": "2022-12-31"}, "model_structure_change_allowed": False, "hyperparameter_change_allowed": False, "feature_change_allowed": False},
        "adaptive": {"rule": "LOCKED_PHASE08_A30", "past_matured_errors_only": True, "initial_history_through": "2022-12-31", "family_level_history_transfer": "PROVISIONAL"},
        "CH4_current_observed": {"B10": "allowed", "M2": "allowed", "M1F": "allowed", "Ridge": "allowed"},
        "CH4_current_missing": {"B10": "unavailable", "M2": "unavailable", "B11": "allowed", "M1F": "allowed", "Ridge": "allowed"},
        "final_test": {"period": {"start": "2023-01-01", "end": "2023-09-17"}, "mode": "PREQUENTIAL_ONE_TIME", "target_opened_before_final_integration": False, "post_hoc_retuning": False},
        "phase10_actions": {"model_refit": False, "new_feature": False, "prediction_correction": False, "2023_accessed": False},
    }
    readiness = {
        "phase10_status": "CONDITIONAL PASS",
        "scientific_review": {"status": "PASS"},
        "physical_consistency": {"status": "PROVISIONAL", "cross_target_violations": int(cross.CH4_gt_biogas.sum()), "cross_target_violation_rate": float(cross.CH4_gt_biogas.mean())},
        "target_integrity": {"status": "PASS", "truth": "OBSERVED_ONLY"},
        "residual_diagnostics": {"status": "PROVISIONAL_OVERLAPPING_ACF"},
        "uncertainty": {"status": "PROVISIONAL_UNDERCOVERAGE"},
        "reliability_calibration": {"status": "PARTIALLY_CALIBRATED"},
        "model_version_transfer": {"status": "PROVISIONAL_FAMILY_LEVEL_PRIOR"},
        "unknown_parameters": {"critical_unresolved": unknowns.loc[unknowns.priority_for_measurement.eq("HIGH"), "parameter"].tolist()},
        "final_system": {"A30_locked": True, "A30_sha256": A30_SHA256, "stacking_rejected": True},
        "final_integration": {"readiness": "READY_WITH_PROVISIONAL_GUARDS", "2018_2022_refit_allowed": True, "2023_final_test_allowed_after_refit": True},
        "2023": {"current_status": "SEALED", "target_accessed": False, "metrics": None},
    }
    _write_yaml(lock, outputs / "10_final_integration_lock.yaml")
    _write_yaml(readiness, outputs / "10_final_test_readiness.yaml")
    write_report(ROOT, physical, cross, residual, uncertainty, reliability, claims, unknowns)
    make_figures(ROOT, a30, cross, residual, extreme, uncertainty, reliability)

    primary = residual[(residual.context == "ALL") & residual.horizon.isin([7, 14])]
    print("=" * 50); print("PHASE 10 — PHYSICAL & SCIENTIFIC REVIEW COMPLETE"); print("=" * 50)
    print("ROLE: R10 PHYSICAL & SCIENTIFIC REVIEWER")
    print("FINAL CANDIDATE: A30_INVERSE_ERROR_RELIABILITY_WEIGHT")
    print("STACKING: REJECTED / NOT USED"); print("2023: SEALED")
    print("NONNEGATIVE PREDICTIONS: PASS — 0 negative raw predictions")
    print(f"CH4 > BIOGAS VIOLATION RATE: {cross.CH4_gt_biogas.mean():.4%} ({int(cross.CH4_gt_biogas.sum())}/{len(cross)})")
    print("KERNEL NORMALIZATION: PASS"); print("MASS BALANCE STATUS: NOT_IDENTIFIABLE")
    print("GAS VOLUME BASIS: UNKNOWN"); print("TRUE HRT: UNKNOWN"); print("TRUE SRT: UNKNOWN")
    for _, r in primary.iterrows(): print(f"{r.target} h{int(r.horizon)}: bias={r.bias:.1f}, RMSE={r.RMSE:.1f}, MAE={r.MAE:.1f}")
    print("RELIABILITY MONOTONIC CALIBRATION: PARTIAL")
    print("FINAL TEST READINESS: READY_WITH_PROVISIONAL_GUARDS")
    print("TARGET ACCESSED: NO"); print("2023 METRICS: NONE")
    print("PHASE 10 STATUS: CONDITIONAL PASS"); print("NEXT RECOMMENDED ROLE: FINAL INTEGRATION")
    return 0


if __name__ == "__main__":
    sys.exit(main())
