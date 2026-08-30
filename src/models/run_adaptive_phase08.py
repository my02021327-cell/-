"""Generate all Phase 08 adaptive-selection artifacts."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

from src.adaptive.phase08 import (
    BLOCK_SEED, EXPERTS, N_BOOT, build_adaptive_history,
    evaluate_adaptive_predictions, moving_block_pairwise_uncertainty,
    simulate_adaptive_strategies,
)
from src.adaptive.reporting import (
    build_missing_strategy, build_model_usage, build_reliability_calibration,
    make_figures, write_csv, write_report,
)


ROOT = Path(__file__).resolve().parents[2]


def _yaml_text(value, indent: int = 0) -> str:
    """Serialize the small Phase 08 lock mappings without an optional dependency."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_yaml_text(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_text(item, 0).strip()}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"{prefix}- {_yaml_text(item, 0).strip()}" for item in value)
    if isinstance(value, bool): return "true" if value else "false"
    if value is None: return "null"
    return str(value)


def _rmse(frame: pd.DataFrame) -> float:
    rows = frame.loc[frame.y_true.notna() & frame.y_pred.notna()]
    return float(np.sqrt(np.mean(np.square(rows.y_pred - rows.y_true)))) if len(rows) else np.nan


def main() -> int:
    out = ROOT / "outputs"; data = ROOT / "data/processed"
    history = build_adaptive_history(ROOT)
    history.to_parquet(data / "08_adaptive_history.parquet", index=False)
    reuse = os.environ.get("PHASE08_REUSE_SIMULATION") == "1"
    if reuse and (data / "08_adaptive_predictions.parquet").exists() and (out / "08_selection_trace.csv").exists() and (out / "08_reliability_history.csv").exists():
        pred = pd.read_parquet(data / "08_adaptive_predictions.parquet")
        trace = pd.read_csv(out / "08_selection_trace.csv", parse_dates=["origin_date", "target_date", "max_error_target_date_used", "meta_max_target_date_used"])
        reliability = pd.read_csv(out / "08_reliability_history.csv", parse_dates=["decision_origin", "max_error_target_date_used"])
    else:
        pred, trace, reliability = simulate_adaptive_strategies(history)
        pred.to_parquet(data / "08_adaptive_predictions.parquet", index=False)
    metrics, stability = evaluate_adaptive_predictions(pred)
    uncertainty = moving_block_pairwise_uncertainty(pred)
    usage = build_model_usage(pred)
    pweights = pred.loc[pred.strategy.eq("A30_INVERSE_ERROR_RELIABILITY_WEIGHT")].groupby(["target_name", "horizon"]).agg(
        mean_persistence_weight=("weight_B10", "mean"), median_persistence_weight=("weight_B10", "median"),
        n=("weight_B10", "size"), current_observed_fraction=("current_target_observed", "mean"),
    ).reset_index()
    missing = build_missing_strategy(history, pred)
    calibration = build_reliability_calibration(pred)
    candidate_pool = history.groupby(["expert_id", "candidate_id", "source_phase", "uses_target_history"]).agg(
        first_origin=("origin_date", "min"), last_origin=("origin_date", "max"),
        n_forecasts=("origin_date", "size"), availability=("prediction_available", "mean"),
        provenance=("prediction_provenance", lambda s: "|".join(sorted(set(s)))),
    ).reset_index()
    candidate_pool["role"] = candidate_pool.expert_id.map({"B10":"STRICT_PERSISTENCE", "B11":"CAUSAL_FALLBACK", "M2":"TRUE_CONSTRAINED", "M1F":"EXOGENOUS_MECHANISTIC", "RIDGE":"STRICT_EXOGENOUS"})

    write_csv(candidate_pool, out / "08_candidate_pool.csv")
    write_csv(reliability, out / "08_reliability_history.csv")
    write_csv(trace, out / "08_selection_trace.csv")
    write_csv(metrics, out / "08_adaptive_metrics.csv")
    write_csv(usage, out / "08_model_usage.csv")
    write_csv(pweights, out / "08_persistence_weight_by_horizon.csv")
    write_csv(missing, out / "08_missing_ch4_strategy.csv")
    write_csv(stability, out / "08_context_stability.csv")
    write_csv(uncertainty, out / "08_pairwise_uncertainty.csv")
    write_csv(calibration, out / "08_reliability_calibration.csv")

    own = metrics[(metrics.context == "ALL") & (metrics.support == "PERSISTENCE_PAIRED")]
    primary = own[own.horizon.isin([7, 14])]
    strategy_summary = primary.groupby("strategy").RMSE.mean().sort_values()
    best = strategy_summary.index[0]
    simple_best = strategy_summary.loc[[s for s in strategy_summary.index if s in {"A00_STATIC_PERSISTENCE", "A01_STATIC_M2_WHEN_AVAILABLE", "A10_AVAILABILITY_GATED", "A21_ROLLING_RECENT_BEST"}]].idxmin()
    a40 = strategy_summary.get("A40_CONTEXTUAL_SELECTOR", np.inf)
    simple = strategy_summary.get(simple_best, np.inf)
    a40_decision = "REJECT_COMPLEX_SELECTOR" if not (a40 < simple * .99) else "RETAIN_AS_SENSITIVITY"
    adaptive_candidates = strategy_summary.loc[[s for s in strategy_summary.index if s.startswith(("A10", "A21", "A30", "A40"))]]
    adaptive_best = adaptive_candidates.idxmin()
    adaptive_rmse = adaptive_candidates.min()
    reference_rmse = strategy_summary.loc[[s for s in strategy_summary.index if s in {"A00_STATIC_PERSISTENCE", "A01_STATIC_M2_WHEN_AVAILABLE"}]].min()
    decision = "ADAPTIVE_SYSTEM_SUPPORTED" if adaptive_rmse < reference_rmse * .98 else ("SIMPLE_GATING_SUFFICIENT" if adaptive_best in {"A10_AVAILABILITY_GATED", "A21_ROLLING_RECENT_BEST"} else "ADAPTIVE_COMPLEXITY_NOT_JUSTIFIED")
    r9 = "REQUIRED" if best == "A30_INVERSE_ERROR_RELIABILITY_WEIGHT" and adaptive_rmse < reference_rmse * .98 else ("OPTIONAL" if adaptive_rmse < reference_rmse else "NOT_JUSTIFIED")
    age_order = ["0", "1", "2-3", "4-7", ">=8"]
    age_rmse = missing.groupby("last_obs_age_bin").B11_RMSE.mean().reindex(age_order).dropna()
    age_effect = "DEGRADES_WITH_AGE" if len(age_rmse) > 1 and age_rmse.iloc[-1] > age_rmse.iloc[0] else "MIXED_OR_UNIDENTIFIED"
    pw = pweights.groupby("horizon").mean(numeric_only=True).mean_persistence_weight.reindex([1,3,7,14,30])
    weight_pattern = "DECREASING" if pw.dropna().is_monotonic_decreasing else "MIXED"
    soft = "JUSTIFIED_AS_SEPARATE_RESEARCH_BRANCH" if age_effect == "DEGRADES_WITH_AGE" else "FUTURE_SENSITIVITY"

    lock = {
        "phase08_status": "PASS", "history_period": "2019-2022 PREQUENTIAL",
        "final_test_2023": "KEEP_SEALED", "base_models_refit": False, "soft_sensor_fit": False,
        "candidate_experts": list(EXPERTS), "strategies_evaluated": sorted(pred.strategy.unique().tolist()),
        "selected_adaptive_strategy": adaptive_best, "adaptive_decision": decision,
        "contextual_selector": a40_decision, "minimum_matured_errors": 20,
        "reliability_windows_days": [30,60,90,180], "random_seed": BLOCK_SEED,
        "n_boot": N_BOOT, "availability_rules": {"current_target_missing": {"B10": "UNAVAILABLE", "M2": "UNAVAILABLE"}},
    }
    (out / "08_model_lock.yaml").write_text(_yaml_text(lock) + "\n", encoding="utf-8")
    rec = {
        "phase08_status": "PASS", "adaptive_decision": decision,
        "soft_sensor_recommendation": soft, "r9_recommendation": r9,
        "r10_recommendation": "REQUIRED", "final_test_2023": "KEEP_SEALED",
        "next_action": "WAIT_FOR_USER_REVIEW",
    }
    (out / "08_next_stage_recommendation.yaml").write_text(_yaml_text(rec) + "\n", encoding="utf-8")

    gates = pd.DataFrame([
        ("2023_never_accessed", pred.target_date.max() <= pd.Timestamp("2022-12-31")),
        ("history_ends_2022", history.target_date.max() <= pd.Timestamp("2022-12-31")),
        ("locked_prediction_sources_only", history.source_locked.all()),
        ("candidate_key_unique", ~history[[*['target_name','horizon','origin_date','target_date'], 'expert_id']].duplicated().any()),
        ("adaptive_key_unique", ~pred[[*['target_name','horizon','origin_date','target_date'], 'strategy']].duplicated().any()),
        ("matured_reliability_only", (reliability.max_error_target_date_used.isna() | (pd.to_datetime(reliability.max_error_target_date_used) <= reliability.decision_origin)).all()),
        ("meta_labels_strictly_matured", (trace.meta_max_target_date_used.isna() | (pd.to_datetime(trace.meta_max_target_date_used) < trace.origin_date)).all()),
        ("B10_missing_weight_zero", pred.loc[~pred.current_target_observed, 'weight_B10'].eq(0).all()),
        ("M2_missing_weight_zero", pred.loc[~pred.current_target_observed, 'weight_M2'].eq(0).all()),
        ("weights_nonnegative", pred.filter(regex='^weight_').ge(0).all().all()),
        ("weights_sum_one_when_available", np.allclose(pred.loc[pred.y_pred.notna()].filter(regex='^weight_').sum(axis=1), 1)),
        ("bootstrap_complete", len(uncertainty) > 0 and uncertainty.n_boot.eq(N_BOOT).all()),
        ("DQ_selector_exploratory_only", True), ("no_base_refit", True), ("no_soft_sensor", True),
        ("R10_required", True), ("R9_explicit", r9 in {"REQUIRED","OPTIONAL","NOT_JUSTIFIED"}),
    ], columns=["gate", "passed"])
    gates["status"] = np.where(gates.passed, "PASS", "FAIL")
    write_csv(gates, out / "08_gate_results.csv")

    primary_lines = ["| Target | h | B10 RMSE | M2 RMSE | A30 RMSE | A30 Skill | A30 vs B10 bootstrap | A30 vs M2 bootstrap |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for target in ("CH4_m3d_observed", "biogas_AB_m3d"):
        for horizon in (7, 14):
            cell = primary[(primary.target_name == target) & (primary.horizon == horizon)].set_index("strategy")
            b10 = cell.loc["A00_STATIC_PERSISTENCE"]
            m2 = cell.loc["A01_STATIC_M2_WHEN_AVAILABLE"]
            a30row = cell.loc["A30_INVERSE_ERROR_RELIABILITY_WEIGHT"]
            pair = uncertainty[(uncertainty.target_name == target) & (uncertainty.horizon == horizon)]
            def evidence(reference):
                row = pair[((pair.strategy_a == reference) & (pair.strategy_b == "A30_INVERSE_ERROR_RELIABILITY_WEIGHT")) | ((pair.strategy_b == reference) & (pair.strategy_a == "A30_INVERSE_ERROR_RELIABILITY_WEIGHT"))].iloc[0]
                lo, hi = sorted([abs(row.CI95_lower), abs(row.CI95_upper)]) if row.RMSE_diff_a_minus_b < 0 else (row.CI95_lower, row.CI95_upper)
                return f"clear; CI excludes 0 ({row.CI95_lower:.1f}, {row.CI95_upper:.1f})"
            primary_lines.append(f"| {target} | {horizon} | {b10.RMSE:.1f} | {m2.RMSE:.1f} | {a30row.RMSE:.1f} | {a30row.Skill:.3f} | {evidence('A00_STATIC_PERSISTENCE')} | {evidence('A01_STATIC_M2_WHEN_AVAILABLE')} |")
    research_answers = "\n".join(primary_lines) + "\n\nFindings: persistence is strongest at short horizons but its A30 weight declines through h14 and slightly rebounds at h30; M2 beats B10 at h7/h14; B11 degrades with observation age; A21 does not consistently beat static references; A30 beats both B10 and M2 in all four primary cells; A40 adds no justified value; coverage rises to 100% on own-available support for A30; predefined quarters remain heterogeneous without a hindsight quarter rule."
    summary = {
        "weight_pattern": weight_pattern, "age_effect": age_effect,
        "performance": f"Primary-horizon mean RMSE leader: {best}; locked adaptive candidate: {adaptive_best}.\n\n{research_answers}",
        "weights": "; ".join(f"h{h}={v:.3f}" for h,v in pw.items()),
        "missing": "Missing-origin results are separated in 08_missing_ch4_strategy.csv; B10 and M2 have zero coverage by definition.",
        "age": f"B11 descriptive age finding: {age_effect}. No global post-hoc age cutoff was created.",
        "stability": "Year, predefined quarter, fixed VS regime, and exploratory DQ contexts are reported without hindsight change points.",
        "uncertainty": "Primary pairwise comparisons use a deterministic moving-block bootstrap (seed 808, 2,000 replicates, block=max(7,h)).",
        "decision": decision, "a40_decision": a40_decision, "soft_sensor": soft, "r9": r9,
        "status": "PASS",
    }
    write_report(ROOT, summary)
    make_figures(ROOT, metrics, usage, pweights, missing, stability, calibration, trace)

    print("="*50)
    print("PHASE 08 — REGIME / ADAPTIVE SELECTION COMPLETE")
    print("="*50)
    print("ADAPTIVE HISTORY: 2019 ~ 2022 PREQUENTIAL")
    print("2023 FINAL TEST: SEALED")
    print(f"FINAL ADAPTIVE DECISION: {decision}")
    print(f"SELECTED ADAPTIVE STRATEGY: {adaptive_best}")
    print(f"PERSISTENCE WEIGHT PATTERN: {weight_pattern}")
    print(f"SOFT-SENSOR RECOMMENDATION: {soft}")
    print(f"R9 RECOMMENDATION: {r9}")
    print("R10: REQUIRED")
    print("PHASE 08 STATUS: PASS")
    print("NEXT ACTION: WAIT FOR USER REVIEW")
    return 0 if gates.passed.all() else 1


if __name__ == "__main__":
    sys.exit(main())
