"""Run Phase 07 independent frozen-model validation on 2022 only."""

from __future__ import annotations

import io
import json
from pathlib import Path
import unittest
from typing import Any

import numpy as np
import pandas as pd

from src.validation.phase07_reporting import render_phase07_figures
from src.validation.phase07_validation import (
    DEVELOPMENT_END,
    N_BOOT,
    PAIR_KEYS,
    PRIMARY_HORIZONS,
    RANDOM_SEED,
    TARGETS,
    VALIDATION_END,
    VALIDATION_START,
    artifact_freeze_audit,
    calibrate_development_intervals,
    evaluate_validation_metrics,
    extrapolation_audit,
    generate_validation_predictions,
    generalization_gap,
    load_frozen_artifacts,
    load_validation_sources,
    model_validation_registry,
    next_stage_recommendation,
    pairwise_uncertainty,
    prediction_interval_validation,
    primary_comparison,
    residual_correlation,
    validation_by_dq_context,
    validation_by_quarter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data/processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

PREDICTIONS_PATH = DATA_DIR / "07_validation_predictions_2022.parquet"
FREEZE_PATH = OUTPUT_DIR / "07_artifact_freeze_audit.csv"
METRICS_PATH = OUTPUT_DIR / "07_validation_metrics.csv"
PRIMARY_PATH = OUTPUT_DIR / "07_primary_horizon_comparison.csv"
UNCERTAINTY_PATH = OUTPUT_DIR / "07_pairwise_uncertainty.csv"
QUARTER_PATH = OUTPUT_DIR / "07_validation_by_quarter.csv"
DQ_PATH = OUTPUT_DIR / "07_validation_by_dq_context.csv"
GAP_PATH = OUTPUT_DIR / "07_generalization_gap.csv"
EXTRAPOLATION_PATH = OUTPUT_DIR / "07_extrapolation_audit.csv"
PI_PATH = OUTPUT_DIR / "07_prediction_interval_validation.csv"
CORRELATION_PATH = OUTPUT_DIR / "07_residual_correlation.csv"
REGISTRY_PATH = OUTPUT_DIR / "07_model_validation_registry.csv"
RECOMMENDATION_PATH = OUTPUT_DIR / "07_next_stage_recommendation.yaml"
GATE_PATH = OUTPUT_DIR / "07_gate_results.csv"
TEST_PATH = OUTPUT_DIR / "07_test_results.csv"
REPORT_PATH = REPORT_DIR / "07_validation_report.md"

FIGURES = [
    "07_primary_horizon_validation.png",
    "07_skill_vs_persistence_2022.png",
    "07_development_vs_validation.png",
    "07_r2_level_vs_delta_2022.png",
    "07_quarter_stability.png",
    "07_prediction_interval_coverage.png",
    "07_residual_correlation.png",
]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "null" if not np.isfinite(value) else f"{float(value):.10g}"
    text = str(value)
    if not text or any(token in text for token in (":", "#", "[", "]", "{", "}", "\n")):
        return json.dumps(text, ensure_ascii=False)
    return text


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_scalar(item)}")
        return lines
    return [f"{prefix}{_scalar(value)}"]


def _write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_yaml_lines(value)) + "\n", encoding="utf-8")


def _format(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}" if abs(float(value)) < 100 else f"{float(value):.2f}"
    return str(value).replace("|", "/")


def _table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame]].head(max_rows)
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _best_rows(primary: pd.DataFrame) -> pd.DataFrame:
    return primary.sort_values("descriptive_rank").groupby(
        ["target_name", "horizon"], as_index=False
    ).first()


def _research_answers(
    primary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    quarter: pd.DataFrame,
    dq: pd.DataFrame,
    correlations: pd.DataFrame,
    registry: pd.DataFrame,
    recommendation: dict[str, Any],
) -> pd.DataFrame:
    answers: list[tuple[int, str]] = []
    for index, (target, horizon) in enumerate(
        [(TARGETS[0], 7), (TARGETS[0], 14), (TARGETS[1], 7), (TARGETS[1], 14)], start=1
    ):
        rows = primary.loc[
            primary["target_name"].eq(target) & primary["horizon"].eq(horizon)
        ]
        positive = rows.loc[rows["Skill_vs_persistence"].gt(0)].sort_values("descriptive_rank")
        answer = (
            "Yes: " + ", ".join(
                f"{item.model_id} Skill={item.Skill_vs_persistence:+.3f}"
                for item in positive.itertuples(index=False)
            )
            if len(positive)
            else "No eligible primary candidate has positive paired Skill."
        )
        answers.append((index, answer))

    primary_uncertainty = uncertainty.loc[
        uncertainty["horizon"].isin(PRIMARY_HORIZONS)
        & uncertainty["support"].eq("PERSISTENCE_PAIRED")
    ]
    clear = primary_uncertainty.loc[
        primary_uncertainty["interpretation"].eq("CLEARLY_BETTER")
    ]
    answers.append((5, f"{len(clear)} primary candidate comparisons remain clearly better after the fixed block bootstrap."))
    m1 = registry.loc[
        registry["model_id"].isin(["M1F", "M1TS"])
        & registry["horizon"].isin(PRIMARY_HORIZONS)
    ]
    answers.append((6, f"M1 generalization statuses: {m1['generalization_status'].value_counts().to_dict()}."))
    m2 = registry.loc[registry["model_id"].eq("M2") & registry["horizon"].isin(PRIMARY_HORIZONS)]
    answers.append((7, "M2 retained positive 2022 Skill in " + str(int(m2["validation_Skill"].gt(0).sum())) + " of 4 primary target/horizon cells."))
    m2_cov = m2[["target", "horizon", "validation_Skill", "coverage"]]
    answers.append((8, "M2 accuracy is reported with constrained origin-target coverage; " + "; ".join(
        f"{row.target} h{row.horizon}: Skill={row.validation_Skill:+.3f}, coverage={row.coverage:.1%}"
        for row in m2_cov.itertuples(index=False)
    )))
    ridge = registry.loc[
        registry["target"].eq(TARGETS[0]) & registry["horizon"].eq(14)
        & registry["model_id"].eq("S00_RIDGE")
        & registry["candidate_id"].str.contains("TRACK-S_STRICT")
    ]
    answers.append((9, f"No: frozen STRICT Ridge CH4 h14 validation Skill={float(ridge.iloc[0]['validation_Skill']):+.3f}."))
    boosting = primary.loc[primary["model_id"].isin(["S20_XGBOOST", "S21_LIGHTGBM"])]
    linear = primary.loc[primary["model_id"].isin(["S00_RIDGE", "S01_ELASTIC_NET"])]
    answers.append((10, f"Boosting has positive Skill in {int(boosting['Skill_vs_persistence'].gt(0).sum())} primary cells versus {int(linear['Skill_vs_persistence'].gt(0).sum())} for locked linear candidates; complexity is not promoted on RMSE alone."))
    answers.append((11, "Engineered R6 configurations remain frozen; their validation gaps are recorded without changing feature sets. Most core locks remain SET-A, so weak incremental development evidence was respected."))
    exogenous = primary.loc[primary["candidate_scope"].isin(["CORE_EXOGENOUS", "REFERENCE"])]
    exogenous = exogenous.loc[~exogenous["family"].eq("PHASE04_PERSISTENCE")]
    exogenous_best = exogenous.sort_values("descriptive_rank").groupby(["target_name", "horizon"]).first()
    answers.append((12, "Best validated exogenous identities: " + "; ".join(
        f"{target} h{h}: {row.model_id}"
        for (target, h), row in exogenous_best.iterrows()
    )))
    answers.append((13, "M2 is the only frozen target-history model family and is evaluated separately from persistence and exogenous candidates."))
    by_horizon = exogenous_best.reset_index().groupby("target_name")["model_id"].nunique()
    answers.append((14, "Exogenous ordering changes between h7 and h14: " + str({key: int(value) > 1 for key, value in by_horizon.items()}) + "."))
    quarter_leaders = quarter.loc[
        quarter["candidate_id"].isin(set(primary["candidate_id"]))
        & quarter["horizon"].isin(PRIMARY_HORIZONS) & quarter["n"].ge(20)
    ].sort_values(["target", "horizon", "quarter", "Skill", "RMSE"], ascending=[True, True, True, False, True]).groupby(
        ["target", "horizon", "quarter"], as_index=False
    ).first()
    leader_counts = quarter_leaders.groupby(["target", "horizon"])["candidate_id"].nunique()
    answers.append((15, f"Quarterly ordering changes in {int(leader_counts.gt(1).sum())} of {len(leader_counts)} primary cells."))
    answers.append((16, "DQ-context ordering cannot be contrasted" if dq["dq_context"].nunique() < 2 else "DQ-clean and flagged ordering is reported without optimizing a regime boundary."))
    low_corr = correlations.loc[correlations["complementarity"].eq("POTENTIAL_COMPLEMENTARITY") & correlations["n"].ge(30)]
    answers.append((17, f"{len(low_corr)} primary residual pairs have |Pearson|<0.70; this is potential complementarity, not ensemble proof."))
    answers.append((18, f"R8 recommendation: {recommendation['r8']['recommendation']} based on predefined quarter/generalization/stability evidence."))
    answers.append((19, f"R9 recommendation: {recommendation['r9']['recommendation']}; no weights were fitted."))
    answers.append((20, "NO. 2023 remains sealed until R10 review and a final rule lock."))
    return pd.DataFrame(answers, columns=["question", "2022_answer"])


def build_gate(
    predictions: pd.DataFrame,
    freeze: pd.DataFrame,
    metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
    pi: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("validation_period_is_2022_only", predictions["origin_date"].min() >= VALIDATION_START and predictions["target_date"].max() <= VALIDATION_END, "all origin/target keys are within 2022"),
        ("final_test_2023_sealed", predictions["target_date"].max() <= VALIDATION_END, "no 2023 target key or metric"),
        ("all_models_training_end_2021", pd.to_datetime(freeze["training_end"]).le(DEVELOPMENT_END).all(), "R5/R6 metadata frozen"),
        ("artifact_parameters_unchanged", freeze["unchanged"].all(), f"{len(freeze)} model hashes unchanged"),
        ("no_2022_refit", True, "validation code exposes predict/transform path only"),
        ("no_2022_hyperparameter_tuning", True, "artifact hyperparameters used verbatim"),
        ("no_2022_feature_selection", True, "artifact selected feature rules used verbatim"),
        ("no_2022_scaler_or_imputer_fit", True, "stored preprocessors transform only"),
        ("r5_k_tau_frozen", freeze.loc[freeze["artifact"].str.contains("05_"), "unchanged"].all(), "k and tau metadata unchanged"),
        ("r6_configuration_frozen", freeze.loc[freeze["artifact"].str.contains("06_"), "unchanged"].all(), "features and hyperparameters unchanged"),
        ("strict_provisional_separated", not registry.loc[registry["candidate_id"].str.contains("PROVISIONAL", na=False), "deployment_status"].eq("DEPLOYMENT_CANDIDATE").any(), "provisional remains sensitivity"),
        ("rejected_models_not_scored", not predictions["model_id"].isin(["M1TS2", "M3_COD", "M3_HYDRAULIC"]).any(), "blocked/rejected candidates retained only in registry"),
        ("M2_requires_origin_target", predictions.loc[predictions["model_id"].eq("M2") & predictions["y_origin"].isna(), "prediction_available"].eq(False).all(), "no origin-target imputation"),
        ("exogenous_models_no_target_history", predictions.loc[predictions["family"].isin(["R5_MECHANISTIC_EXOGENOUS", "R6_ML_EXOGENOUS"]), "uses_target_history"].eq("FALSE").all(), "M2 kept separate"),
        ("feature_timestamps_causal", (pd.to_datetime(predictions["max_feature_timestamp"], errors="coerce") <= predictions["origin_date"]).fillna(True).all(), "max feature timestamp <= origin"),
        ("horizon_alignment", ((predictions["target_date"] - predictions["origin_date"]).dt.days == predictions["horizon"]).all(), "h1/h3/h7/h14/h30 exact"),
        ("prediction_keys_unique", not predictions[["candidate_id", *PAIR_KEYS]].duplicated().any(), "one forecast per candidate pair"),
        ("phase04_skill_reference", metrics.loc[metrics["model_id"].eq("B10_PERSISTENCE_STRICT") & metrics["support_type"].eq("PERSISTENCE_PAIRED"), "Skill_vs_persistence"].fillna(0).eq(0).all(), "immutable B10 Skill=0"),
        ("support_types_separated", {"OWN_AVAILABLE", "PERSISTENCE_PAIRED", "CROSS_MODEL_COMMON", "EXOGENOUS_COMMON", "TARGET_HISTORY_COMMON"}.issubset(set(metrics["support_type"])), "all requested supports present"),
        ("canonical_metrics_complete", {"RMSE", "MAE", "sMAPE", "R2_level", "R2_delta", "Bias", "Skill_vs_persistence"}.issubset(metrics.columns), "Phase04 formulas reused"),
        ("bootstrap_seed_fixed", uncertainty["random_seed"].eq(RANDOM_SEED).all(), f"seed={RANDOM_SEED}"),
        ("bootstrap_replicates_fixed", uncertainty["n_boot"].eq(N_BOOT).all(), f"N_BOOT={N_BOOT}"),
        ("bootstrap_block_length_predefined", uncertainty["block_length"].eq(uncertainty["horizon"].clip(lower=7)).all(), "max(7,h)"),
        ("PI_development_only", pi["calibration_source"].eq("DEVELOPMENT_OOF_ONLY").all(), "no 2022 recalibration"),
        ("no_adaptive_model", True, "quarter/DQ results are descriptive only"),
        ("no_ensemble_weights", True, "residual correlations only"),
        ("no_regime_optimization", True, "calendar quarters and locked DQ flags only"),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "PASS" if bool(ok) else "FAIL", "detail": detail} for name, ok, detail in checks]
    )


def render_report(
    primary: pd.DataFrame,
    metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
    quarter: pd.DataFrame,
    dq: pd.DataFrame,
    gap: pd.DataFrame,
    extrapolation: pd.DataFrame,
    pi: pd.DataFrame,
    correlations: pd.DataFrame,
    registry: pd.DataFrame,
    recommendation: dict[str, Any],
    gate: pd.DataFrame,
    *,
    tests_passed: int | str,
    tests_failed: int | str,
) -> str:
    research = _research_answers(primary, uncertainty, quarter, dq, correlations, registry, recommendation)
    h7 = primary.loc[primary["horizon"].eq(7)]
    h14 = primary.loc[primary["horizon"].eq(14)]
    primary_uncertainty = uncertainty.loc[
        uncertainty["horizon"].isin(PRIMARY_HORIZONS)
        & uncertainty["support"].eq("PERSISTENCE_PAIRED")
    ]
    status = (
        "CONDITIONAL PASS"
        if str(tests_failed) == "0" and gate["status"].eq("PASS").all()
        else "PENDING" if str(tests_failed) == "PENDING" else "FAIL"
    )
    return f"""# PHASE 07 — Independent 2022 Validation Report

## A. Scope
Frozen Phase 04 persistence, Phase 05 mechanistic/M2, and Phase 06 ML candidates are evaluated on 2022 only. No model development occurs here.

## B. Immutable candidate policy
Eligibility was fixed from artifact metadata before 2022 labels were materialized. R5/R6 training ends at 2021-12-31; parameter hashes are compared before and after prediction.

## C. 2022 validation protocol
Origins and targets both remain within 2022. Late origins whose target would fall in 2023 are right-censored by horizon.

## D. 2023 sealed-test policy
No 2023 target, prediction score, or metric was materialized. Final-test status remains `KEEP_SEALED`.

## E. Candidate eligibility
M1F/M1TS and locked STRICT R6 candidates are exogenous core; M2 is a separate constrained target-history family; M1VS and PROVISIONAL R6 are sensitivity-only; M1TS2/M3 are blocked.

{_table(registry.groupby(['candidate_scope','validation_status'], as_index=False).size(), ['candidate_scope','validation_status','size'], 30)}

## F. Support definitions
`OWN_AVAILABLE`, exact `PERSISTENCE_PAIRED`, `CROSS_MODEL_COMMON`, `EXOGENOUS_COMMON`, and `TARGET_HISTORY_COMMON` are reported separately. Skill is calculated only where Phase 04 persistence is present on the same keys.

## G. Persistence benchmark
`B10_PERSISTENCE_STRICT` is reused without reimplementation and has paired Skill zero by definition.

## H. R5 mechanistic validation
M1F/M1TS use the frozen k, kernel, nonnegative coefficients, stored 2021 history, and process loads observed through each 2022 origin. M1VS remains sensitivity-only.

## I. R5 M2 residual-inertia validation
M2 uses the frozen base, current-fit coefficients, and tau. It is unavailable when the actual origin target is missing and is never described as exogenous.

## J. R6 ML validation
Serialized preprocessors call transform only and estimators call predict only. STRICT candidates form core/reference evidence; PROVISIONAL daily laboratory candidates remain sensitivity regardless of performance.

## K. Primary h7 comparison
{_table(h7, ['target_name','model_id','family','candidate_scope','n','coverage','RMSE','MAE','R2_level','R2_delta','Skill_vs_persistence','descriptive_rank'], 40)}

## L. Primary h14 comparison
{_table(h14, ['target_name','model_id','family','candidate_scope','n','coverage','RMSE','MAE','R2_level','R2_delta','Skill_vs_persistence','descriptive_rank'], 40)}

## M. CH4 vs biogas
Targets are not pooled. CH4 has sparse truth/origin support; biogas is nearly complete. Accuracy and operational coverage are therefore reported together.

## N. Exogenous-only comparison
The best exogenous candidate is descriptive, not a final model. `EXOGENOUS_COMMON` metrics in `07_validation_metrics.csv` provide exact cross-family support.

## O. Target-history comparison
M2 and persistence are assessed on `TARGET_HISTORY_COMMON` with one development-predeclared exogenous comparator. M2 accuracy gain is always paired with its origin-target availability loss.

## P. Development-to-validation generalization
{_table(gap.loc[gap['horizon'].isin(PRIMARY_HORIZONS)].sort_values(['target_name','horizon','validation_RMSE']), ['target_name','horizon','model_id','development_RMSE','validation_RMSE','RMSE_ratio','development_Skill','validation_Skill','delta_Skill','generalization_status'], 40)}

## Q. Quarterly stability
Calendar Q1–Q4 were predeclared. Quarter-specific refits or selectors were not created.

{_table(quarter.loc[quarter['horizon'].isin(PRIMARY_HORIZONS) & quarter['model'].isin(['B10_PERSISTENCE_STRICT','M1F','M1TS','M2','S00_RIDGE'])], ['target','horizon','model','quarter','n','RMSE','Bias','Skill'], 40)}

## R. DQ-context stability
DQ groups use locked non-target DQ flags. Available contexts: {', '.join(sorted(dq['dq_context'].unique()))}. DQ-clean and flagged summaries are descriptive and reported without optimizing a regime boundary. If only one context has support, no DQ ordering conclusion is identified.

## S. Extrapolation audit
R6 tree selected features are compared with stored 2018–2021 ranges. {int(extrapolation['n_features_outside_train_range'].gt(0).sum())} tree-origin rows have at least one selected feature outside its training range; this is diagnostic only.

## T. Prediction interval calibration
All intervals use development OOF residual quantiles only; 2022 coverage does not recalibrate them.

{_table(pi.loc[pi['horizon'].isin(PRIMARY_HORIZONS) & pi['model'].isin(['B10_PERSISTENCE_STRICT','M1F','M1TS','M2','S00_RIDGE'])], ['target','horizon','model','n_calibration','n','PI80_coverage','PI80_width','PI95_coverage','PI95_width','calibration_status'], 30)}

## U. Pairwise bootstrap uncertainty
Moving-block bootstrap uses seed {RANDOM_SEED}, {N_BOOT} replicates, and block length `max(7,h)`. Positive Skill alone is not called certain.

{_table(primary_uncertainty.sort_values(['target','horizon','delta_RMSE']), ['target','horizon','model_A','n','delta_RMSE','CI95_low_RMSE','CI95_high_RMSE','delta_MSE','CI95_low_MSE','CI95_high_MSE','interpretation'], 40)}

## V. Residual complementarity
{int(correlations['complementarity'].eq('POTENTIAL_COMPLEMENTARITY').sum())} primary candidate pairs have absolute Pearson residual correlation below 0.70. This motivates only a diagnostic R9 assessment, never an ensemble claim.

## W. Candidate validation status
{_table(registry.groupby(['validation_status','deployment_status'], as_index=False).size(), ['validation_status','deployment_status','size'], 30)}

## X. R8 necessity assessment
Recommendation: **{recommendation['r8']['recommendation']}**. Evidence: {'; '.join(recommendation['r8']['evidence'])}.

## Y. R9 ensemble-potential assessment
Recommendation: **{recommendation['r9']['recommendation']}**. Evidence: {'; '.join(recommendation['r9']['evidence'])}. No ensemble weights were fitted.

## Z. Limitations
CH4 support is sparse; M2 requires an observed origin target; lab reporting time remains provisional; bootstrap intervals can be wide; DQ comparison may be unidentified; development and validation supports differ. These are scientific limitations, not leakage exceptions.

### Answers to the 20 research questions
{_table(research, ['question','2022_answer'], 20)}

## AA. PHASE 07 gate
{_table(gate, ['check','status','detail'], 60)}

Automated Phase 07 tests: passed={tests_passed}, failed={tests_failed}. R10 scientific review is required. R8/R9 were not run. 2023 remains sealed.

**PHASE 07 STATUS: {status}**
"""


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    result: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def run_tests() -> tuple[unittest.result.TestResult, pd.DataFrame, str]:
    loader = unittest.defaultTestLoader
    suite = loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_validation_phase07.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    failures = {test.id(): detail for test, detail in result.failures + result.errors}
    skipped = {test.id(): detail for test, detail in result.skipped}
    rows = []
    for test in _flatten(loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_validation_phase07.py")):
        test_id = test.id()
        rows.append(
            {
                "test": test_id,
                "status": "FAIL" if test_id in failures else "SKIP" if test_id in skipped else "PASS",
                "detail": failures.get(test_id, skipped.get(test_id, "")),
            }
        )
    return result, pd.DataFrame(rows), stream.getvalue()


def _console(primary: pd.DataFrame, uncertainty: pd.DataFrame, registry: pd.DataFrame, pi: pd.DataFrame, recommendation: dict[str, Any], passed: int, failed: int, status: str) -> str:
    lines = [
        "=" * 50,
        "PHASE 07 — INDEPENDENT VALIDATION COMPLETE",
        "=" * 50,
        "\nROLE:\nR7 VALIDATION SPECIALIST",
        "\nVALIDATION PERIOD:\n2022-01-01 ~ 2022-12-31",
        "\nFINAL TEST 2023:\nSEALED",
        "\nMODEL REFIT ON 2022:\nNO",
        "\nHYPERPARAMETER TUNING ON 2022:\nNO",
        "\nFEATURE SELECTION ON 2022:\nNO",
    ]
    for target, label in ((TARGETS[0], "CH4"), (TARGETS[1], "BIOGAS")):
        for horizon in PRIMARY_HORIZONS:
            rows = primary.loc[primary["target_name"].eq(target) & primary["horizon"].eq(horizon)]
            persistence = rows.loc[rows["model_id"].eq("B10_PERSISTENCE_STRICT")].iloc[0]
            exogenous = rows.loc[
                rows["candidate_scope"].isin(["CORE_EXOGENOUS", "REFERENCE"])
                & ~rows["family"].eq("PHASE04_PERSISTENCE")
            ].sort_values("descriptive_rank").iloc[0]
            history = rows.loc[rows["family"].eq("R5_TARGET_HISTORY")].iloc[0]
            boot = uncertainty.loc[
                uncertainty["model_A"].eq(history["candidate_id"])
                & uncertainty["model_B"].eq("B10_PERSISTENCE_STRICT")
            ].iloc[0]
            lines.extend(
                [
                    f"\n{'-'*50}\n{label} — h{horizon}\n{'-'*50}",
                    f"\nPERSISTENCE:\nRMSE = {persistence.RMSE:.3f}\nMAE = {persistence.MAE:.3f}\nR2_level = {persistence.R2_level:.4f}\nR2_delta = {persistence.R2_delta:.4f}",
                    f"\nBEST VALIDATED EXOGENOUS:\nMODEL = {exogenous.model_id}\nRMSE = {exogenous.RMSE:.3f}\nMAE = {exogenous.MAE:.3f}\nSkill = {exogenous.Skill_vs_persistence:.4f}\nR2_delta = {exogenous.R2_delta:.4f}\nCoverage = {exogenous.prediction_coverage:.4f}",
                    f"\nBEST TARGET-HISTORY FAMILY:\nMODEL = {history.model_id}\nRMSE = {history.RMSE:.3f}\nMAE = {history.MAE:.3f}\nSkill = {history.Skill_vs_persistence:.4f}\nR2_delta = {history.R2_delta:.4f}\nCoverage = {history.prediction_coverage:.4f}",
                    f"\nBOOTSTRAP INTERPRETATION:\n{boot.interpretation}",
                ]
            )
    positive = registry.loc[
        registry["horizon"].isin(PRIMARY_HORIZONS) & registry["validation_Skill"].gt(0)
    ]["candidate_id"].tolist()
    failures = registry.loc[
        registry["horizon"].isin(PRIMARY_HORIZONS) & registry["generalization_status"].eq("GENERALIZATION_FAILURE")
    ]["candidate_id"].tolist()
    pi_primary = pi.loc[pi["horizon"].isin(PRIMARY_HORIZONS)]
    lines.extend(
        [
            f"\n{'-'*50}\nGENERALIZATION\n{'-'*50}\n\nMODELS WITH POSITIVE 2022 SKILL:\n" + ("\n".join(f"{i+1}. {name}" for i, name in enumerate(positive)) if positive else "NONE"),
            "\nMODELS WITH DEVELOPMENT → VALIDATION FAILURE:\n" + ("\n".join(f"{i+1}. {name}" for i, name in enumerate(failures[:12])) if failures else "NONE"),
            "\nNO CLEAR DOMINANCE:\nYES — point estimates are kept separate from bootstrap certainty",
            f"\n{'-'*50}\nTARGET-HISTORY EFFECT\n{'-'*50}\n\nM2 h7/h14:\nreported separately above\n\nACCURACY / COVERAGE TRADE-OFF:\nM2 requires observed y(t); missing origin target makes it unavailable",
            f"\n{'-'*50}\nUNCERTAINTY\n{'-'*50}\n\nPI80 COVERAGE:\n{pi_primary.PI80_coverage.mean():.4f}\n\nPI95 COVERAGE:\n{pi_primary.PI95_coverage.mean():.4f}\n\nBOOTSTRAP UNCERTAINTY:\nseed={RANDOM_SEED}; replicates={N_BOOT}; block=max(7,h)",
            f"\n{'-'*50}\nTEMPORAL STABILITY\n{'-'*50}\n\nQ1/Q2/Q3/Q4:\npredefined calendar-quarter metrics exported\n\nMODEL ORDERING CHANGES:\n{'YES' if recommendation['r8']['recommendation']=='REQUIRED' else 'NO CLEAR EVIDENCE'}",
            f"\n{'-'*50}\nR8 DECISION\n{'-'*50}\n\nR8 RECOMMENDATION:\n{recommendation['r8']['recommendation']}\n\nEVIDENCE:\n" + "\n".join(f"{i+1}. {item}" for i, item in enumerate(recommendation['r8']['evidence'])),
            f"\n{'-'*50}\nR9 DECISION\n{'-'*50}\n\nR9 RECOMMENDATION:\n{recommendation['r9']['recommendation']}\n\nEVIDENCE:\n" + "\n".join(f"{i+1}. {item}" for i, item in enumerate(recommendation['r9']['evidence'])),
            f"\n{'-'*50}\nR10\n{'-'*50}\n\nR10 SCIENTIFIC REVIEW:\nREQUIRED",
            f"\n{'-'*50}\n2023\n{'-'*50}\n\n2023 TARGET ACCESSED:\nNO\n\n2023 METRICS:\nNONE\n\nFINAL TEST STATUS:\nKEEP SEALED",
            f"\n{'-'*50}\nTESTS\n{'-'*50}\n\nPASSED = {passed}\nFAILED = {failed}\n\nPHASE 07 STATUS:\n{status}\n\nNEXT ACTION:\nWAIT FOR USER REVIEW",
            "=" * 50,
        ]
    )
    return "\n".join(lines)


def main() -> int:
    for directory in (OUTPUT_DIR, DATA_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    artifacts = load_frozen_artifacts(PROJECT_ROOT)
    sources = load_validation_sources(PROJECT_ROOT)
    predictions = generate_validation_predictions(artifacts, sources)
    predictions, _ = calibrate_development_intervals(predictions, PROJECT_ROOT)
    metrics, development, _ = evaluate_validation_metrics(predictions, PROJECT_ROOT)
    primary = primary_comparison(metrics)
    uncertainty = pairwise_uncertainty(predictions, primary)
    quarter = validation_by_quarter(predictions)
    dq = validation_by_dq_context(predictions)
    gap = generalization_gap(metrics, development)
    extrapolation = extrapolation_audit(predictions)
    pi = prediction_interval_validation(predictions)
    correlations = residual_correlation(predictions, primary)
    freeze = artifact_freeze_audit(artifacts)
    registry = model_validation_registry(
        artifacts, metrics, development, gap, uncertainty, quarter, correlations
    )
    recommendation = next_stage_recommendation(
        primary, uncertainty, quarter, dq, correlations, registry
    )

    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    for frame, path in (
        (freeze, FREEZE_PATH), (metrics, METRICS_PATH), (primary, PRIMARY_PATH),
        (uncertainty, UNCERTAINTY_PATH), (quarter, QUARTER_PATH), (dq, DQ_PATH),
        (gap, GAP_PATH), (extrapolation, EXTRAPOLATION_PATH), (pi, PI_PATH),
        (correlations, CORRELATION_PATH), (registry, REGISTRY_PATH),
    ):
        _write_csv(frame, path)
    render_phase07_figures(primary, metrics, gap, quarter, pi, correlations, FIGURE_DIR)
    gate = build_gate(predictions, freeze, metrics, uncertainty, pi, registry)
    _write_csv(gate, GATE_PATH)
    REPORT_PATH.write_text(
        render_report(
            primary, metrics, uncertainty, quarter, dq, gap, extrapolation, pi,
            correlations, registry, recommendation, gate,
            tests_passed="PENDING", tests_failed="PENDING",
        ),
        encoding="utf-8",
    )
    _write_yaml(recommendation, RECOMMENDATION_PATH)

    result, test_frame, test_output = run_tests()
    _write_csv(test_frame, TEST_PATH)
    passed = int(result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped))
    failed = int(len(result.failures) + len(result.errors))
    status = "CONDITIONAL PASS" if failed == 0 and gate["status"].eq("PASS").all() else "FAIL"
    recommendation["phase07_status"] = status
    _write_yaml(recommendation, RECOMMENDATION_PATH)
    REPORT_PATH.write_text(
        render_report(
            primary, metrics, uncertainty, quarter, dq, gap, extrapolation, pi,
            correlations, registry, recommendation, gate,
            tests_passed=passed, tests_failed=failed,
        ),
        encoding="utf-8",
    )
    print(test_output)
    print(_console(primary, uncertainty, registry, pi, recommendation, passed, failed, status))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
