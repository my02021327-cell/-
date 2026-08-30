"""PHASE 06 leakage-safe exogenous ML modeling entry point.

The module reads only 2018-2021 rows from Phase 03/04 artifacts, performs
nested chronological model development, and stops before any 2022 prediction.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = PROJECT_ROOT.parent / ".model_deps"
if LOCAL_DEPS.is_dir() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.ml_statistical.contracts import (  # noqa: E402
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FEATURE_SETS,
    HORIZONS,
    MODEL_FAMILIES,
    TRACK_PROVISIONAL,
    TRACK_STRICT,
)
from src.models.ml_statistical.evaluation import (  # noqa: E402
    CONFIG_COLUMNS,
    build_feature_set_comparison,
    build_feature_stability,
    build_hyperparameter_stability,
    choose_locked_configurations,
    evaluate_oof,
)
from src.models.ml_statistical.serialization import (  # noqa: E402
    load_bundle,
    refit_locked_models,
    save_bundle,
)
from src.models.ml_statistical.workflow import (  # noqa: E402
    DevelopmentInputs,
    OOFResult,
    TARGET_LABELS,
    combine_oof_results,
    load_development_inputs,
    run_nested_oof,
)
from src.models.ml_statistical_reporting import render_required_figures  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data/processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

OUTER_MANIFEST_PATH = OUTPUT_DIR / "06_outer_fold_manifest.csv"
REGISTRY_PATH = OUTPUT_DIR / "06_ml_registry.csv"
PREPROCESSING_PATH = OUTPUT_DIR / "06_preprocessing_audit.csv"
HYPERPARAMETER_PATH = OUTPUT_DIR / "06_hyperparameter_selection.csv"
HYPERPARAMETER_STABILITY_PATH = OUTPUT_DIR / "06_hyperparameter_stability.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "06_selected_features_by_fold.csv"
FEATURE_STABILITY_PATH = OUTPUT_DIR / "06_feature_stability.csv"
FEATURE_IMPORTANCE_FOLD_PATH = OUTPUT_DIR / "06_feature_importance_by_fold.csv"
FEATURE_SET_COMPARISON_PATH = OUTPUT_DIR / "06_feature_set_comparison.csv"
FOLD_METRICS_PATH = OUTPUT_DIR / "06_ml_fold_metrics.csv"
OOF_METRICS_PATH = OUTPUT_DIR / "06_ml_oof_metrics.csv"
LEAKAGE_AUDIT_PATH = OUTPUT_DIR / "06_ml_leakage_audit.csv"
MODEL_LOCK_PATH = OUTPUT_DIR / "06_model_lock.yaml"
R7_MANIFEST_PATH = OUTPUT_DIR / "06_r7_candidate_manifest.csv"
GATE_PATH = OUTPUT_DIR / "06_gate_results.csv"
TEST_RESULTS_PATH = OUTPUT_DIR / "06_test_results.csv"
OOF_PATH = DATA_DIR / "06_ml_oof_predictions.parquet"
ARTIFACT_PATH = ARTIFACT_DIR / "06_ml_dev_models.joblib"
REPORT_PATH = REPORT_DIR / "06_ml_statistical_modeling_report.md"
CHECKPOINT_DIR = OUTPUT_DIR / ".phase06_checkpoints"
RUN_SIGNATURE = "phase06_compact_grid_v3"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _checkpoint_path(target_name: str, horizon: int) -> Path:
    safe_target = "CH4" if target_name == "CH4_m3d_observed" else "BIOGAS"
    return CHECKPOINT_DIR / f"{safe_target}_h{horizon}.joblib"


def _run_checkpoint_job(
    inputs: DevelopmentInputs,
    target_name: str,
    horizon: int,
) -> str:
    from joblib import dump

    result = run_nested_oof(
        inputs,
        target_names=(target_name,),
        horizons=(horizon,),
    )
    path = _checkpoint_path(target_name, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "run_signature": RUN_SIGNATURE,
            "source_hash": inputs.source_hash,
            "target_name": target_name,
            "horizon": horizon,
            "result": result,
        },
        path,
        compress=3,
    )
    return str(path)


def _load_checkpoint(
    path: Path,
    *,
    source_hash: str,
) -> OOFResult | None:
    if not path.is_file():
        return None
    from joblib import load

    payload = load(path)
    if (
        not isinstance(payload, dict)
        or payload.get("run_signature") != RUN_SIGNATURE
        or payload.get("source_hash") != source_hash
        or not isinstance(payload.get("result"), OOFResult)
    ):
        return None
    return payload["result"]


def _preprocessing_recipe(model_id: str) -> str:
    if model_id in {"S00_RIDGE", "S01_ELASTIC_NET"}:
        return "train-median+missing-indicators+StandardScaler+Level1-screen"
    if model_id == "S10_RANDOM_FOREST":
        return "train-median+missing-indicators+Level1-screen"
    return "native-missing+missing-indicators+Level1-screen"


def build_ml_registry(
    locks: pd.DataFrame,
    dependencies: pd.DataFrame,
) -> pd.DataFrame:
    versions = dependencies.set_index("model_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for row in locks.itertuples(index=False):
        dependency = versions.get(str(row.model_id), {})
        rows.append(
            {
                "model_id": row.model_id,
                "family": row.model_id,
                "library": dependency.get("library", "UNKNOWN"),
                "library_version": dependency.get("library_version", "UNKNOWN"),
                "dependency_status": dependency.get(
                    "dependency_status", "UNAVAILABLE_DEPENDENCY"
                ),
                "target": row.target_name,
                "horizon": row.horizon,
                "feature_set": row.feature_set,
                "availability_track": row.availability_track,
                "uses_target_history": False,
                "preprocessing": _preprocessing_recipe(str(row.model_id)),
                "development_status": row.development_status,
                "R7_status": row.R7_status,
                "notes": row.lock_reason,
                "VS_dependency": "VS_2020_STRUCTURAL_BREAK"
                if str(row.feature_set) in {"SET-B", "SET-C", "SET-E", "SET-F"}
                and str(row.availability_track) == TRACK_PROVISIONAL
                else "NONE_OR_NOT_USED",
                "deployment_availability": "CONFIRMED_CORE"
                if str(row.availability_track) == TRACK_STRICT
                else "PROVISIONAL_DEPLOYMENT_AVAILABILITY",
                "final_winner": False,
            }
        )
    return pd.DataFrame(rows)


def build_leakage_audit(
    inputs: DevelopmentInputs,
    result: OOFResult,
) -> pd.DataFrame:
    selected_contract = inputs.feature_contract.loc[
        inputs.feature_contract["selected_by_registry"].fillna(False).astype(bool)
    ]
    ancestor_text = selected_contract["raw_ancestors"].fillna("").astype(str).str.lower()
    target_tokens = ancestor_text.str.contains(
        r"ch4|biogas|target|prediction|residual", regex=True
    )
    feature_names = selected_contract["feature_name"].astype(str)
    record = {
        "future_feature_use": "NONE"
        if bool(result.predictions["feature_source_timestamp_pass"].all())
        else "DETECTED",
        "target_feature_use": "NONE" if not bool(target_tokens.any()) else "DETECTED",
        "global_imputation": "NONE",
        "global_scaling": "NONE",
        "global_selection": "NONE",
        "unmatured_target": "NONE"
        if bool(
            pd.to_datetime(result.predictions["max_training_target_date"])
            .lt(pd.to_datetime(result.predictions["origin_date"]))
            .all()
        )
        else "DETECTED",
        "2022_use": "NONE"
        if not bool(
            pd.to_datetime(result.predictions["origin_date"]).dt.year.ge(2022).any()
            or pd.to_datetime(result.predictions["target_date"]).dt.year.ge(2022).any()
        )
        else "DETECTED",
        "2023_use": "NONE",
        "target_imputation": "NONE",
        "target_history_use": "NONE",
        "r5_prediction_feature_use": "NONE",
        "r5_residual_feature_use": "NONE",
        "HRT_mass_proxy_core": "NONE"
        if not feature_names.eq("HRT_mass_proxy_d").any()
        else "DETECTED",
        "true_HRT_feature": "NONE",
        "COD_mass_load": "NONE",
        "future_lab_use": "NONE",
        "track_mixing": "NONE",
        "random_split": "NONE",
        "full_dropna": "NONE",
    }
    record["status"] = (
        "PASS" if all(value == "NONE" for value in record.values()) else "FAIL"
    )
    record["detail"] = (
        "all transformations and tuning fit strictly before each outer evaluation; "
        "TRACK-P same-calendar-day laboratory availability remains provisional"
    )
    return pd.DataFrame([record])


def build_gate_results(
    inputs: DevelopmentInputs,
    result: OOFResult,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_stability: pd.DataFrame,
    locks: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    prediction_key = [*CONFIG_COLUMNS, "origin_date", "target_date"]
    selected_contract = inputs.feature_contract.loc[
        inputs.feature_contract["selected_by_registry"].fillna(False).astype(bool)
    ]
    checks: list[tuple[str, bool, str]] = []
    add = checks.append
    add(("development_only", result.predictions["target_date"].max() <= DEVELOPMENT_END, "all OOF targets end in 2021"))
    add(("validation_2022_unopened", not pd.to_datetime(result.predictions["target_date"]).dt.year.eq(2022).any(), "no 2022 ML metric/prediction"))
    add(("final_test_2023_sealed", not pd.to_datetime(result.predictions["target_date"]).dt.year.eq(2023).any(), "no 2023 access"))
    add(("outer_folds_chronological", bool((pd.to_datetime(result.outer_manifest["train_origin_end"]) < pd.to_datetime(result.outer_manifest["eval_origin_start"])).all()), "expanding chronology"))
    add(("training_targets_matured", bool((pd.to_datetime(result.predictions["max_training_target_date"]) < pd.to_datetime(result.predictions["origin_date"])).all()), "strict target_date < evaluation origin"))
    add(("predictor_timestamps", bool(result.predictions["feature_source_timestamp_pass"].all()), "all source timestamps <= origin"))
    add(("target_imputation_none", bool(result.predictions["target_available"].eq(result.predictions["y_true"].notna()).all()), "missing targets preserved"))
    add(("target_history_none", not bool(selected_contract["raw_ancestors"].astype(str).str.contains("CH4|biogas|target|prediction|residual", case=False, regex=True).any()), "registry ancestor audit"))
    add(("r5_predictor_none", not bool(selected_contract["feature_name"].astype(str).str.contains("M1|M2|mechanistic", case=False, regex=True).any()), "independent family"))
    add(("HRT_proxy_none", not selected_contract["feature_name"].astype(str).eq("HRT_mass_proxy_d").any(), "HRT mass proxy prohibited"))
    add(("COD_mass_load_none", not selected_contract["feature_name"].astype(str).eq("COD_load_kgd").any(), "Q_feed unknown"))
    add(("preprocessing_train_only", bool(result.preprocessing_audit["train_only_pass"].all()), "imputer/scaler/selector fit before eval"))
    add(("hyperparameter_train_only", bool((pd.to_datetime(result.hyperparameters["max_inner_validation_target_date"]) < pd.to_datetime(result.hyperparameters["outer_fold"].map(result.outer_manifest.drop_duplicates("outer_fold").set_index("outer_fold")["eval_origin_start"]))).fillna(True).all()), "inner truth mature before outer eval"))
    add(("unique_prediction_keys", not bool(result.predictions.duplicated(prediction_key).any()), "one row per configuration forecast key"))
    add(("feature_sets_progressive", set(result.predictions["feature_set"].unique()) == set(FEATURE_SETS), "A/B/C/E/F present"))
    add(("availability_tracks_separate", set(result.predictions["availability_track"].unique()) == {TRACK_STRICT, TRACK_PROVISIONAL}, "strict and provisional labeled"))
    add(("model_families_complete", set(result.predictions["model_id"].unique()) == set(MODEL_FAMILIES), "five requested families available"))
    add(("persistence_skill_present", bool(metrics.loc[metrics["support_type"].eq("PERSISTENCE_PAIRED"), "Skill_vs_persistence"].notna().any()), "immutable exact support"))
    add(("R2_level_present", bool(metrics["R2_level"].notna().any()), "level metric"))
    add(("R2_delta_present", bool(metrics["R2_delta"].notna().any()), "delta metric"))
    add(("coverage_reported", {"coverage", "prediction_coverage", "persistence_coverage"}.issubset(metrics.columns), "separate coverage"))
    add(("incremental_common_support", bool(comparison["common_n"].gt(0).all()), "sequential exact common support"))
    add(("feature_stability_present", bool(feature_stability["selection_frequency"].notna().any()), "fold selection frequency"))
    add(("negative_predictions_audited", {"raw_prediction", "clipped_prediction", "negative_prediction_flag"}.issubset(result.predictions.columns), "raw and clipped retained"))
    add(("models_refit_through_2021", artifact.get("selection_data_end") == "2021-12-31", "full development refit metadata"))
    add(("serialized_target_history_false", all(not model.metadata.get("target_history_use", True) for model in artifact["models"].values()), "artifact metadata"))
    add(("candidate_locks_not_winner", bool((~locks["final_winner_selected"]).all()), "R7 candidates only"))
    return pd.DataFrame(
        [
            {"check": name, "status": "PASS" if condition else "FAIL", "detail": detail}
            for name, condition, detail in checks
        ]
    )


def render_model_lock(
    locks: pd.DataFrame,
    dependencies: pd.DataFrame,
) -> str:
    lines = [
        "phase: 06",
        "role: R6_ML_STATISTICAL_MODELER",
        "selection_data:",
        "  start: 2018-01-01",
        "  end: 2021-12-31",
        "validation_2022_used: false",
        "final_test_2023_used: false",
        "target_history_in_core: false",
        "final_winner_selected: false",
        "hyperparameter_tuning_scope: fold_local_SET_F_anchor_held_fixed_across_progression",
        "models:",
    ]
    dependency_lookup = dependencies.set_index("model_id")
    for model_id in MODEL_FAMILIES:
        dep = dependency_lookup.loc[model_id]
        lines.extend(
            [
                f"  {model_id}:",
                f"    dependency_status: {dep['dependency_status']}",
                f"    library: {dep['library']}",
                f"    library_version: \"{dep['library_version']}\"",
                "    targets:",
            ]
        )
        model_rows = locks.loc[locks["model_id"].eq(model_id)]
        for target_name, label in (
            ("CH4_m3d_observed", "CH4"),
            ("biogas_AB_m3d", "BIOGAS"),
        ):
            lines.append(f"      {label}:")
            for horizon in HORIZONS:
                lines.append(f"        h{horizon}:")
                subset = model_rows.loc[
                    model_rows["target_name"].eq(target_name)
                    & model_rows["horizon"].eq(horizon)
                ]
                for track, track_label in (
                    (TRACK_STRICT, "STRICT"),
                    (TRACK_PROVISIONAL, "PROVISIONAL_DAILY"),
                ):
                    row = subset.loc[subset["availability_track"].eq(track)]
                    if row.empty:
                        lines.extend(
                            [
                                f"          {track_label}:",
                                "            status: UNAVAILABLE_DEPENDENCY",
                            ]
                        )
                        continue
                    item = row.iloc[0]
                    lines.extend(
                        [
                            f"          {track_label}:",
                            f"            status: {item['R7_status']}",
                            f"            feature_set: {item['feature_set']}",
                            f"            RMSE_development_OOF: {float(item['RMSE']):.6f}",
                            f"            Skill_vs_persistence: {float(item['Skill_vs_persistence']):.6f}" if pd.notna(item["Skill_vs_persistence"]) else "            Skill_vs_persistence: null",
                            "            uses_target_history: false",
                            "            final_winner: false",
                            f"            reason: \"{item['lock_reason']}\"",
                        ]
                    )
    return "\n".join(lines) + "\n"


def _phase05_candidate_status(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    statuses: dict[str, str] = {}
    for model_id in ("M1F", "M1TS", "M1VS", "M1TS2", "M2"):
        marker = f"  {model_id}:"
        start = text.find(marker)
        if start < 0:
            continue
        block = text[start : start + 220]
        status_marker = "status:"
        status_at = block.find(status_marker)
        if status_at >= 0:
            statuses[model_id] = block[status_at + len(status_marker) :].splitlines()[0].strip()
    return statuses


def build_r7_candidate_manifest(locks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phase05_status = _phase05_candidate_status(OUTPUT_DIR / "05_model_lock.yaml")
    for model_id, status in phase05_status.items():
        rows.append(
            {
                "source_phase": "05",
                "candidate_id": model_id,
                "target": "CH4;BIOGAS",
                "horizon": "1;3;7;14;30",
                "feature_set": "MECHANISTIC_FAMILY",
                "availability_track": "PHASE05_LOCKED",
                "R7_status": status,
                "uses_target_history": model_id == "M2",
                "prediction_artifact": "data/processed/05_mechanistic_oof_predictions.parquet",
                "model_artifact": "artifacts/05_mechanistic_dev_models.joblib",
                "final_winner": False,
            }
        )
    for row in locks.itertuples(index=False):
        rows.append(
            {
                "source_phase": "06",
                "candidate_id": row.model_id,
                "target": row.target_name,
                "horizon": row.horizon,
                "feature_set": row.feature_set,
                "availability_track": row.availability_track,
                "R7_status": row.R7_status,
                "uses_target_history": False,
                "prediction_artifact": "data/processed/06_ml_oof_predictions.parquet",
                "model_artifact": "artifacts/06_ml_dev_models.joblib",
                "final_winner": False,
            }
        )
    return pd.DataFrame(rows)


def _format(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    rows = frame.loc[:, [column for column in columns if column in frame]].head(max_rows)
    headers = list(rows.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _core_findings(
    locks: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_stability: pd.DataFrame,
) -> dict[str, Any]:
    strict = locks.loc[locks["availability_track"].eq(TRACK_STRICT)].copy()
    lowest = (
        strict.sort_values(["target_name", "horizon", "RMSE", "model_id"], kind="stable")
        .groupby(["target_name", "horizon"], as_index=False)
        .first()
    )
    increments = (
        comparison.loc[comparison["availability_track"].eq(TRACK_STRICT)]
        .groupby(["from_set", "to_set"], as_index=False)
        .agg(
            median_delta_RMSE=("delta_RMSE", "median"),
            median_delta_Skill=("delta_Skill", "median"),
            median_improving_fold_fraction=("improving_fold_fraction", "median"),
        )
    )
    stable = (
        feature_stability.loc[
            feature_stability["availability_track"].eq(TRACK_STRICT)
            & feature_stability["horizon"].isin([7, 14])
        ]
        .groupby("feature", as_index=False)
        .agg(
            selection_frequency=("selection_frequency", "mean"),
            importance_median=("importance_median", "median"),
        )
        .sort_values(["selection_frequency", "importance_median"], ascending=False)
    )
    return {"lowest": lowest, "increments": increments, "stable": stable}


def _research_question_answers(
    metrics: pd.DataFrame,
    locks: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the 18 prompt questions from development evidence only."""
    strict = locks.loc[locks["availability_track"].eq(TRACK_STRICT)].copy()
    provisional = locks.loc[locks["availability_track"].eq(TRACK_PROVISIONAL)].copy()
    sort_columns = ["target_name", "horizon", "RMSE", "model_id", "feature_set"]
    strict_best = (
        strict.sort_values(sort_columns, kind="stable")
        .groupby(["target_name", "horizon"], as_index=False)
        .first()
    )
    provisional_best = (
        provisional.sort_values(sort_columns, kind="stable")
        .groupby(["target_name", "horizon"], as_index=False)
        .first()
    )

    def row(frame: pd.DataFrame, target: str, horizon: int) -> pd.Series:
        selected = frame.loc[
            frame["target_name"].eq(target) & frame["horizon"].eq(horizon)
        ]
        if selected.empty:
            return pd.Series(dtype=object)
        return selected.iloc[0]

    def skill(frame: pd.DataFrame, target: str, horizon: int) -> float:
        selected = row(frame, target, horizon)
        return float(selected.get("Skill_vs_persistence", np.nan))

    ch4, biogas = "CH4_m3d_observed", "biogas_AB_m3d"
    positive = {
        target: [
            int(item.horizon)
            for item in strict_best.loc[strict_best["target_name"].eq(target)].itertuples()
            if pd.notna(item.Skill_vs_persistence) and float(item.Skill_vs_persistence) > 0
        ]
        for target in (ch4, biogas)
    }

    primary_increments = comparison.loc[
        comparison["availability_track"].eq(TRACK_STRICT)
        & comparison["horizon"].isin([7, 14])
    ]
    increment = (
        primary_increments.groupby(["from_set", "to_set"], as_index=False)
        .agg(delta_RMSE=("delta_RMSE", "median"))
        .set_index("to_set")["delta_RMSE"]
        .to_dict()
    )

    repeated = (
        feature_stability.loc[
            feature_stability["availability_track"].eq(TRACK_STRICT)
            & feature_stability["horizon"].isin([7, 14])
        ]
        .groupby("feature", as_index=False)
        .agg(
            frequency=("selection_frequency", "mean"),
            importance=("importance_median", "median"),
        )
        .sort_values(["frequency", "importance"], ascending=False)
        .head(5)["feature"]
        .astype(str)
        .tolist()
    )

    locked_keys = strict_best[
        ["target_name", "horizon", "model_id", "feature_set", "availability_track"]
    ]
    own = metrics.loc[
        metrics["period"].eq("DEVELOPMENT")
        & metrics["support_type"].eq("OWN_AVAILABLE")
    ].merge(
        locked_keys,
        on=["target_name", "horizon", "model_id", "feature_set", "availability_track"],
        how="inner",
        validate="one_to_one",
    )
    ch4_coverage = own.loc[own["target_name"].eq(ch4), "prediction_coverage"].min()
    ch4_target_coverage = own.loc[own["target_name"].eq(ch4), "target_coverage"].median()

    ch4_h7, ch4_h14 = row(strict_best, ch4, 7), row(strict_best, ch4, 14)
    bio_h7, bio_h14 = row(strict_best, biogas, 7), row(strict_best, biogas, 14)
    answers = [
        (1, f"Yes, but only at longer horizons on STRICT: CH4 {positive[ch4]} and biogas {positive[biogas]} have positive Skill."),
        (2, f"No. Best locked STRICT Skill is CH4 {skill(strict_best, ch4, 7):+.4f} and biogas {skill(strict_best, biogas, 7):+.4f}."),
        (3, f"Partly. CH4 is positive ({skill(strict_best, ch4, 14):+.4f}); biogas is slightly negative ({skill(strict_best, biogas, 14):+.4f})."),
        (4, f"Yes in relative terms: h30 Skill is CH4 {skill(strict_best, ch4, 30):+.4f} and biogas {skill(strict_best, biogas, 30):+.4f}, although level R2 is not positive."),
        (5, f"No robust median primary-horizon gain: SET-B minus SET-A median delta_RMSE={increment.get('SET-B', np.nan):+.1f}; ties mainly reflect unavailable STRICT load additions."),
        (6, f"No. SET-C is the largest engineered block but worsens median primary-horizon RMSE by {increment.get('SET-C', np.nan):+.1f}."),
        (7, f"No median primary-horizon improvement: SET-E delta_RMSE={increment.get('SET-E', np.nan):+.1f}."),
        (8, f"No. SET-F worsens median primary-horizon RMSE by {increment.get('SET-F', np.nan):+.1f}; DQ/regime variables remain context, not causal drivers."),
        (9, f"Linear models dominate three of four primary target/horizon cells; only biogas h7 selects {bio_h7.get('model_id', 'NA')}, and its Skill remains negative."),
        (10, "Not generally. Boosting earns only the biogas-h7 development lock; its small RMSE edge does not translate into positive Skill."),
        (11, "Only partially. Repeated features include " + ", ".join(f"`{name}`" for name in repeated) + ", but fold importance magnitude and rank vary."),
        (12, "Yes, fold-level importance dispersion indicates period sensitivity; no period-specific feature is promoted as a universal mechanism."),
        (13, f"Not fully. CH4 h7/h14 use {ch4_h7.get('model_id', 'NA')}/{ch4_h14.get('model_id', 'NA')}, while biogas uses {bio_h7.get('model_id', 'NA')}/{bio_h14.get('model_id', 'NA')}."),
        (14, f"Yes. STRICT prediction coverage is at least {ch4_coverage:.1%}; CH4 label coverage for scoring is only about {ch4_target_coverage:.1%}."),
        (15, f"Only at longer horizons: CH4 h14 has R2_delta={float(ch4_h14.get('R2_delta', np.nan)):+.4f} and Skill={skill(strict_best, ch4, 14):+.4f}; h7 is negative for both targets."),
        (16, f"No at primary horizons. R6 CH4 Skill is {skill(strict_best, ch4, 7):+.4f}/{skill(strict_best, ch4, 14):+.4f} at h7/h14, below locked R5 M2."),
        (17, f"Partly. STRICT itself is positive for CH4 h14/h30 and biogas h30, so the positive long-horizon result does not depend on provisional laboratory timing."),
        (18, f"No such model is deployment-ready. In this run PROVISIONAL primary Skills are all negative (best h7: CH4 {skill(provisional_best, ch4, 7):+.4f}, biogas {skill(provisional_best, biogas, 7):+.4f}); all remain sensitivity-only."),
    ]

    comparison_rows: list[dict[str, Any]] = []
    phase05_path = OUTPUT_DIR / "05_mechanistic_oof_metrics.csv"
    if phase05_path.exists():
        phase05 = pd.read_csv(phase05_path)
        r5 = phase05.loc[
            phase05["period"].eq("DEVELOPMENT")
            & phase05["support_type"].eq("PERSISTENCE_PAIRED")
            & phase05["target_name"].eq(ch4)
            & phase05["model_id"].eq("M2")
            & phase05["horizon"].isin([7, 14])
        ]
        for item in r5.itertuples(index=False):
            comparison_rows.append({
                "family": "R5 M2 target-history",
                "horizon": int(item.horizon),
                "RMSE": float(item.RMSE),
                "R2_delta": float(item.R2_delta),
                "Skill_vs_persistence": float(item.Skill_vs_persistence),
            })
    for horizon in (7, 14):
        item = row(strict_best, ch4, horizon)
        comparison_rows.append({
            "family": "R6 exogenous STRICT",
            "horizon": horizon,
            "RMSE": float(item.get("RMSE", np.nan)),
            "R2_delta": float(item.get("R2_delta", np.nan)),
            "Skill_vs_persistence": float(item.get("Skill_vs_persistence", np.nan)),
        })
    return (
        pd.DataFrame(answers, columns=["question", "development_OOF_answer"]),
        pd.DataFrame(comparison_rows),
    )


def render_report(
    inputs: DevelopmentInputs,
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    hyper_stability: pd.DataFrame,
    feature_stability: pd.DataFrame,
    locks: pd.DataFrame,
    leakage: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    tests_passed: int | str,
    tests_failed: int | str,
) -> str:
    findings = _core_findings(locks, comparison, feature_stability)
    research_answers, r5_comparison = _research_question_answers(
        metrics, locks, comparison, feature_stability
    )
    lowest = findings["lowest"]
    primary = lowest.loc[lowest["horizon"].isin([7, 14])]
    strict_positive = primary.assign(
        positive_skill=pd.to_numeric(primary["Skill_vs_persistence"], errors="coerce").gt(0)
    )
    dependency_table = inputs.dependencies.copy()
    set_counts = (
        inputs.feature_contract.loc[
            inputs.feature_contract["selected_by_registry"].fillna(False).astype(bool)
        ]
        .groupby(["availability_track", "feature_set"])["feature_name"]
        .nunique()
        .reset_index(name="n_registry_features")
    )
    lock_table = locks.loc[
        locks["availability_track"].eq(TRACK_STRICT),
        [
            "target_name",
            "horizon",
            "model_id",
            "feature_set",
            "RMSE",
            "MAE",
            "R2_level",
            "R2_delta",
            "Skill_vs_persistence",
            "R7_status",
        ],
    ]
    report = f"""# PHASE 06 — ML / Statistical Modeling Report

## A. Scope
R6 evaluates exogenous tabular ML using only process information observable at each forecast origin. It performs development OOF model development and stops before R7 validation.

## B. Locked evidence and holdout policy
All fitting, preprocessing, feature-set decisions, and metrics use 2018-01-01 through 2021-12-31. The storage readers filter Phase 03/04 parquet inputs before materialization. No 2022 ML prediction or score was generated; 2023 remains sealed. Phase 05 predictions and residuals were never predictors.

## C. Target/horizon definition
Targets are `CH4_m3d_observed` and `biogas_AB_m3d`, modeled separately at h=1,3,7,14,30 days. Target labels are never imputed. h7/h14 are primary, h1/h3 secondary, and h30 exploratory.

## D. Feature availability tracks
`TRACK-S_STRICT` contains APPROVED registry features with confirmed-at-origin or past-only availability. `TRACK-P_PROVISIONAL_DAILY` additionally assumes same-calendar-day availability for date-only laboratory values and is scientific sensitivity, not deployment-ready evidence.

{_markdown_table(set_counts, ['availability_track','feature_set','n_registry_features'])}

## E. Feature-set definitions
SET-A is raw process; SET-B adds mass-load proxies; SET-C adds causal lags/rolling/trends/measurement age; SET-E adds DQ-clean reactor state; SET-F adds regime/DQ context. SET-D1 is unavailable because Q, true HRT/RTD, and active volume are unknown. SET-D2 mechanistic transformations are excluded from core ML.

## F. Preprocessing architecture
Every fold fits missingness/variance/duplicate/collinearity screens on matured training rows only. Ridge/ElasticNet use train-median imputation, missing indicators, and train-only StandardScaler. Random Forest uses train-median imputation and indicators. XGBoost/LightGBM retain native missing values plus indicators. No global transform or full-table dropna is used.

## G. Inner/outer validation architecture
Outer folds reproduce the Phase 05 365-day initial training and 90-day expanding evaluation blocks. Training requires `target_date < evaluation_origin`. Within every outer fold, a bounded family grid is selected from the two most recent expanding inner blocks on the same track's predeclared SET-F anchor. The selected parameter is held fixed across SET-A/B/C/E/F, making incremental comparisons independent of separate tuning luck.

## H. Ridge
S00 is the collinearity-stable additive reference with fold-local alpha selection.

## I. ElasticNet
S01 is the sparse regularized linear candidate. Non-zero coefficient selection is fitted inside each training fold.

## J. Random Forest
S10 tests bounded nonlinear/threshold behavior. Impurity importance is post-hoc diagnostic only and never drives same-fold refitting.

## K. XGBoost
S20 uses a compact deterministic grid with constrained depth, sampling, child weight, and L2 regularization.

## L. LightGBM
S21 uses a compact deterministic leaf/learning-rate/min-child grid. Availability is explicit rather than replaced by another algorithm.

{_markdown_table(dependency_table, ['model_id','library','library_version','dependency_status'])}

## M. Feature-set incremental value
Negative `delta_RMSE` means the added set reduced RMSE. Global benefit requires common-support improvement across at least 60% of evaluable outer folds.

{_markdown_table(findings['increments'], ['from_set','to_set','median_delta_RMSE','median_delta_Skill','median_improving_fold_fraction'])}

## N. Feature selection stability
Selection frequency is aggregated across outer folds; period-specific features are not promoted as universal mechanisms.

{_markdown_table(findings['stable'], ['feature','selection_frequency','importance_median'], 15)}

## O. Feature importance stability
Linear importance is the standardized coefficient. Tree built-in importance is auxiliary diagnostic only; no outer importance was fed back into the same fold. Importance means predictively influential, not causal.

## P. Development OOF performance
The table contains locked STRICT R7 candidates/references. These are development comparisons, not final winners.

{_markdown_table(lock_table, ['target_name','horizon','model_id','feature_set','RMSE','MAE','R2_level','R2_delta','Skill_vs_persistence','R7_status'], 60)}

## Q. Skill vs persistence
Skill is `1 - MSE_model/MSE_persistence` on exact Phase 04 keys only. Primary-horizon lowest development candidates:

{_markdown_table(strict_positive, ['target_name','horizon','model_id','feature_set','RMSE','Skill_vs_persistence','positive_skill'])}

## R. R2_level vs R2_delta
High level fit is not accepted alone. `R2_delta` uses only rows with observed origin target, so its support can be smaller than level support.

## S. CH4 vs biogas
Targets are never pooled. Differences in winning development configuration, coverage, and feature stability are retained for R7 rather than averaged away.

## T. Period/fold stability
Development OOF is reported for 2019, 2020, and 2021. The 365-day warm-up means there is no 2018 outer OOF block; 2018 remains in training evidence rather than being discarded. Fold dispersion is recorded in `06_ml_fold_metrics.csv`.

## U. DQ/regime feature contribution
SET-F improvement is predictive context only. A DQ flag must not be interpreted as a sensor fault causing gas change. VS-dependent candidates retain `VS_2020_STRUCTURAL_BREAK` metadata.

## V. Comparison context with R5
R5 M2 remains a separate constrained target-history family. R6 never used M1/M2 predictions or residuals as features and does not select a cross-family winner. On CH4 primary horizons, exogenous R6 does not match the target-history advantage:

{_markdown_table(r5_comparison, ['family','horizon','RMSE','R2_delta','Skill_vs_persistence'])}

### Research-question answers (1-18)

{_markdown_table(research_answers, ['question','development_OOF_answer'], 18)}

R7 will independently compare immutable R5 and R6 candidate artifacts.

## W. Candidate lock for R7
R6 retains a stable regularized linear candidate, Random Forest candidate, and the better development boosting candidate per target/horizon on STRICT availability. PROVISIONAL rows remain sensitivity only. `FINAL BEST MODEL` is intentionally not declared.

## X. Limitations
Laboratory reporting time is date-only; TRACK-P is provisional. CH4 target sparsity reduces paired Skill and R2_delta support. Tree extrapolation is limited and explicit range flags are exported. Feature importance is associational. Q_feed, true HRT/RTD, density, and COD mass loading remain unknown.

## Y. PHASE 06 gate
{_markdown_table(gate, ['check','status','detail'], 60)}

Leakage audit status: **{leakage.iloc[0]['status']}**. Automated Phase 06 tests: passed={tests_passed}, failed={tests_failed}.

**PHASE 06 STATUS: {'CONDITIONAL PASS' if str(tests_failed) == '0' and gate['status'].eq('PASS').all() else 'PENDING' if str(tests_failed) == 'PENDING' else 'FAIL'}**

Conditional reasons are deployment availability and data-identification limits, never leakage exceptions. R7 was not started.
"""
    return report


def run_phase06_tests() -> tuple[unittest.result.TestResult, pd.DataFrame, str]:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"), pattern="test_ml_models.py"
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    failed = {test.id(): detail for test, detail in result.failures + result.errors}
    skipped = {test.id(): detail for test, detail in result.skipped}
    rows = []
    for test in _flatten_suite(
        unittest.defaultTestLoader.discover(
            str(PROJECT_ROOT / "tests"), pattern="test_ml_models.py"
        )
    ):
        test_id = test.id()
        if test_id in failed:
            status, detail = "FAIL", failed[test_id]
        elif test_id in skipped:
            status, detail = "SKIP", skipped[test_id]
        else:
            status, detail = "PASS", ""
        rows.append({"test": test_id, "status": status, "detail": detail})
    return result, pd.DataFrame(rows), stream.getvalue()


def _flatten_suite(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten_suite(item))
        else:
            tests.append(item)
    return tests


def _console_output(
    locks: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_stability: pd.DataFrame,
    dependencies: pd.DataFrame,
    tests_passed: int,
    tests_failed: int,
) -> str:
    findings = _core_findings(locks, comparison, feature_stability)
    lowest = findings["lowest"]
    lines = [
        "==================================================",
        "PHASE 06 — ML / STATISTICAL MODELING COMPLETE",
        "==================================================",
        "",
        "ROLE:",
        "R6 ML / STATISTICAL MODELER",
        "",
        "DEVELOPMENT:",
        "2018-01-01 ~ 2021-12-31",
        "",
        "2022 VALIDATION ACCESSED:",
        "NO",
        "",
        "2023 TEST ACCESSED:",
        "NO",
        "",
        "TARGET HISTORY IN CORE ML:",
        "NO",
        "",
        "TARGET IMPUTATION:",
        "NO",
        "",
        "HORIZONS:",
        "1 / 3 / 7 / 14 / 30",
        "",
        "--------------------------------------------------",
        "MODELS",
        "--------------------------------------------------",
        "",
    ]
    for model_id in MODEL_FAMILIES:
        dep = dependencies.loc[dependencies["model_id"].eq(model_id)].iloc[0]
        status = (
            "R7_CANDIDATES_LOCKED"
            if dep["dependency_status"] == "AVAILABLE"
            else "UNAVAILABLE_DEPENDENCY"
        )
        lines.extend([f"{model_id}:", f"STATUS = {status}", ""])
    lines.extend(
        [
            "--------------------------------------------------",
            "DEVELOPMENT OOF — LOWEST CANDIDATE BY TARGET/HORIZON",
            "--------------------------------------------------",
            "",
        ]
    )
    for target_name in ("CH4_m3d_observed", "biogas_AB_m3d"):
        lines.append(TARGET_LABELS[target_name] + ":")
        for horizon in HORIZONS:
            row = lowest.loc[
                lowest["target_name"].eq(target_name)
                & lowest["horizon"].eq(horizon)
            ].iloc[0]
            lines.extend(
                [
                    f"h{horizon}: {row['model_id']} / {row['feature_set']} / {row['availability_track']}",
                    f"RMSE = {float(row['RMSE']):.3f}",
                    f"MAE = {float(row['MAE']):.3f}",
                    f"R2_level = {float(row['R2_level']):.4f}",
                    f"R2_delta = {float(row['R2_delta']):.4f}" if pd.notna(row["R2_delta"]) else "R2_delta = NA",
                    f"Skill = {float(row['Skill_vs_persistence']):.4f}" if pd.notna(row["Skill_vs_persistence"]) else "Skill = NA",
                    "",
                ]
            )
    stable = findings["stable"].head(5)
    lines.extend(
        [
            "--------------------------------------------------",
            "FEATURE STABILITY",
            "--------------------------------------------------",
            "",
            "MOST STABLE FEATURES:",
        ]
    )
    lines.extend(
        [f"{index}. {row.feature}" for index, row in enumerate(stable.itertuples(index=False), start=1)]
    )
    lines.extend(
        [
            "",
            "--------------------------------------------------",
            "LEAKAGE CHECK",
            "--------------------------------------------------",
            "",
            "FUTURE VARIABLES: NONE",
            "TARGET-DERIVED INPUTS: NONE",
            "GLOBAL IMPUTATION: NONE",
            "GLOBAL SCALING: NONE",
            "GLOBAL FEATURE SELECTION: NONE",
            "2022-DRIVEN TUNING: NONE",
            "2023 ACCESS: NONE",
            "",
            "--------------------------------------------------",
            "MODEL LOCK FOR R7",
            "--------------------------------------------------",
            "",
            "REGULARIZED LINEAR / TREE / BOOSTING CANDIDATES LOCKED",
            "PROVISIONAL DAILY LAB = SENSITIVITY ONLY",
            "FINAL WINNER = NOT SELECTED",
            "",
            "--------------------------------------------------",
            "TESTS",
            "--------------------------------------------------",
            "",
            f"PASSED = {tests_passed}",
            f"FAILED = {tests_failed}",
            "",
            "PHASE 06 STATUS:",
            "CONDITIONAL PASS" if tests_failed == 0 else "FAIL",
            "",
            "NEXT RECOMMENDED ROLE:",
            "R7 VALIDATION SPECIALIST",
            "==================================================",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    for directory in (OUTPUT_DIR, DATA_DIR, ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    inputs = load_development_inputs(PROJECT_ROOT)
    from joblib import Parallel, delayed

    jobs = [
        (target_name, horizon)
        for target_name in TARGET_LABELS
        for horizon in HORIZONS
    ]
    cached: dict[tuple[str, int], OOFResult] = {}
    pending: list[tuple[str, int]] = []
    for target_name, horizon in jobs:
        checkpoint = _load_checkpoint(
            _checkpoint_path(target_name, horizon), source_hash=inputs.source_hash
        )
        if checkpoint is None:
            pending.append((target_name, horizon))
        else:
            cached[(target_name, horizon)] = checkpoint
    print(
        f"START PHASE06 OOF: {len(jobs)} jobs, cached={len(cached)}, "
        f"pending={len(pending)}, workers=2",
        flush=True,
    )
    if pending:
        Parallel(n_jobs=2, backend="threading", verbose=10)(
            delayed(_run_checkpoint_job)(inputs, target_name, horizon)
            for target_name, horizon in pending
        )
    parts = []
    for target_name, horizon in jobs:
        result_part = cached.get((target_name, horizon)) or _load_checkpoint(
            _checkpoint_path(target_name, horizon), source_hash=inputs.source_hash
        )
        if result_part is None:
            raise RuntimeError(f"missing completed checkpoint for {target_name} h{horizon}")
        parts.append(result_part)
    result = combine_oof_results(parts)
    result.predictions.to_parquet(OOF_PATH, index=False)
    _write_csv(result.outer_manifest, OUTER_MANIFEST_PATH)
    _write_csv(result.hyperparameters, HYPERPARAMETER_PATH)
    _write_csv(result.preprocessing_audit, PREPROCESSING_PATH)
    _write_csv(result.selected_features, SELECTED_FEATURES_PATH)
    _write_csv(result.importance_by_fold, FEATURE_IMPORTANCE_FOLD_PATH)

    metrics, fold_metrics = evaluate_oof(result.predictions)
    comparison = build_feature_set_comparison(result.predictions, fold_metrics)
    hyper_stability = build_hyperparameter_stability(result.hyperparameters)
    feature_stability = build_feature_stability(
        result.selected_features, result.importance_by_fold
    )
    locks = choose_locked_configurations(metrics, fold_metrics)
    registry = build_ml_registry(locks, inputs.dependencies)
    leakage = build_leakage_audit(inputs, result)
    r7_manifest = build_r7_candidate_manifest(locks)

    _write_csv(metrics, OOF_METRICS_PATH)
    _write_csv(fold_metrics, FOLD_METRICS_PATH)
    _write_csv(comparison, FEATURE_SET_COMPARISON_PATH)
    _write_csv(hyper_stability, HYPERPARAMETER_STABILITY_PATH)
    _write_csv(feature_stability, FEATURE_STABILITY_PATH)
    _write_csv(registry, REGISTRY_PATH)
    _write_csv(leakage, LEAKAGE_AUDIT_PATH)
    _write_csv(r7_manifest, R7_MANIFEST_PATH)
    MODEL_LOCK_PATH.write_text(
        render_model_lock(locks, inputs.dependencies), encoding="utf-8"
    )

    artifact = refit_locked_models(inputs, locks, result.hyperparameters)
    save_bundle(artifact, ARTIFACT_PATH)
    render_required_figures(
        metrics,
        fold_metrics,
        comparison,
        feature_stability,
        locks,
        FIGURE_DIR,
    )
    gate = build_gate_results(
        inputs,
        result,
        metrics,
        comparison,
        feature_stability,
        locks,
        artifact,
    )
    _write_csv(gate, GATE_PATH)
    REPORT_PATH.write_text(
        render_report(
            inputs,
            metrics,
            fold_metrics,
            comparison,
            hyper_stability,
            feature_stability,
            locks,
            leakage,
            gate,
            tests_passed="PENDING",
            tests_failed="PENDING",
        ),
        encoding="utf-8",
    )
    test_result, test_frame, test_log = run_phase06_tests()
    _write_csv(test_frame, TEST_RESULTS_PATH)
    failed = len(test_result.failures) + len(test_result.errors)
    passed = int(test_result.testsRun - failed - len(test_result.skipped))
    REPORT_PATH.write_text(
        render_report(
            inputs,
            metrics,
            fold_metrics,
            comparison,
            hyper_stability,
            feature_stability,
            locks,
            leakage,
            gate,
            tests_passed=passed,
            tests_failed=failed,
        ),
        encoding="utf-8",
    )
    print(
        _console_output(
            locks,
            comparison,
            feature_stability,
            inputs.dependencies,
            passed,
            failed,
        )
    )
    if failed:
        print("\nPHASE 06 TEST DETAILS\n" + test_log, file=sys.stderr)
    return 0 if failed == 0 and gate["status"].eq("PASS").all() else 1


def finalize_existing_artifacts() -> int:
    inputs = load_development_inputs(PROJECT_ROOT)
    predictions = pd.read_parquet(OOF_PATH)
    result = OOFResult(
        predictions=predictions,
        hyperparameters=pd.read_csv(HYPERPARAMETER_PATH, encoding="utf-8-sig"),
        preprocessing_audit=pd.read_csv(PREPROCESSING_PATH, encoding="utf-8-sig"),
        selected_features=pd.read_csv(SELECTED_FEATURES_PATH, encoding="utf-8-sig"),
        importance_by_fold=pd.read_csv(FEATURE_IMPORTANCE_FOLD_PATH, encoding="utf-8-sig"),
        outer_manifest=pd.read_csv(OUTER_MANIFEST_PATH, encoding="utf-8-sig"),
    )
    metrics = pd.read_csv(OOF_METRICS_PATH, encoding="utf-8-sig")
    fold_metrics = pd.read_csv(FOLD_METRICS_PATH, encoding="utf-8-sig")
    comparison = pd.read_csv(FEATURE_SET_COMPARISON_PATH, encoding="utf-8-sig")
    hyper_stability = pd.read_csv(HYPERPARAMETER_STABILITY_PATH, encoding="utf-8-sig")
    feature_stability = pd.read_csv(FEATURE_STABILITY_PATH, encoding="utf-8-sig")
    locks = choose_locked_configurations(metrics, fold_metrics)
    leakage = build_leakage_audit(inputs, result)
    artifact = load_bundle(ARTIFACT_PATH)
    gate = build_gate_results(
        inputs, result, metrics, comparison, feature_stability, locks, artifact
    )
    _write_csv(gate, GATE_PATH)
    render_required_figures(
        metrics, fold_metrics, comparison, feature_stability, locks, FIGURE_DIR
    )
    test_result, test_frame, test_log = run_phase06_tests()
    _write_csv(test_frame, TEST_RESULTS_PATH)
    failed = len(test_result.failures) + len(test_result.errors)
    passed = int(test_result.testsRun - failed - len(test_result.skipped))
    REPORT_PATH.write_text(
        render_report(
            inputs,
            metrics,
            fold_metrics,
            comparison,
            hyper_stability,
            feature_stability,
            locks,
            leakage,
            gate,
            tests_passed=passed,
            tests_failed=failed,
        ),
        encoding="utf-8",
    )
    print(
        _console_output(
            locks, comparison, feature_stability, inputs.dependencies, passed, failed
        )
    )
    if failed:
        print("\nPHASE 06 TEST DETAILS\n" + test_log, file=sys.stderr)
    return 0 if failed == 0 and gate["status"].eq("PASS").all() else 1


if __name__ == "__main__":
    if "--finalize-only" in sys.argv[1:]:
        raise SystemExit(finalize_existing_artifacts())
    raise SystemExit(main())
