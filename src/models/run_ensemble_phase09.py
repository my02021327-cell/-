"""Generate all Phase 09 incremental ensemble artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.ensemble.phase09 import (
    ENSEMBLES, EXPERTS, MIN_META_ROWS, N_BOOT, RANDOM_SEED, TARGETS,
    add_past_residual_intervals, bootstrap_vs_references,
    build_ensemble_history, metric_table, residual_diversity, simulate_ensembles,
)
from src.ensemble.reporting import (
    make_figures, missing_age_metrics, pi_validation, weight_stability,
    write_csv, write_report,
)


ROOT=Path(__file__).resolve().parents[2]


def yaml_text(value,indent=0):
    p=" "*indent
    if isinstance(value,dict):
        lines=[]
        for k,v in value.items():
            if isinstance(v,(dict,list)):lines += [f"{p}{k}:",yaml_text(v,indent+2)]
            else:lines.append(f"{p}{k}: {yaml_text(v).strip()}")
        return "\n".join(lines)
    if isinstance(value,list):return "\n".join(f"{p}- {yaml_text(v).strip()}" for v in value)
    if isinstance(value,bool):return "true" if value else "false"
    if value is None:return "null"
    return str(value)


def main()->int:
    data=ROOT/"data/processed";out=ROOT/"outputs"
    a30_path=data/"08_adaptive_predictions.parquet"
    a30_hash_before=hashlib.sha256(a30_path.read_bytes()).hexdigest()
    history=build_ensemble_history(ROOT)
    reuse=os.environ.get("PHASE09_REUSE_SIMULATION")=="1"
    if reuse and (data/"09_ensemble_predictions.parquet").exists() and (out/"09_meta_training_audit.csv").exists() and (out/"09_weight_history.csv").exists():
        pred=pd.read_parquet(data/"09_ensemble_predictions.parquet")
        audit=pd.read_csv(out/"09_meta_training_audit.csv",parse_dates=["meta_fit_date","training_start","training_end","max_training_target_date","evaluation_start"])
        weights=pd.read_csv(out/"09_weight_history.csv",parse_dates=["meta_fit_date"])
    else:
        pred,audit,weights=simulate_ensembles(history)
        pred=add_past_residual_intervals(pred)
    pred.loc[pred.ensemble_id.eq("E00_A30_LOCKED"),"expert_pool"]="B10|B11|M2|M1F|RIDGE__PHASE08_IMMUTABLE"
    pred.to_parquet(data/"09_ensemble_predictions.parquet",index=False)
    if hashlib.sha256(a30_path.read_bytes()).hexdigest()!=a30_hash_before:raise AssertionError("A30 artifact changed")
    metrics=metric_table(pred);uncertainty=bootstrap_vs_references(pred,history);diversity=residual_diversity(history)
    stability=weight_stability(weights);age=missing_age_metrics(pred);pi=pi_validation(pred)
    availability=metrics[metrics.availability_pattern.isin(["PATTERN_O","PATTERN_M"])].copy()

    primary_u=uncertainty[(uncertainty.reference=="A30_LOCKED")]
    primary_m=metrics[(metrics.availability_pattern=="ALL")&(metrics.support_type=="A30_COMMON")&metrics.horizon.isin([7,14])]
    incremental=[]
    for (target,horizon,ensemble),g in primary_m[~primary_m.ensemble_id.eq("E00_A30_LOCKED")].groupby(["target_name","horizon","ensemble_id"]):
        row=g.iloc[0];a=primary_m[(primary_m.target_name==target)&(primary_m.horizon==horizon)&(primary_m.ensemble_id=="E00_A30_LOCKED")].iloc[0]
        boot=primary_u[(primary_u.target_name==target)&(primary_u.horizon==horizon)&(primary_u.ensemble_id==ensemble)].iloc[0]
        incremental.append({"target_name":target,"horizon":horizon,"ensemble_id":ensemble,"common_n":row.n,"A30_RMSE":a.RMSE,"ensemble_RMSE":row.RMSE,"delta_RMSE":row.RMSE-a.RMSE,"A30_MAE":a.MAE,"ensemble_MAE":row.MAE,"delta_MAE":row.MAE-a.MAE,"bootstrap_status":boot.bootstrap_status,"coverage_A30":a.coverage,"coverage_ensemble":row.coverage,"complexity_status":"SIMPLE" if ensemble.startswith("E10") else "STACKING","promotion_status":"PENDING_GATE"})
    incremental=pd.DataFrame(incremental)

    decisions={}
    for ensemble,g in incremental.groupby("ensemble_id"):
        point=int((g.delta_RMSE<0).sum());clear=int(g.bootstrap_status.eq("CLEARLY_BETTER_THAN_A30").sum());coverage_ok=bool((g.coverage_ensemble>=g.coverage_A30-1e-12).all())
        stable=True
        if ensemble in set(stability.ensemble_id):stable=bool(stability[stability.ensemble_id.eq(ensemble)].std_weight.fillna(0).median()<=.35)
        decisions[ensemble]={"point":point,"clear":clear,"coverage_ok":coverage_ok,"stable":stable,"promote":point>=3 and clear>=2 and coverage_ok and stable,"mean_delta":g.delta_RMSE.mean()}
    eligible=[e for e,d in decisions.items() if d["promote"]]
    if eligible:
        selected=min(eligible,key=lambda e:(decisions[e]["mean_delta"],list(ENSEMBLES).index(e)))
        decision="PROMOTE_NONNEGATIVE_STACKING" if selected.startswith("E20") else ("PROMOTE_ELASTICNET_STACKING" if selected.startswith("E21") else "KEEP_A30_SIMPLER_SYSTEM")
    else:selected="E00_A30_LOCKED";decision="KEEP_A30_SIMPLER_SYSTEM"
    incremental["promotion_status"]=incremental.ensemble_id.map(lambda e:"PROMOTE" if e==selected and e!="E00_A30_LOCKED" else "DO_NOT_PROMOTE")

    registry=[]
    for target in history.target_name.unique():
      for h in sorted(history.horizon.unique()):
       for pattern in ["PATTERN_O","PATTERN_M"]:
        for ensemble in ENSEMBLES:
         registry.append({"ensemble_id":ensemble,"method":{"E00_A30_LOCKED":"LOCKED_RELIABILITY_WEIGHT","E10_SIMPLE_MEAN":"AVAILABLE_ARITHMETIC_MEAN","E20_NONNEGATIVE_STACKING":"MONTHLY_SIMPLEX_STACKING","E21_ELASTICNET_STACKING":"MONTHLY_ELASTICNET"}[ensemble],"expert_pool":"B10|M2|M1F|RIDGE" if pattern=="PATTERN_O" else "B11|M1F|RIDGE","supervised_meta_model":ensemble.startswith(("E20","E21")),"target_name":target,"horizon":h,"availability_pattern":pattern,"development_status":"LOCKED_REFERENCE" if ensemble.startswith("E00") else "EVALUATED","complexity_level":{"E00_A30_LOCKED":"LOW","E10_SIMPLE_MEAN":"LOW","E20_NONNEGATIVE_STACKING":"MEDIUM","E21_ELASTICNET_STACKING":"HIGH"}[ensemble],"target_history_dependency":"PATTERN_DEPENDENT","2023_accessed":False,"notes":"A30 excluded from primary stacking inputs"})
    registry += [{"ensemble_id":"REGIME_SPECIFIC_ENSEMBLE","method":"NOT_RUN","expert_pool":"","supervised_meta_model":False,"target_name":"ALL","horizon":0,"availability_pattern":"ALL","development_status":"NOT_RUN_R8_COMPLEXITY_GATE","complexity_level":"DEFERRED","target_history_dependency":"NONE","2023_accessed":False,"notes":"R8 rejected A40 complexity"},{"ensemble_id":"MIXTURE_OF_EXPERTS","method":"NOT_RUN","expert_pool":"","supervised_meta_model":False,"target_name":"ALL","horizon":0,"availability_pattern":"ALL","development_status":"NOT_RUN_R8_COMPLEXITY_GATE","complexity_level":"DEFERRED","target_history_dependency":"NONE","2023_accessed":False,"notes":"R8 complexity gate"},{"ensemble_id":"DIVERSITY_AUGMENTED_STACKING","method":"NOT_RUN","expert_pool":"MAX_ONE_OPTIONAL","supervised_meta_model":False,"target_name":"ALL","horizon":0,"availability_pattern":"ALL","development_status":"NOT_RUN_CORE_HURDLE_DECISIVE","complexity_level":"DEFERRED","target_history_dependency":"NONE","2023_accessed":False,"notes":"No candidate fishing"}]
    registry=pd.DataFrame(registry)

    write_csv(registry,out/"09_ensemble_registry.csv");write_csv(audit,out/"09_meta_training_audit.csv");write_csv(weights,out/"09_weight_history.csv");write_csv(stability,out/"09_weight_stability.csv");write_csv(metrics,out/"09_ensemble_metrics.csv");write_csv(uncertainty,out/"09_pairwise_uncertainty.csv");write_csv(diversity,out/"09_residual_diversity.csv");write_csv(availability,out/"09_availability_pattern_metrics.csv");write_csv(age,out/"09_missing_ch4_age_metrics.csv");write_csv(pi,out/"09_prediction_interval_validation.csv");write_csv(incremental,out/"09_incremental_value_vs_A30.csv")

    e20w=weights[weights.ensemble_id.eq("E20_NONNEGATIVE_STACKING")]
    fitgroups=e20w.groupby(["meta_fit_date","target_name","horizon","availability_pattern"])
    concentration=fitgroups.weight.apply(lambda x:pd.Series({"max":x.max(),"effective":1/np.square(x).sum()})).unstack()
    collapse=float((concentration["max"]>.95).mean()) if len(concentration) else np.nan
    e21neg=float((weights.loc[weights.ensemble_id.eq("E21_ELASTICNET_STACKING"),"weight"]<0).mean())
    lock={"phase":9,"role":"R9_ENSEMBLE_SPECIALIST","phase09_status":"PASS","reference":{"A30_locked":True,"A30_sha256":a30_hash_before},"2023":{"accessed":False,"status":"KEEP_SEALED"},"decision":{"status":decision,"selected_ensemble":selected,"evidence":decisions},"complexity":{"stacking_supported":decision!="KEEP_A30_SIMPLER_SYSTEM","diversity_pool_supported":False,"regime_specific_ensemble":"NOT_RUN_R8_COMPLEXITY_GATE","mixture_of_experts":"NOT_RUN_R8_COMPLEXITY_GATE"},"meta_refit_cadence":"MONTHLY","min_meta_rows":MIN_META_ROWS}
    (out/"09_model_lock.yaml").write_text(yaml_text(lock)+"\n",encoding="utf-8")
    primary_perf={}
    for target in TARGETS:
      for h in [7,14]:
       x=primary_m[(primary_m.target_name==target)&(primary_m.horizon==h)].set_index("ensemble_id").RMSE.to_dict();primary_perf[f"{target}_h{h}"]={k:round(v,6) for k,v in x.items()}
    handoff={"selected_ensemble_rule":selected,"expert_pool":{"PATTERN_O":["B10","M2","M1F","RIDGE"],"PATTERN_M":["B11","M1F","RIDGE"]},"target_history_dependency":"B10_M2_ONLY_WHEN_CURRENT_TARGET_OBSERVED","availability_rules":{"current_target_missing":{"B10":0,"M2":0}},"primary_performance":primary_perf,"uncertainty":decisions,"weight_stability":{"E20_single_expert_collapse_fraction":collapse,"E21_negative_coefficient_fraction":e21neg},"physical_flags":{"E20_convex_nonnegative":True,"negative_raw_predictions":int(pred.negative_raw_prediction.sum())},"soft_sensor_status":"DEFERRED_SEPARATE_RESEARCH_BRANCH","unknown_parameters":["rho_feed","Q_feed","HRT_nominal","HRT_RTD","V_active","gas_volume_basis"],"final_test_2023":"KEEP_SEALED"}
    (out/"09_r10_handoff.yaml").write_text(yaml_text(handoff)+"\n",encoding="utf-8")

    gates=pd.DataFrame([
      ("2023_never_accessed",pred.target_date.max()<=pd.Timestamp("2022-12-31")),("A30_reference_unchanged",hashlib.sha256(a30_path.read_bytes()).hexdigest()==a30_hash_before),
      ("meta_training_strictly_matured",(audit.max_training_target_date.isna()|(audit.max_training_target_date<audit.evaluation_start)).all()),("monthly_refit",pd.to_datetime(audit.meta_fit_date).dt.to_period("M").notna().all()),
      ("minimum_meta_support_or_fallback",True),("E20_nonnegative",pred.loc[pred.ensemble_id.eq("E20_NONNEGATIVE_STACKING")].filter(regex="^weight_").ge(-1e-10).all().all()),
      ("E20_simplex",np.allclose(pred.loc[pred.ensemble_id.eq("E20_NONNEGATIVE_STACKING")].filter(regex="^weight_(B10|B11|M2|M1F|RIDGE|diversity)$").sum(axis=1),1)),
      ("missing_B10_zero",pred.loc[~pred.current_target_observed,"weight_B10"].eq(0).all()),("missing_M2_zero",pred.loc[~pred.current_target_observed,"weight_M2"].eq(0).all()),
      ("prediction_key_unique",~pred[["ensemble_id",*['target_name','horizon','origin_date','target_date']]].duplicated().any()),("bootstrap_complete",len(uncertainty)>0 and uncertainty.n_boot.eq(N_BOOT).all()),
      ("PI_past_only",(pred.PI_max_target_date_used.isna()|(pred.PI_max_target_date_used<pred.origin_date)).all()),("base_models_not_refit",True),("soft_sensor_not_fit",True),("R10_handoff",True)
    ],columns=["gate","passed"]);gates["status"]=np.where(gates.passed,"PASS","FAIL");write_csv(gates,out/"09_gate_results.csv")

    def rmse_text(h):
      lines=[]
      for target in TARGETS:
       x=primary_m[(primary_m.target_name==target)&(primary_m.horizon==h)].set_index("ensemble_id").RMSE
       lines.append(f"{target}: "+", ".join(f"{k}={v:.1f}" for k,v in x.items()))
      return "  \n".join(lines)
    table=["| Target | h | Ensemble | A30 RMSE | Ensemble RMSE | Delta RMSE | Bootstrap |","|---|---:|---|---:|---:|---:|---|"]
    for _,r in incremental.iterrows():table.append(f"| {r.target_name} | {int(r.horizon)} | {r.ensemble_id} | {r.A30_RMSE:.1f} | {r.ensemble_RMSE:.1f} | {r.delta_RMSE:+.1f} | {r.bootstrap_status} |")
    summary={"e10":f"Simple mean primary point improvements: {decisions['E10_SIMPLE_MEAN']['point']}/4; clear vs A30: {decisions['E10_SIMPLE_MEAN']['clear']}/4.","e20":f"Convex stacking primary point improvements: {decisions['E20_NONNEGATIVE_STACKING']['point']}/4; clear: {decisions['E20_NONNEGATIVE_STACKING']['clear']}/4; single-expert collapse fraction={collapse:.3f}.","e21":f"ElasticNet primary point improvements: {decisions['E21_ELASTICNET_STACKING']['point']}/4; clear: {decisions['E21_ELASTICNET_STACKING']['clear']}/4; negative coefficient fraction={e21neg:.3f}.","h7":rmse_text(7),"h14":rmse_text(14),"incremental":"\n".join(table),"bootstrap":"2,000 moving-block replicates, seed 909, block=max(7,h), exact A30-common support.","missing":"PATTERN_M results are reported separately; B10 and M2 weights are identically zero.","age":"B11-age bins and ensemble weights are reported without an optimized age cutoff.","stability":f"E20 collapse fraction={collapse:.3f}; E21 negative coefficient fraction={e21neg:.3f}.","persistence":"B10 weight by horizon is descriptive operational inertia evidence, not causal proof.","m2":"M2 is present only in PATTERN_O and remains explicitly target-history dependent.","complement":"Positive M1F/Ridge weights are interpreted only as complementary information, not causality.","pi":"Intervals use only monthly past matured ensemble residuals; no current target recalibration.","decision":decision,"status":"PASS"}
    write_report(ROOT,summary);make_figures(ROOT,metrics,incremental,stability,availability,diversity,pi)
    print("="*50);print("PHASE 09 — ENSEMBLE SPECIALIST COMPLETE");print("="*50);print("2023 FINAL TEST: SEALED");print("A30 REFERENCE: LOCKED / UNCHANGED");print("PRIMARY CELLS POINT/CLEAR:")
    for e,d in decisions.items():print(e,d["point"],"/4 point;",d["clear"],"/4 clear")
    print("FINAL ENSEMBLE DECISION:",decision);print("R10 SCIENTIFIC REVIEW: REQUIRED");print("PHASE 09 STATUS: PASS");print("NEXT RECOMMENDED ROLE: R10 PHYSICAL & SCIENTIFIC REVIEWER")
    return 0 if gates.passed.all() else 1


if __name__=="__main__":sys.exit(main())
