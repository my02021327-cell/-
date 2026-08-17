"""
파이프라인 실행 — PROMPT §13 작업 순서 그대로.

    python -m src.bioguard.run_pipeline

산출: outputs/results_3track.json, outputs/report_3track.html,
      outputs/health_program.json
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import health, physics
from .config import (CV_HORIZON_D, CV_INITIAL_TRAIN_D, CV_STEP_D, HOLDOUT_START, K_GRID,
                     K_HYD, REPORTED_BASELINES, SEED, SRT_REFERENCE_D, SUBSTRATES, SUB_KR,
                     V_DIGESTER_M3, V_SENSITIVITY)
from .cv import (fold_rmse, make_folds, metrics, oof_predictions, paired_test, valid_folds)
from .data import build_frame, target_coverage
from .gpu_backend import BK
from .kernels import front_set, kernel_stats
from .empirical import make_M1_feed, make_empirical, reconstruction_check
from .models import (T3_VARS, combine, design_matrix, final_weights, m5_intercept_path,
                     make_M1, make_M2, make_M3, make_M4, make_M5, nnls_fit, stoichiometry,
                     t3_features)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
np.random.seed(SEED)


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    R: dict = {"실행환경": {"backend": BK.info(), "python": platform.python_version(), "seed": SEED}}

    # ---------------------------------------------------------------- §13-1
    _log("\n[1] 데이터 적재 · 품질 · 재현 검사")
    F = build_frame()
    y = F.ch4.to_numpy(float)
    n = len(F)
    R["데이터"] = {
        "출처": "data/영천BGP_MASTER_2018-2023.csv (PROMPT §1 의 7개 파일 미제공 — 대체)",
        "기간": [str(F.date.min().date()), str(F.date.max().date())], "일수": n,
        "타깃": target_coverage(F),
        "반입구성비_평균_pct": {SUB_KR[s]: round(float(F[f"ratio_{s}"].mean() * 100), 1)
                          for s in SUBSTRATES},
    }
    _log(f"    n={n}  타깃 관측 {R['데이터']['타깃']['관측일']}일 "
         f"(결측 {R['데이터']['타깃']['결측률_pct']}%)")

    # §9 재현 검사: SRT 12일·무절편 → 원본 θ = 40.076 / 52.111 / 13.407
    W12 = design_matrix(F, "기존", 12.0, K_HYD, intercept=False, basis="tonnage")
    tr_idx = np.where(np.isfinite(y) & (F.date <= "2021-12-31").to_numpy())[0]
    th12 = nnls_fit(W12, y, tr_idx)
    pred12 = W12[tr_idx] @ th12
    r2_12 = 1 - ((y[tr_idx] - pred12) ** 2).sum() / ((y[tr_idx] - y[tr_idx].mean()) ** 2).sum()
    R["재현검사_§9"] = {
        "설정": "SRT 12일 · 무절편 · 기존 전단(τ_mix=3) · 학습 2018~2021",
        "원본_θ": {"음폐수": 40.076, "가축분뇨": 52.111, "음식물": 13.407, "학습R2": 0.293},
        "재계산_θ": {SUB_KR[s]: round(float(v), 3) for s, v in zip(SUBSTRATES, th12)},
        "재계산_학습R2": round(float(r2_12), 3),
        "판정": "원본 θ 는 제공되지 않은 `01_일별_소화조상태.csv` 의 파생 투입톤 기준이라 "
              "정확 재현 불가. 우리는 PROMPT 금지사항 6 에 따라 반입 원자료에서 출발하므로 "
              "설계행렬 자체가 다르다. 이 차이는 아래 모든 결과에 일관되게 적용된다.",
    }
    _log(f"    §9 재현: θ={R['재현검사_§9']['재계산_θ']}  학습R²={r2_12:.3f}")

    # ---------------------------------------------------------------- §13-2
    _log("[2] §4 VS 물질수지 이탈 재현")
    R["물질수지_§4"] = {str(int(v)): physics.vs_balance(build_frame(v_digester=v), v)
                    for v in V_SENSITIVITY}
    b8 = R["물질수지_§4"]["8000"]
    _log(f"    V=8000: 소비VS {b8['소비VS_kgd']:,} kg/d, 분해율 {b8['VS분해율_pct']}%, "
         f"함축수율 {b8['함축수율_소비VS기준']} (이론 {b8['이론상한_소비VS기준']}) "
         f"→ {b8['초과배수']}배 초과")

    # ---------------------------------------------------------------- §13-3
    _log("[3] rolling-origin 폴드 고정")
    folds = valid_folds(make_folds(n), y)
    R["검증프로토콜"] = {"방식": "확장창 rolling-origin", "초기학습일": CV_INITIAL_TRAIN_D,
                   "예측창": CV_HORIZON_D, "원점전진": CV_STEP_D, "폴드수": len(folds),
                   "폴드수_주석": f"격자상 19개 중 {19 - len(folds)}개는 학습 라벨 200일 미만으로 "
                              f"제외됐다(타깃 결측률 {R['데이터']['타깃']['결측률_pct']}%). "
                              "PROMPT §8 은 19폴드를 말하지만 그 계열은 결측 63일 기준이다.",
                   "폴드": [{"train_end": str(F.date.iloc[te[0] - 1].date()),
                           "test": [str(F.date.iloc[te[0]].date()), str(F.date.iloc[te[-1]].date())],
                           "n_test_obs": int(np.isfinite(y[te]).sum())} for _, te in folds]}
    _log(f"    폴드 {len(folds)}개")

    # ---------------------------------------------------------------- §13-4
    _log("[4] 기준선 4종 재현")
    base_specs = {
        "기존전단_무절편": ("기존", False), "기존전단_절편": ("기존", True),
        "문헌전단_무절편": ("표준", False), "문헌전단_절편": ("표준", True),
    }
    base_r = {}
    for nm, (scen, ic) in base_specs.items():
        base_r[nm] = fold_rmse(make_M1(F, y, scen, SRT_REFERENCE_D, K_HYD, ic), folds, y)
    R["기준선_§5.5"] = {
        "설명": "선행 보고값과 우리 재계산. 타깃 계열(실측 CH₄ 관측일만)과 설계행렬이 달라 "
              "절대값은 일치하지 않는다 — 비교는 반드시 우리 재계산 기준선으로 한다.",
        "표": [{"사양": nm, "선행보고_RMSE": REPORTED_BASELINES[nm],
               "재계산_RMSE": round(float(np.nanmean(v)), 1),
               "폴드SD": round(float(np.nanstd(v, ddof=1)), 1)} for nm, v in base_r.items()],
        "전단효과": [
            paired_test(base_r["문헌전단_무절편"], base_r["기존전단_무절편"], "문헌전단", "기존전단"),
            paired_test(base_r["문헌전단_절편"], base_r["기존전단_절편"], "문헌전단절편", "기존전단절편"),
            paired_test(base_r["문헌전단_절편"], base_r["문헌전단_무절편"], "절편있음", "절편없음"),
        ],
    }
    # 전단 제거 ablation (§5.4 대조)
    none_r = fold_rmse(make_M1(F, y, "없음", SRT_REFERENCE_D, K_HYD, True), folds, y)
    R["기준선_§5.5"]["전단제거_대조"] = paired_test(base_r["문헌전단_절편"], none_r, "문헌전단", "전단없음")
    BASELINE = base_r["문헌전단_절편"]
    R["기준선_§5.5"]["채택기준선"] = {"사양": "문헌전단_절편",
                              "CV_RMSE": round(float(np.nanmean(BASELINE)), 1)}
    for row in R["기준선_§5.5"]["표"]:
        _log(f"    {row['사양']:16s} 선행 {row['선행보고_RMSE']:>7} → 재계산 {row['재계산_RMSE']:>7}")

    # 전단 커널 요약 + 시나리오 민감도
    R["전단커널"] = {sc: {SUB_KR[s]: kernel_stats(g) for s, g in front_set(sc).items()}
                 for sc in ("최소", "표준", "최대")}
    scen_r = {sc: fold_rmse(make_M1(F, y, sc, SRT_REFERENCE_D, K_HYD, True), folds, y)
              for sc in ("최소", "표준", "최대")}
    R["전단시나리오_민감도"] = [{"시나리오": sc, "CV_RMSE": round(float(np.nanmean(v)), 1),
                        **{k: v2 for k, v2 in paired_test(v, BASELINE, sc, "표준").items()
                           if k in ("ΔRMSE", "SE", "p", "판정")}} for sc, v in scen_r.items()]

    # ---------------------------------------------------------------- §13-5
    _log("[5] 트랙별 구축 · 기준선 대비 검정")
    models = {
        "M1_기계론합성곱": make_M1(F, y, "표준", SRT_REFERENCE_D, K_HYD, True),
        "M1F_실측투입구동": make_M1_feed(F, y),
        "M2_ADL": make_M2(F, y, "표준", SRT_REFERENCE_D, K_HYD),
        "M3_VS물질수지": make_M3(F, y, True),
        "M4_트리_lag격자": make_M4(F, y, "forecast", True),
        "M5_상태공간칼만": make_M5(F, y, "표준", SRT_REFERENCE_D, K_HYD),
    }
    fr, oof = {}, {}
    for nm, fp in models.items():
        fr[nm] = fold_rmse(fp, folds, y)
        oof[nm] = oof_predictions(fp, folds, y, n)
        _log(f"    {nm:18s} CV-RMSE {np.nanmean(fr[nm]):8.1f}")
    m2 = models["M2_ADL"]
    R["M2_함축HRT"] = {
        "φ": round(getattr(m2, "last_phi", float("nan")), 4),
        "함축HRT_d": (round(getattr(m2, "last_implied_hrt", float("nan")), 1)
                   if getattr(m2, "last_implied_hrt", None) else None),
        "문헌대조": "설계 HRT 40.5일 (V=8,000㎥ / Q≈197 t/d)",
        "주의": "저장소 명세 W1 은 y_lag1 사용을 금지한다(HRT 관성이 만드는 허위 R²). "
              "PROMPT §7 이 M2 를 명시 요구해 구현했으나, 90일 앞 예측에서는 y(t−1) 을 알 수 "
              "없으므로 재귀 시뮬레이션으로 굴렸다. 관측 lag 를 그대로 넣으면 그것이 곧 누출이다.",
    }
    R["모델계열"] = {
        "표": [{"모델": nm, "트랙": t, "CV_RMSE": round(float(np.nanmean(v)), 1),
               "폴드SD": round(float(np.nanstd(v, ddof=1)), 1),
               "유효폴드": int(np.isfinite(v).sum()),
               **({"ΔRMSE": None, "SE": None, "p": None, "판정": "기준선 자체 (동일 모델)"}
                  if nm == "M1_기계론합성곱" else
                  {k: v2 for k, v2 in paired_test(v, BASELINE, nm, "기준선").items()
                   if k in ("ΔRMSE", "SE", "p", "판정")})}
              for (nm, v), t in zip(fr.items(),
                                    ["T1", "T1", "T1", "T2", "T1+T3", "T2+T3"])],
    }

    # 현장 경험식 벤치마크 + 구동 변수 진단 (docs/영천BGP_메탄발생량_예측경험식.md)
    emp_now = fold_rmse(make_empirical(F, y, "nowcast"), folds, y)
    emp_fc = fold_rmse(make_empirical(F, y, "forecast"), folds, y)
    R["경험식_대조"] = {
        "출처": "docs/영천BGP_메탄발생량_예측경험식.md",
        "Nowcast_CV_RMSE": round(float(np.nanmean(emp_now)), 1),
        "Forecast_CV_RMSE": round(float(np.nanmean(emp_fc)), 1),
        "검정_forecast_vs_기준선": paired_test(emp_fc, BASELINE, "경험식", "기준선"),
        "구동변수_진단": reconstruction_check(F),
        "상세": "scripts/compare_empirical.py → outputs/compare_empirical.json",
    }

    # T1 계수와 물리 검사
    W = design_matrix(F, "표준", SRT_REFERENCE_D, K_HYD, True, "tonnage")
    obs = np.where(np.isfinite(y))[0]
    th = nnls_fit(W, y, obs)
    theta = {s: float(v) for s, v in zip(SUBSTRATES, th[:3])}
    R["T1_계수"] = {"기준": "㎥CH₄ / 반입 t", "θ": {SUB_KR[s]: round(v, 3) for s, v in theta.items()},
                 "절편_m3d": round(float(th[3]), 1),
                 "절편비중_pct": round(float(100 * th[3] / np.nanmean(y)), 1)}
    Wv = design_matrix(F, "표준", SRT_REFERENCE_D, K_HYD, True, "vs")
    thv = nnls_fit(Wv, y, obs)
    theta_vs = {s: float(v) for s, v in zip(SUBSTRATES, thv[:3])}
    R["T1_수율기준_계수"] = {"기준": "㎥CH₄ / kgVS_in",
                      "Y": {SUB_KR[s]: round(v, 4) for s, v in theta_vs.items()},
                      "절편_m3d": round(float(thv[3]), 1)}
    R["화학양론"] = stoichiometry()

    # PROMPT §3 T1-4 이탈 판정 — 억지로 맞추지 않고 전환 경로를 따른다
    st_lit = stoichiometry()["기질"]
    bd_fit = {s: (theta_vs[s] / st_lit[s]["B_th"] if st_lit[s]["B_th"] else np.nan)
              for s in SUBSTRATES}
    ic_frac = float(th[3] / np.nanmean(y))
    R["T1_이탈판정_§3"] = {
        "BD_적합값": {SUB_KR[s]: round(float(v), 3) for s, v in bd_fit.items()},
        "발화": [
            {"지표": "BD_j > 1", "값": f"음폐수 BD={bd_fit['foodww']:.3f}",
             "임계": "물리적 불가", "발화": bool(bd_fit["foodww"] > 1),
             "전환": "화학양론 강제를 풀고 θ_j 를 자유 계수로 — 본 구현은 처음부터 NNLS 자유계수다"},
            {"지표": "절편 > 실측의 15%", "값": f"{100*ic_frac:.1f}%",
             "임계": "15%", "발화": bool(ic_frac > 0.15),
             "전환": "T1 단독 예측을 중단하고 T2·T3 결합으로 이관 → 앙상블에서 M1 가중 0 으로 "
                   "자동 배제됨(§7 결과와 일치)"},
        ],
        "해석": "NNLS 가 가축분뇨·음식물 계수를 0 으로 밀고 절편이 실측의 42.9% 를 가져간다. "
              "이는 PROMPT §5.3 의 '미설명 메탄' 을 더 크게 재확인한 것이며, 추적 3기질의 "
              "반입 물량만으로는 메탄 수준이 설명되지 않는다는 뜻이다. 계수를 수율로 해석 금지.",
    }

    # T2 상대 예측력 (절편 허용 시 기울기는 유효한가)
    m3_fp = models["M3_VS물질수지"]
    R["T2_상대예측력"] = {
        "주의": "함축수율이 이론상한의 2배 → 절대수준 예측 불가. 절편 허용 하에 기울기만 평가.",
        **{k: v for k, v in paired_test(fr["M3_VS물질수지"], BASELINE, "M3", "기준선").items()},
    }

    # T3 Nowcast vs Forecast
    now_r = fold_rmse(make_M4(F, y, "nowcast", True), folds, y)
    R["T3_nowcast_vs_forecast"] = {
        "Nowcast_CV_RMSE": round(float(np.nanmean(now_r)), 1),
        "Forecast_CV_RMSE": round(float(np.nanmean(fr["M4_트리_lag격자"])), 1),
        "검정": paired_test(now_r, fr["M4_트리_lag격자"], "Nowcast", "Forecast"),
        "주의": "Nowcast 는 동시점 내부 이화학을 쓴다 — 예측 성능 비교에 쓰지 않는다(§8-5).",
    }

    # ---------------------------------------------------------------- §13-6
    _log("[6] 변수 채택/배제 판정 (§6)")
    base_t3 = fold_rmse(make_M4(F, y, "forecast", False), folds, y)   # T3 없는 트리
    adopt = []
    for v in T3_VARS:
        keep = [c for c in t3_features(F, "forecast").columns if c.startswith(v)]
        r = fold_rmse(_m4_subset(F, y, keep), folds, y)
        t = paired_test(r, base_t3, f"+{v}", "T3없음")
        adopt.append({"변수": v, "CV_RMSE": round(float(np.nanmean(r)), 1),
                      "ΔRMSE": t.get("ΔRMSE"), "SE": t.get("SE"), "p": t.get("p"),
                      "채택": bool(t.get("유의") and (t.get("ΔRMSE") or 0) < 0),
                      "판정": t.get("판정")})
        _log(f"    {v:16s} ΔRMSE {t.get('ΔRMSE'):>8} p={t.get('p')}  "
             f"{'채택' if adopt[-1]['채택'] else '배제→건강지표'}")
    R["변수채택_§6"] = {"기준": "rolling-origin 포함/제외 대응 t검정. p≥0.05 또는 개선 아니면 배제",
                    "T3없음_CV_RMSE": round(float(np.nanmean(base_t3)), 1), "판정": adopt,
                    "배제변수_재배치": [a["변수"] for a in adopt if not a["채택"]]}

    # ---------------------------------------------------------------- §13-7
    _log("[7] 앙상블")
    ens = {m: combine(oof, y, folds, m) for m in ("simple", "perf", "nnls")}
    ens_fold = {}
    for m, p in ens.items():
        ens_fold[m] = np.array([_rmse_at(y, p, te) for _, te in folds])
    R["앙상블"] = {
        "구성": list(models),
        "표": [{"결합": {"simple": "단순평균", "perf": "성능가중평균", "nnls": "NNLS 스태킹"}[m],
               "CV_RMSE": round(float(np.nanmean(v)), 1),
               **{k: v2 for k, v2 in paired_test(v, BASELINE, m, "기준선").items()
                  if k in ("ΔRMSE", "SE", "p", "판정")}} for m, v in ens_fold.items()],
        "최종_스태킹_가중치": final_weights(oof, y),
        "메타학습_규약": "폴드 i 의 결합계수는 폴드 <i 의 OOF 만으로 적합 — 누출 없음(§7)",
    }
    best = min(ens_fold, key=lambda m: np.nanmean(ens_fold[m]))
    R["앙상블"]["채택"] = {"simple": "단순평균", "perf": "성능가중평균", "nnls": "NNLS 스태킹"}[best]
    R["앙상블"]["채택_검정"] = paired_test(ens_fold[best], BASELINE, "앙상블", "기준선")
    for row in R["앙상블"]["표"]:
        _log(f"    {row['결합']:12s} {row['CV_RMSE']:>8}  Δ{row['ΔRMSE']:>8} p={row['p']} {row['판정']}")

    # ---------------------------------------------------------------- §13-8
    _log("[8] ablation · 잔차 진단 · 물리 검사")
    abl = [{"제거": "전단 커널", "CV_RMSE": round(float(np.nanmean(none_r)), 1),
            **{k: v for k, v in paired_test(none_r, BASELINE, "전단없음", "기준선").items()
               if k in ("ΔRMSE", "p", "판정")}},
           {"제거": "절편", "CV_RMSE": round(float(np.nanmean(base_r["문헌전단_무절편"])), 1),
            **{k: v for k, v in paired_test(base_r["문헌전단_무절편"], BASELINE, "무절편", "기준선").items()
               if k in ("ΔRMSE", "p", "판정")}},
           {"제거": "T3 변수군", "CV_RMSE": round(float(np.nanmean(base_t3)), 1),
            **{k: v for k, v in paired_test(base_t3, fr["M4_트리_lag격자"], "T3없음", "M4").items()
               if k in ("ΔRMSE", "p", "판정")}}]
    for drop in models:
        sub = {k: v for k, v in oof.items() if k != drop}
        p = combine(sub, y, folds, best)
        v = np.array([_rmse_at(y, p, te) for _, te in folds])
        abl.append({"제거": drop, "CV_RMSE": round(float(np.nanmean(v)), 1),
                    **{k: v2 for k, v2 in paired_test(v, ens_fold[best], f"−{drop}", "앙상블").items()
                       if k in ("ΔRMSE", "p", "판정")}})
    R["ablation"] = abl

    ens_best = ens[best]
    resid = y - ens_best
    R["잔차진단"] = _residual_diag(F, y, ens_best, resid)

    fronts = front_set("표준")
    R["물리검사_§9"] = {
        "톤수기준": physics.check_coefficients(theta, float(th[3]), float(np.nanmean(y)), "tonnage"),
        "VS기준": physics.check_coefficients(theta_vs, float(thv[3]), float(np.nanmean(y)), "vs"),
        "커널": physics.check_kernels(fronts),
        "VS수율상한": {"함축수율_소비VS기준": b8["함축수율_소비VS기준"],
                  "기준": "≤ 0.50 ㎥CH₄/kgVS_destroyed",
                  "통과": bool(b8["함축수율_소비VS기준"] <= 0.50)},
    }
    R["비식별성_§5.2"] = physics.identifiability(SRT_REFERENCE_D, K_HYD)

    # SRT / k 민감도
    R["SRT_민감도"] = []
    for v in V_SENSITIVITY:
        srt = v / float(np.nanmean(F.feed_AB))
        rr = fold_rmse(make_M1(F, y, "표준", srt, K_HYD, True), folds, y)
        R["SRT_민감도"].append({"V_m3": v, "SRT_d": round(srt, 1),
                            "CV_RMSE": round(float(np.nanmean(rr)), 1),
                            **{k: v2 for k, v2 in paired_test(rr, BASELINE, f"V{int(v)}", "기준(SRT25)").items()
                               if k in ("ΔRMSE", "p", "판정")}})
    R["k_민감도"] = []
    for kf in K_GRID["foodww"]:
        kk = dict(K_HYD, foodww=kf)
        rr = fold_rmse(make_M1(F, y, "표준", SRT_REFERENCE_D, kk, True), folds, y)
        R["k_민감도"].append({"k_음폐수": kf, "CV_RMSE": round(float(np.nanmean(rr)), 1)})

    # M5 시변 절편 궤적
    level, _ = m5_intercept_path(F, y)
    R["M5_시변절편"] = {
        "설명": "§5.3 의 '미설명 메탄 상수'를 상수가 아닌 느리게 변하는 상태로 추적한 궤적",
        "dates": F.date.dt.strftime("%Y-%m-%d").tolist(),
        "level": [round(float(x), 1) for x in level],
        "연평균": {str(int(yr)): round(float(np.nanmean(level[(F.year == yr).to_numpy()])), 1)
                for yr in sorted(F.year.unique())},
    }

    # 최종 hold-out 1회 보고 (모델 선택에 쓰지 않음 — §8-2)
    ho = (F.date >= HOLDOUT_START).to_numpy()
    R["holdout_최종보고"] = {"구간": f"{HOLDOUT_START}~{F.date.max().date()}",
                        "주의": "모델 선택에 쓰지 않았다. 최종 1회 보고용.",
                        "앙상블": metrics(y[ho], ens_best[ho]),
                        "기준선_M1": metrics(y[ho], oof["M1_기계론합성곱"][ho])}

    # 예측 시계열 + 예측구간
    sd = float(np.nanstd(resid, ddof=1))
    R["예측시계열"] = {"dates": F.date.dt.strftime("%Y-%m-%d").tolist(),
                  "obs": _js(y), "pred": _js(ens_best),
                  "lo": _js(ens_best - 1.96 * sd), "hi": _js(ens_best + 1.96 * sd),
                  "pi_sd": round(sd, 1),
                  "note": "예측구간은 OOF 잔차 표준편차 기반 ±1.96σ (정규 근사)"}

    # ---------------------------------------------------------------- §13-9
    _log("[9] 건강상태 프로그램 · 리포트")
    HP = health.build(F)
    HP["배제되어_재배치된_변수"] = R["변수채택_§6"]["배제변수_재배치"]
    (OUT / "health_program.json").write_text(json.dumps(HP, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    R["건강상태"] = {"지표": HP["지표"], "종합": HP["종합"], "이상구간": HP["이상구간"],
                 "주의사항": HP["주의사항"]}

    R["한계와_다음단계"] = [
        "SRT 는 여전히 외생 입력이다. 확정하려면 리튬 등 추적자 펄스 시험이 필요하다 — "
        "본 자료로는 여섯 경로 전부 비식별(§5.2)이며 본 실행의 V 민감도도 이를 재확인했다.",
        f"VS 물질수지 이탈(함축수율 {b8['함축수율_소비VS기준']} vs 이론 0.50)의 정체 규명이 최우선이다. "
        "모델 보정이 아니라 계량 감사 — 가스 유량계 단위(Nm³/㎥)·온도압력 보정과 유입 VS 계량 누락부터.",
        "환경부 2025 통합 바이오가스화 지침 원문과 실제 조 용적(유기산화조 유효용적 포함)을 "
        "확보하면 전단 커널과 V 를 갱신해야 한다. 현재 유기산화조 용적은 공정도·워크북 모두 미기재다.",
        "PROMPT §1 의 7개 데이터 파일이 제공되지 않아 master CSV 로 대체했다. 선행 기준선 "
        "861.5 ㎥/d 와 직접 비교할 수 없으므로 동일 폴드에서 기준선을 재구성해 비교했다.",
    ]

    (OUT / "results_3track.json").write_text(json.dumps(R, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    from .report import render
    render(R, HP, OUT / "report_3track.html")
    _log(f"\n완료 → {OUT/'results_3track.json'}\n      {OUT/'report_3track.html'}"
         f"\n      {OUT/'health_program.json'}")
    return 0


# --------------------------------------------------------------------------- helpers
def _m4_subset(F, y, keep_cols):
    """T3 변수 하나만 추가한 트리 (§6 포함/제외 검정용)."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from .models import t1_lag_features

    X = pd.concat([t1_lag_features(F), t3_features(F, "forecast")[keep_cols]], axis=1)
    Xv = X.to_numpy(float)

    def fit_predict(tr, te):
        ok = np.isfinite(y[tr])
        if ok.sum() < 50:
            return np.full(len(te), np.nan)
        m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=SEED)
        m.fit(Xv[tr][ok], y[tr][ok])
        return m.predict(Xv[te])

    return fit_predict


def _rmse_at(y, p, te):
    m = np.isfinite(y[te]) & np.isfinite(p[te])
    if m.sum() < 2:
        return np.nan
    r = y[te][m] - p[te][m]
    return float(np.sqrt(r @ r / len(r)))


def _js(a):
    return [None if not np.isfinite(x) else round(float(x), 1) for x in a]


def _residual_diag(F, y, pred, resid):
    ok = np.isfinite(resid)
    r = pd.Series(resid)
    out = {"n": int(ok.sum()), "평균": round(float(np.nanmean(resid)), 1),
           "SD": round(float(np.nanstd(resid, ddof=1)), 1),
           "자기상관": {f"lag{l}": round(float(r.autocorr(l)), 3) for l in (1, 7, 30)}}
    out["연도별_RMSE"] = {}
    for yr in sorted(F.year.unique()):
        rr = resid[(F.year == yr).to_numpy()]
        rr = rr[np.isfinite(rr)]
        out["연도별_RMSE"][str(int(yr))] = (round(float(np.sqrt((rr ** 2).mean())), 1)
                                         if len(rr) else "OOF 없음(초기 학습창)")
    out["요일별_평균잔차"] = {int(d): round(float(np.nanmean(resid[(F.dow == d).to_numpy()])), 1)
                       for d in range(7)}
    q = pd.qcut(F.feed_AB, 4, labels=False, duplicates="drop")
    out["부하4분위_평균잔차"] = {int(k): round(float(np.nanmean(resid[(q == k).to_numpy()])), 1)
                        for k in sorted(pd.Series(q).dropna().unique())}
    mix = pd.qcut(F.ratio_manure, 4, labels=False, duplicates="drop")
    out["가축분뇨비중4분위_평균잔차"] = {int(k): round(float(np.nanmean(resid[(mix == k).to_numpy()])), 1)
                              for k in sorted(pd.Series(mix).dropna().unique())}
    return out


if __name__ == "__main__":
    sys.exit(main())
