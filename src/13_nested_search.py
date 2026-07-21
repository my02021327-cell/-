"""중첩 탐색: 변환×모델을 내부 루프에서 선택, 외부 LOGO 로 정직 성능 (PROMPT 2 §1,3).

산출:
  artifacts/nested.pkl   : 모델별 결과(honest/optimistic/leaked, oof, per-site, modal cfg)
  artifacts/winner.json  : 승자 모델 + 고정 config
  reports/grid_vs_nested.md : 탐색 그리드(낙관) vs 중첩 CV(정직) 표
"""
import os, sys, pickle, time
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from adpipe.config import get_modeling_data
from adpipe.pipeline import PipelineFactory
from adpipe.models import MODEL_REGISTRY
from adpipe.cv import (nested_cv_oof, seed_r2_summary, metrics_from_oof, modal_config,
                       optimistic_best, random_kfold_r2, make_estimator, per_site_scores)
from adpipe.report_util import save_json, log_decision

SEEDS = (0, 1, 2, 3, 4)
d = get_modeling_data("genus")
fac = PipelineFactory(d["taxonomy"], d["domain_taxa"], d["meta"])
y, groups = np.asarray(d["y"]), d["groups"].values

results = {}
for mk in MODEL_REGISTRY:
    t = time.time()
    try:
        res = nested_cv_oof(mk, fac, d, seeds=SEEDS)
    except Exception as e:
        print(f"{mk:11s} SKIP — 실패: {type(e).__name__}: {e}")
        continue
    summ = seed_r2_summary(res)
    oof0 = res["oof"][SEEDS[0]]
    modal = modal_config(mk, res["chosen"])
    opt_r2, _ = optimistic_best(mk, fac, d)
    ef = lambda modal=modal: make_estimator(modal["model_key"], modal["transform"],
                                            modal["filter_params"], dict(modal["model_kwargs"]), fac)
    leaked = float(np.mean([random_kfold_r2(ef, d["X"], y, seed=s) for s in range(3)]))
    results[mk] = dict(
        honest_mean=summ["mean"], honest_sd=summ["sd"], honest_values=summ["values"],
        pooled=metrics_from_oof(y, oof0),
        per_site_mae=per_site_scores(y, oof0, groups, "mae"),
        modal_cfg=modal, optimistic_r2=float(opt_r2), leaked_r2=leaked,
        oof_by_seed={int(s): res["oof"][s].tolist() for s in res["oof"]},
        chosen=res["chosen"],
    )
    print(f"{mk:11s} honest={summ['mean']:+.3f}±{summ['sd']:.3f}  "
          f"optimistic={opt_r2:+.3f}  leaked={leaked:+.3f}  "
          f"tf={modal['transform']}  ({time.time()-t:.0f}s)")

winner = max(results, key=lambda k: results[k]["honest_mean"])
save_json("artifacts/winner.json", dict(winner=winner, cfg=results[winner]["modal_cfg"],
                                        honest_mean=results[winner]["honest_mean"]))
with open("artifacts/nested.pkl", "wb") as f:
    pickle.dump(dict(results=results, y=y.tolist(), groups=groups.tolist(),
                     seeds=list(SEEDS), winner=winner), f)

# 그리드(낙관) vs 중첩(정직) 표
G = ["# 탐색 그리드(낙관) vs 중첩 CV(정직) — 최종 성능은 중첩 CV 값", "",
     "> ⚠ '탐색 그리드' 열은 test 를 보고 config 를 고른 **선택편향으로 낙관 편향된** 값이다. "
     "'누수(무작위K-fold)' 열은 site 구조를 무시한 **누수된 낙관 추정치**다. "
     "**정직한 성능은 '중첩 CV(정직)' 열 뿐이다.**", "",
     "| 모델 | Tier | 중첩 CV 정직 R² | 탐색그리드(낙관) | 무작위K-fold(누수) |",
     "|---|---|---|---|---|"]
for mk in sorted(results, key=lambda k: -results[k]["honest_mean"]):
    r = results[mk]
    star = " ⬅ winner" if mk == winner else ""
    G.append(f"| {mk}{star} | T{MODEL_REGISTRY[mk]['tier']} | "
             f"{r['honest_mean']:+.3f} ± {r['honest_sd']:.3f} | "
             f"{r['optimistic_r2']:+.3f} | {r['leaked_r2']:+.3f} |")
G += ["", f"- **승자: {winner}** (정직 R²={results[winner]['honest_mean']:+.3f}).",
      "- '누수' 열이 '정직' 열보다 큰 격차 = site 반복측정 구조를 무작위 K-fold 가 새어 학습한 증거.",
      "- '탐색그리드'가 '정직'보다 큰 격차 = 소표본 승자의 저주(선택편향)."]
open("reports/grid_vs_nested.md", "w").write("\n".join(G))
print("\n승자:", winner, "| 저장: artifacts/nested.pkl, winner.json, reports/grid_vs_nested.md")
log_decision("13_nested_search",
             f"8모델 중첩 CV(5 seed) 완료. 승자={winner}. 최종 성능은 중첩 외부루프 값으로 고정. "
             f"탐색그리드/무작위K-fold 는 낙관·누수 추정치로만 병기.")
