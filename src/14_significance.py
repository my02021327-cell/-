"""통계적 유의성 검정 (PROMPT 2 §4): 순열검정 · 부트스트랩 CI · 모델비교 · 학습곡선."""
import os, sys, json, pickle
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adpipe.config import get_modeling_data
from adpipe.pipeline import PipelineFactory
from adpipe.significance import (permutation_test, bootstrap_r2_ci, model_comparison,
                                 learning_curve_sites)
from adpipe.report_util import save_json, log_decision

nested = pickle.load(open("artifacts/nested.pkl", "rb"))
winner = nested["winner"]
cfg = dict(nested["results"][winner]["modal_cfg"])
y = np.array(nested["y"]); groups = np.array(nested["groups"])
d = get_modeling_data("genus")
fac = PipelineFactory(d["taxonomy"], d["domain_taxa"], d["meta"])

# 1) 순열검정 (site 그룹 보존 셔플)
perm = permutation_test(cfg, fac, d, n_perms=500)
# 2) 부트스트랩 CI (site 단위) — winner seed0 OOF
_seed0 = nested["seeds"][0]
_ob = nested["results"][winner]["oof_by_seed"]
oof0 = np.array(_ob.get(_seed0, _ob.get(str(_seed0))))
boot = bootstrap_r2_ci(y, oof0, groups, n_boot=2000)
# 3) 모델 간 비교 (site 단위 MAE, Wilcoxon + BH)
per_site_mae = {m: nested["results"][m]["per_site_mae"] for m in nested["results"]}
cmp = model_comparison(per_site_mae, winner)
# 4) 학습곡선 (site 3→9)
lc = learning_curve_sites(cfg, fac, d, sizes=range(3, 10), n_draws=10)

save_json("artifacts/significance.json", dict(
    winner=winner, permutation=dict(score=perm["score"], pvalue=perm["pvalue"],
                                    perm_mean=perm["perm_mean"]),
    bootstrap=boot, model_comparison=cmp, learning_curve=lc))

# 그림: 순열 귀무분포 (라벨 영문 — 한글 글리프 tofu 방지)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(perm["perm_scores"], bins=30, color="#BBBBBB", label="null (permutation)")
ax.axvline(perm["score"], color="crimson", lw=2, label=f"observed R2={perm['score']:.2f}")
ax.set_xlabel("R2"); ax.set_title(f"Permutation test - {winner} (p={perm['pvalue']:.3f})")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("figures/permutation_null.png", dpi=120); plt.close(fig)

# 그림: 학습곡선
if lc:
    ks = [c[0] for c in lc]; ms = [c[1] for c in lc]; sd = [c[2] for c in lc]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ks, ms, yerr=sd, marker="o", capsize=3, color="#2E75B6")
    ax.set_xlabel("# training sites"); ax.set_ylabel("held-out R2")
    ax.set_title(f"Learning curve - {winner}"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/learning_curve.png", dpi=120); plt.close(fig)

# 그림: 예측-관측 산점도(site 색상)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
import pandas as pd
sc = ax.scatter(y, oof0, c=pd.Categorical(groups).codes, cmap="tab10", s=40,
                edgecolor="k", linewidth=0.3)
lim = [min(y.min(), np.nanmin(oof0)), max(y.max(), np.nanmax(oof0))]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("observed methane"); ax.set_ylabel("predicted (OOF)")
ax.set_title(f"Predicted vs observed - {winner}")
fig.tight_layout(); fig.savefig("figures/pred_vs_obs.png", dpi=120); plt.close(fig)

print(f"순열검정: 실제R²={perm['score']:.3f}, p={perm['pvalue']:.3f} (귀무평균={perm['perm_mean']:.3f})")
print(f"부트스트랩 R² CI: {boot['point']:.3f} [{boot['lo']:.3f}, {boot['hi']:.3f}]")
print(f"모델비교 fold(site)수={cmp['n_folds']} — 검정력 낮음(경고 병기)")
log_decision("14_significance",
             f"순열 p={perm['pvalue']:.3f}, 부트스트랩 R² CI=[{boot['lo']:.2f},{boot['hi']:.2f}]. "
             "10 fold 다중비교 검정력 한계 명시.")
