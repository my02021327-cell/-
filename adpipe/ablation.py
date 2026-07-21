"""특징 블록 절제 실험 (PROMPT 2 §6).

S5(전체)가 S1(계절만)보다 유의하게 낫지 않으면, 마이크로바이옴이 계절 이상을 주지 못한다는
뜻이다 — 실망이 아니라 발표 가치가 있는 결론.
"""
import numpy as np
from .cv import nested_cv_oof, seed_r2_summary, per_site_scores
from scipy import stats

ABLATION_SETS = {
    "S1_design":      ["design"],
    "S2_alpha":       ["alpha"],
    "S3_taxa":        ["taxa"],
    "S4_taxa_alpha":  ["taxa", "alpha"],
    "S5_full":        ["taxa", "alpha", "beta", "domain", "design"],
}


def run_ablation(model_key, factory, data, seeds=(0, 1, 2)):
    results = {}
    per_site = {}
    for name, blocks in ABLATION_SETS.items():
        res = nested_cv_oof(model_key, factory, data, feature_blocks=blocks, seeds=seeds)
        summ = seed_r2_summary(res)
        oof0 = res["oof"][seeds[0]]
        results[name] = dict(blocks=blocks, r2_mean=summ["mean"], r2_sd=summ["sd"],
                             values=summ["values"])
        per_site[name] = per_site_scores(res["y"], oof0, res["groups"], metric="mae")
    # S5 vs S1 Wilcoxon(site 단위 MAE)
    sites = sorted(set(per_site["S5_full"]) & set(per_site["S1_design"]))
    a = np.array([per_site["S5_full"][s] for s in sites])
    b = np.array([per_site["S1_design"][s] for s in sites])
    try:
        _, p_s5_s1 = stats.wilcoxon(a, b)
    except ValueError:
        p_s5_s1 = 1.0
    return dict(results=results, per_site=per_site,
                s5_vs_s1_wilcoxon_p=float(p_s5_s1), n_sites=len(sites))
