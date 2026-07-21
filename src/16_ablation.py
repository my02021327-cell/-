"""특징 블록 절제 실험 S1–S5 (PROMPT 2 §6)."""
import os, sys, json
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.config import get_modeling_data
from adpipe.pipeline import PipelineFactory
from adpipe.ablation import run_ablation
from adpipe.report_util import save_json, log_decision

winner = json.load(open("artifacts/winner.json"))["winner"]
d = get_modeling_data("genus")
fac = PipelineFactory(d["taxonomy"], d["domain_taxa"], d["meta"])
abl = run_ablation(winner, fac, d, seeds=(0, 1, 2))
save_json("artifacts/ablation.json", abl)

print(f"절제 실험 (모델={winner}, 3 seed 중첩 CV):")
for name, r in abl["results"].items():
    print(f"  {name:16s} blocks={r['blocks']}  R²={r['r2_mean']:+.3f} ± {r['r2_sd']:.3f}")
print(f"S5(전체) vs S1(계절) Wilcoxon p={abl['s5_vs_s1_wilcoxon_p']:.3f} (site={abl['n_sites']})")
log_decision("16_ablation",
             f"S1~S5 절제(모델={winner}). S5 vs S1 Wilcoxon p={abl['s5_vs_s1_wilcoxon_p']:.3f}.")
