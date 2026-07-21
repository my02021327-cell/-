"""특징 중요도 — 안정성 선택 (PROMPT 2 §5)."""
import os, sys, json, pickle
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adpipe.config import get_modeling_data
from adpipe.pipeline import PipelineFactory
from adpipe.importance import stability_selection
from adpipe.report_util import save_json, log_decision

win = json.load(open("artifacts/winner.json"))
cfg = win["cfg"]
d = get_modeling_data("genus")
fac = PipelineFactory(d["taxonomy"], d["domain_taxa"], d["meta"])
res = stability_selection(cfg, fac, d, seeds=(0, 1, 2, 3, 4), top_k=10, n_repeats=8)

freq = res["freq"]
stable = freq[freq["selection_freq"] >= 0.70]
save_json("artifacts/importance.json", dict(
    total_fits=res["total_fits"],
    top20=freq.head(20).to_dict("records"),
    stable_70=stable.to_dict("records"),
    sign_top20=(res["sign"].head(20).to_dict("records") if len(res["sign"]) else []),
))

# 안정성 히트맵(상위 20 taxa 선택빈도)
top = freq.head(20)
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top["taxon"][::-1], top["selection_freq"][::-1], color="#2E75B6")
ax.axvline(0.70, color="crimson", ls="--", lw=1, label="70% stability threshold")
ax.set_xlabel("freq in top-10 permutation importance (5 seed x 10 fold)")
ax.set_title(f"Stability selection - winner={cfg['model_key']} ({res['total_fits']} fits)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("figures/stability_selection.png", dpi=120); plt.close(fig)

print(f"안정성 선택(모델={cfg['model_key']}, {res['total_fits']}회):")
print(f"  70% 이상 안정 taxa: {len(stable)} 개")
for _, r in stable.iterrows():
    print(f"    {r['taxon']:34s} freq={r['selection_freq']:.2f}")
if len(stable) == 0:
    print("    (없음 — 안정적으로 상위권에 드는 taxa 없음 → 해석 금지)")
log_decision("15_importance",
             f"안정성 선택 완료(70% 기준 {len(stable)}개). 소표본 SHAP 불안정 경고 병기.")
