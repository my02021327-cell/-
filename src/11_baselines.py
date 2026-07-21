"""베이스라인 B0–B4 실행 (PROMPT 2 §2)."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from adpipe.config import get_modeling_data
from adpipe.baselines import run_baselines
from adpipe.report_util import save_json, log_decision

d = get_modeling_data("genus")
res = run_baselines(d, seeds=(0, 1, 2, 3, 4))
table = {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in res.items()}
save_json("artifacts/baselines.json", table)

print("베이스라인 (LOGO, site 그룹 CV):")
for k, v in res.items():
    print(f"  {k:12s} R²={v['r2_mean']:+.3f} ± {v['r2_sd']:.3f}   ({v['note']})")
log_decision("11_baselines",
             "B0~B4 평가 완료. B4(무작위 특징)와 B0 를 기준선으로 고정. 이후 모델은 이를 이겨야 함.")
