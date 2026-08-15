# -*- coding: utf-8 -*-
"""§2.12 보완 — 문맥길이 격자 확장과, 실제로 이긴 구성에서의 절제 연구.

본 실행에서 두 가지가 미흡했다.
  (1) 문맥길이 최적이 격자 상한(2·SRT=24일)에 붙었다 → 3·SRT, 4·SRT까지 확장한다.
  (2) 절제 연구를 '논문충실(Q_in 포함)' 변수로 돌렸는데, 정작 persistence를 이긴 것은
      '분모배제' 구성이다. 쓸모 있는 구성에서 각 요소의 기여를 다시 잰다.
"""
import json
import numpy as np

src = open("src/mtsf_yield.py", encoding="utf-8").read()
exec(src.split("# ================================================================ 1) 문맥길이 선택")[0])

J = json.load(open("outputs/mtsf_results.json", encoding="utf-8"))
NOD = [f for f in FEATS if f not in DENOM_FEATS]

print("=== 1) 문맥길이 격자 확장 (H=7, 사고학습 기준) ===")
ext = dict(J["context"]["scan"])
for nm, L in [("3·SRT", 36), ("4·SRT", 48)]:
    r = [run("proposed", L, 7, FEATS, s) for s in SEEDS]
    ext[nm] = {"L": L, "valid_MSE": round(float(np.mean([x["valid"]["MSE"] for x in r])), 6),
               "test_MSE": round(float(np.mean([x["test"]["MSE"] for x in r])), 6)}
for nm, v in ext.items():
    print(f"  {nm:8s} (L={v['L']:2d}일)  사고학습 MSE {v['valid_MSE']:.6f}")
bestn = min(ext, key=lambda k: ext[k]["valid_MSE"])
print(f"  → 확장 후 최적 {bestn} (L={ext[bestn]['L']}일)"
      f"{'   ★여전히 격자 상한' if ext[bestn]['L'] == 48 else ''}")
J["context"]["scan_extended"] = ext
J["context"]["selected_extended"] = ext[bestn]["L"]
J["context"]["at_edge_after_extension"] = bool(ext[bestn]["L"] == 48)

print("\n=== 2) 절제 연구 — 분모배제 구성, H=14 (persistence를 이긴 조건) ===")
L = J["context"]["selected_L"]
abl = {}
for nm, cfg in [("제안모델", {}), ("w/o Patching", {"patch": False}),
                ("w/o Norm", {"revin": False}), ("w/o Temporal", {"temporal": False})]:
    rr = [run("proposed", L, 14, NOD, s, cfg) for s in SEEDS]
    abl[nm] = {"MSE": round(float(np.mean([x["test"]["MSE"] for x in rr])), 6),
               "MAE": round(float(np.mean([x["test"]["MAE"] for x in rr])), 5),
               "R2": round(float(np.mean([x["test"]["R2"] for x in rr])), 4)}
pm = J["models"]["분모배제(Q_in·SRT_d 제외)"]["H14"]["persistence"]["test"]["MSE"]
base = abl["제안모델"]["MSE"]
for nm in abl:
    abl[nm]["MSE_increase_pct"] = round(100 * (abl[nm]["MSE"] - base) / base, 2)
    abl[nm]["skill"] = round(1 - abl[nm]["MSE"] / pm, 4)
    print(f"  {nm:16s} MSE {abl[nm]['MSE']:.6f}  R² {abl[nm]['R2']:+.4f}  "
          f"MSE 증가 {abl[nm]['MSE_increase_pct']:+6.2f}%  skill {abl[nm]['skill']:+.4f}")
J["ablation_nodenom_H14"] = abl

json.dump(J, open("outputs/mtsf_results.json", "w", encoding="utf-8"), ensure_ascii=False)
print("\n갱신: outputs/mtsf_results.json")
