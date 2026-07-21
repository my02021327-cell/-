"""최종 보고서 조립 (PROMPT 2 §8). 한계 먼저 → 베이스라인 → 정직 성능 → 결론."""
import os, sys, json, pickle
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from adpipe.config import get_modeling_data, CONFIG

d = get_modeling_data("genus")
n, p = d["X"].shape
nsite = d["meta"]["site"].nunique()
tname = CONFIG["TARGET"]["name"]

def jload(p, default=None):
    try: return json.load(open(p))
    except Exception: return default

baselines = jload("artifacts/baselines.json", {})
sig = jload("artifacts/significance.json", {})
imp = jload("artifacts/importance.json", {})
abl = jload("artifacts/ablation.json", {})
nested = pickle.load(open("artifacts/nested.pkl", "rb"))
R = nested["results"]; winner = nested["winner"]

L = []
def w(s=""): L.append(s)

w("# 최종 보고서 — 혐기성 소화조 16S RA → methane 예측 (정직한 검증)")
w()
w("## 0. 데이터·타깃 출처")
w(f"- 특징: 16S RA(genus 집계, p={p}), n={n} 시료 = {nsite} site × 4 계절.")
w(f"- 타깃: **methane(메탄생성량)**, targets.xlsx 2022년 3/6/9/12월 1일 값.")
w("- ⚠ **핵심 가정**: 메탄생성량이 지역별로 다르나 현재 미준비 → **'모든 지역 동일'로 가정**. "
  "따라서 타깃은 계절(날짜)에만 의존하고 10개 지역 값이 같다(예측 시뮬레이션).")
w("- ⚠ 결측: 2022-06-01(전 컬럼 결측)은 최근접일 2022-05-31 로 대체(자동 아님, 명시).")
w()

w("## 1. 한계 먼저 (Limitations first)")
w(f"- **유효 표본 = site 수 = {nsite}** (반복측정: 같은 site×4계절). 시료 n={n} 는 명목값.")
w(f"- p/n(genus) = {p/n:.2f}. 소표본 p≫유효n → 과적합·검정력 부족이 지배적 제약.")
w("- 딥러닝/AutoML/대규모 스태킹 **배제**(유효 n=10 에서 검증 불가).")
w("- 모든 성능은 **점추정 단독 금지** → CI·순열 p 동반.")
w("- 🔴 **가정의 직접 귀결**: 타깃이 계절에만 의존하므로 **계절(날짜)만으로 완전 예측**되고, "
  "마이크로바이옴이 계절을 넘어선 정보를 줄 여지가 원리적으로 없다. 아래 결과는 이 점을 확인한다.")
w()

w("## 2. 베이스라인 (모델보다 먼저)")
w()
w("| 베이스라인 | 내용 | R² (mean ± sd, LOGO) |")
w("|---|---|---|")
order = ["B0_mean", "B1_season", "B2_alpha", "B3_pcoa", "B4_random"]
for k in order:
    if k in baselines:
        v = baselines[k]
        w(f"| {k} | {v['note']} | {v['r2_mean']:+.3f} ± {v['r2_sd']:.3f} |")
b1 = baselines.get("B1_season", {}).get("r2_mean", float("nan"))
w()
w(f"- **B1(계절)=R² {b1:+.3f}**. 계절만으로 타깃이 (거의) 완전 예측됨 — 가정의 귀결.")
w("- B4(무작위 특징)·B2·B3 는 ≈0 → 조성/다양성 자체는 계절 밖 신호 없음.")
w()

w("## 3. 중첩 CV 최종 성능 (정직, 외부 루프)")
w()
w("| 모델 | Tier | 정직 R² (mean±sd) | pooled RMSE | pooled MAE |")
w("|---|---|---|---|---|")
for mk in sorted(R, key=lambda k: -R[k]["honest_mean"]):
    r = R[mk]; star = " ⬅" if mk == winner else ""
    w(f"| {mk}{star} | T{'1' if mk in ['elasticnet','pls','spls','gpr','krr','svr'] else '2'} | "
      f"{r['honest_mean']:+.3f} ± {r['honest_sd']:.3f} | {r['pooled']['rmse']:.1f} | {r['pooled']['mae']:.1f} |")
w()
if sig:
    pm = sig.get("permutation", {}); bt = sig.get("bootstrap", {})
    w(f"- 승자 **{winner}** 순열검정: 실제 R²={pm.get('score',float('nan')):+.3f}, "
      f"**p={pm.get('pvalue',float('nan')):.3f}** (귀무평균={pm.get('perm_mean',float('nan')):+.3f}).")
    w(f"- 승자 부트스트랩(site 단위) R² CI: {bt.get('point',float('nan')):+.3f} "
      f"[{bt.get('lo',float('nan')):+.3f}, {bt.get('hi',float('nan')):+.3f}].")
    w(f"- ⚠ 순열 p 가 작아도 이는 **계절 신호**를 반영한다(마이크로바이옴 고유 신호 아님). "
      f"승자조차 B1(계절, R²={b1:+.3f})을 **넘지 못한다** → taxa 는 계절 이상을 못 준다.")
w()

w("## 4. 탐색 그리드(낙관) vs 정직 — 편향 진단")
w()
w("| 모델 | 정직(중첩) | 탐색그리드(낙관·선택편향) | 무작위K-fold(누수) |")
w("|---|---|---|---|")
for mk in sorted(R, key=lambda k: -R[k]["honest_mean"]):
    r = R[mk]
    w(f"| {mk} | {r['honest_mean']:+.3f} | {r['optimistic_r2']:+.3f} | {r['leaked_r2']:+.3f} |")
w()
w("- '누수(무작위K-fold)'≈'정직' → 타깃에 **site 구조가 없어**(지역 동일 가정) 그룹CV·무작위CV 가 "
  "일치. 즉 이 시뮬레이션에선 누수 격차가 사라진다(그룹 CV 는 여전히 방법론적 기본값).")
w("- '탐색그리드'>'정직' 격차 = 소표본 승자의 저주(선택편향). 최종 수치로 쓰면 안 됨.")
w()

if abl:
    w("## 5. 절제 실험 (블록 기여)")
    w()
    w("| 조합 | 블록 | R² (mean±sd) |")
    w("|---|---|---|")
    for name, r in abl["results"].items():
        w(f"| {name} | {','.join(r['blocks'])} | {r['r2_mean']:+.3f} ± {r['r2_sd']:.3f} |")
    w()
    w(f"- S5(전체) vs S1(계절) Wilcoxon p={abl['s5_vs_s1_wilcoxon_p']:.3f} (site={abl['n_sites']}).")
    w("- **S5 가 S1 을 유의하게 넘지 못함** → 마이크로바이옴은 계절 이상을 주지 못한다(가정의 귀결).")
    w()

w("## 6. 안정적 특징 (70% 기준)")
w()
if imp:
    stable = imp.get("stable_70", [])
    w(f"- 5 seed × 10 fold = {imp.get('total_fits','?')}회 순열 중요도 상위10 진입 빈도 집계.")
    if stable:
        w("| taxon(genus) | 선택빈도 |")
        w("|---|---|")
        for r in stable:
            w(f"| {r['taxon']} | {r['selection_freq']:.2f} |")
    else:
        w("- **70% 이상 안정 taxa 없음** → 안정적으로 예측에 기여하는 taxa 는 확인되지 않음(해석 금지).")
    w("- ⚠ 소표본 SHAP/단일 fit 중요도는 불안정 → 본 안정성 기준만 신뢰.")
w()

w("## 7. 부정적 결과 (작동하지 않은 것)")
w("- 커널법(KRR/SVR)·트리(RF/HGB)는 대체로 음수/저조 R² — 소표본 p≫n 에서 불안정.")
w("- 어떤 마이크로바이옴 모델도 **계절 베이스라인(B1)을 넘지 못함**.")
w("- 누수 격차가 관측되지 않음 — 타깃에 site 구조가 없기 때문(지역 동일 가정).")
w()

w("## 8. 결론")
w()
w(f"- **현 표본·가정 하에서 마이크로바이옴 조성이 methane 을 '계절 이상으로' 예측한다는 증거가 없다.**")
w(f"  - 근거: B1(계절)=R² {b1:+.3f} 로 이미 완전 예측되고, 최고 마이크로바이옴 모델"
  f"(**{winner}**, 정직 R²={R[winner]['honest_mean']:+.3f})도 이를 넘지 못함. "
  f"절제 실험에서 S5≈S1.")
w("- 이는 **'모든 지역 methane 동일' 가정의 직접 귀결**이다(타깃이 계절만의 함수). "
  "따라서 이 결과는 마이크로바이옴의 무용을 증명하는 것이 아니라, **현 타깃 정의로는 판별 불가**함을 뜻한다.")
w("- **권고**: 지역별 실측 methane(반복측정, 시료별 변동)을 확보해 타깃을 재정의해야 "
  "마이크로바이옴→methane 예측을 실제로 검정할 수 있다. 검증 프로토콜(그룹 CV)은 완화하지 말 것.")
w()
w("---")
w("### 첨부")
w("- `reports/decision_log.md` (데이터 스누핑 로그)")
w("- `reports/grid_vs_nested.md`, `reports/01_validation.md`, `reports/02_sparsity_profile.md`, "
  "`reports/03_transform_comparison_EDA.md`")
w("- `figures/`: 순열 귀무분포, 학습곡선, 예측-관측(site색), 안정성 선택")

open("reports/final_report.md", "w").write("\n".join(L))
print("작성: reports/final_report.md")
print(f"winner={winner} honest R²={R[winner]['honest_mean']:+.3f}; B1_season={b1:+.3f}")
