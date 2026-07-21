"""STEP 7 runner — 파이프라인 팩토리 직렬화 + 변환비교 EDA(리포트 03).

EDA 용 fit-on-all 산출물은 figures/eda/*_EDA_DO_NOT_MODEL.* 로만 저장한다(모델 금지).
"""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import warnings, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from adpipe.config import get_modeling_data
from adpipe.transforms import TRANSFORM_REGISTRY
from adpipe.pipeline import build_pipeline, PipelineFactory

LEVEL = "genus"
d = get_modeling_data(LEVEL)
X, meta = d["X"], d["meta"]
ctx = dict(taxonomy=d["taxonomy"], domain_taxa=d["domain_taxa"], meta=meta)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("figures/eda", exist_ok=True)

# ---- 1. 파이프라인 팩토리 직렬화 -------------------------------------------
factory = PipelineFactory(taxonomy=d["taxonomy"], domain_taxa=d["domain_taxa"], meta=meta)
with open("artifacts/pipeline_factory.pkl", "wb") as f:
    pickle.dump(factory, f)
print("저장: artifacts/pipeline_factory.pkl  (level=%s)" % LEVEL)

# ---- 2. 변환별 EDA: 전체데이터 fit → PCoA(상위3축) → site/season 색칠 -------
sites = meta["site"].values
seasons = meta["season"].values
site_codes = pd.Categorical(sites).codes
season_codes = pd.Categorical(seasons).codes

def transformed_all(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline(key, filter_params={"min_prevalence": 0.25},
                              feature_blocks=["taxa"], ctx=ctx)   # 순수 taxa 변환만
        Z = pipe.fit_transform(X)                                 # ⚠ 전체데이터 fit = EDA 전용
    return Z

rows = []
# 대표 변환들에 대한 PCoA 그림(격자)
show_keys = ["clr", "prop", "hellinger", "alr", "ilr_taxo", "rank"]
fig, axes = plt.subplots(len(show_keys), 2, figsize=(9, 3.0 * len(show_keys)))
for r, key in enumerate(show_keys):
    Z = transformed_all(key)
    ncomp = min(3, min(Z.shape) - 1) if min(Z.shape) > 1 else 1
    coords = PCA(n_components=max(ncomp, 2)).fit_transform(Z - Z.mean(0))
    # 구조 지표: 실루엣(좌표상 site vs season 분리도)
    try:
        sil_site = silhouette_score(coords, site_codes)
    except Exception:
        sil_site = np.nan
    try:
        sil_season = silhouette_score(coords, season_codes)
    except Exception:
        sil_season = np.nan
    rows.append((key, sil_site, sil_season))
    for c, (codes, title, cmap) in enumerate(
            [(site_codes, "site", "tab10"), (season_codes, "season", "viridis")]):
        ax = axes[r, c]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=codes, cmap=cmap, s=28,
                        edgecolor="k", linewidth=0.3)
        ax.set_title(f"{key} · PCoA colored by {title}", fontsize=9)
        ax.set_xlabel("Axis1", fontsize=8); ax.set_ylabel("Axis2", fontsize=8)
        ax.tick_params(labelsize=7)
fig.suptitle("Transform comparison — Aitchison/Euclidean PCoA (EDA, fit-on-ALL, DO NOT MODEL)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig("figures/eda/pcoa_transform_grid_EDA_DO_NOT_MODEL.png", dpi=110)
plt.close(fig)

# ---- 3. 리포트 03 -----------------------------------------------------------
R = ["# 03 · 변환 비교 EDA (Transform comparison)", "",
     "> ⚠ 이 리포트의 좌표/그림은 **전체 데이터로 fit** 한 EDA 전용 산출물입니다. "
     "모델 성능 추정에 사용하면 누수입니다(파일명 `_EDA_DO_NOT_MODEL`).", "",
     f"- 분류계급: **{LEVEL}** (p={X.shape[1]}), n={X.shape[0]}, 독립단위(site)=10", "",
     "## 변환별 PCoA 구조 지표 (실루엣: 클수록 그 라벨로 잘 뭉침)", "",
     "| 변환 | silhouette(site) | silhouette(season) | 우세 구조 |",
     "|---|---|---|---|"]
for key, ss, se in rows:
    dom = "site(반복측정)" if (np.nan_to_num(ss) >= np.nan_to_num(se)) else "season"
    R.append(f"| {key} | {ss:.3f} | {se:.3f} | {dom} |")

mean_site = np.nanmean([r[1] for r in rows])
mean_season = np.nanmean([r[2] for r in rows])
R += ["",
      f"- 평균 실루엣: site={mean_site:.3f}, season={mean_season:.3f}",
      ""]
if mean_site >= mean_season:
    R.append("**해석**: 대부분 변환에서 **site 군집이 season 군집보다 뚜렷**하다. 즉 반복측정"
             "(같은 site×4계절) 구조가 지배적이며, 같은 site 의 다른 계절이 서로 강하게 상관한다. "
             "→ **무작위 K-fold 는 낙관 편향**되고, PROMPT 2 의 **site 그룹 CV(LeaveOneGroupOut)가 "
             "필수**임을 재확인한다.")
else:
    R.append("**해석**: 조성(community) 상에서는 site/season 군집이 모두 약하다(실루엣이 음수). "
             "이 합성 데이터에서 각 시료의 조성은 독립 생성되어 조성 자체에는 강한 site 군집이 "
             "없다. 그러나 **TARGET(`ch4_yield`)에는 설계상 site 수준 반복측정 구조**"
             "(site_intercept, 같은 site 4계절 공유)가 존재한다. 따라서 조성 군집과 무관하게 "
             "**site 그룹 CV(LeaveOneGroupOut)는 필수**다 — 무작위 K-fold 는 같은 site 의 다른 "
             "계절로부터 site 수준 성분을 새어 학습해 낙관 편향된다.")
R += ["",
      "> **결정적 근거는 PROMPT 2 의 '누수 격차'**: 동일 모델을 무작위 K-fold(누수) vs "
      "site 그룹 CV 로 평가해 성능 차이를 직접 보인다. 이 격차가 그룹 CV 필요성의 정량적 증거다.",
      "",
      "- 그림: `figures/eda/pcoa_transform_grid_EDA_DO_NOT_MODEL.png`",
      "- 변환 8종은 어느 것도 여기서 배제하지 않는다. 최종 변환 선택은 PROMPT 2 의 "
      "중첩 CV **내부 루프**가 판정한다(전처리 단계 선택은 선택편향)."]
open("reports/03_transform_comparison_EDA.md", "w").write("\n".join(R))
print("작성: reports/03_transform_comparison_EDA.md")
print(f"평균 실루엣  site={mean_site:.3f}  season={mean_season:.3f}  "
      f"→ 지배구조={'site' if mean_site>=mean_season else 'season'}")
