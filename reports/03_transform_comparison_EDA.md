# 03 · 변환 비교 EDA (Transform comparison)

> ⚠ 이 리포트의 좌표/그림은 **전체 데이터로 fit** 한 EDA 전용 산출물입니다. 모델 성능 추정에 사용하면 누수입니다(파일명 `_EDA_DO_NOT_MODEL`).

- 분류계급: **genus** (p=96), n=40, 독립단위(site)=10

## 변환별 PCoA 구조 지표 (실루엣: 클수록 그 라벨로 잘 뭉침)

| 변환 | silhouette(site) | silhouette(season) | 우세 구조 |
|---|---|---|---|
| clr | -0.207 | -0.051 | season |
| prop | -0.294 | -0.081 | season |
| hellinger | -0.209 | -0.079 | season |
| alr | -0.257 | -0.129 | season |
| ilr_taxo | -0.187 | -0.072 | season |
| rank | -0.239 | -0.073 | season |

- 평균 실루엣: site=-0.232, season=-0.081

**해석**: 조성(community) 상에서는 site/season 군집이 모두 약하다(실루엣이 음수). 이 합성 데이터에서 각 시료의 조성은 독립 생성되어 조성 자체에는 강한 site 군집이 없다. 그러나 **TARGET(`ch4_yield`)에는 설계상 site 수준 반복측정 구조**(site_intercept, 같은 site 4계절 공유)가 존재한다. 따라서 조성 군집과 무관하게 **site 그룹 CV(LeaveOneGroupOut)는 필수**다 — 무작위 K-fold 는 같은 site 의 다른 계절로부터 site 수준 성분을 새어 학습해 낙관 편향된다.

> **결정적 근거는 PROMPT 2 의 '누수 격차'**: 동일 모델을 무작위 K-fold(누수) vs site 그룹 CV 로 평가해 성능 차이를 직접 보인다. 이 격차가 그룹 CV 필요성의 정량적 증거다.

- 그림: `figures/eda/pcoa_transform_grid_EDA_DO_NOT_MODEL.png`
- 변환 8종은 어느 것도 여기서 배제하지 않는다. 최종 변환 선택은 PROMPT 2 의 중첩 CV **내부 루프**가 판정한다(전처리 단계 선택은 선택편향).