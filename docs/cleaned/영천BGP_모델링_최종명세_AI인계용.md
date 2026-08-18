<!-- docs/source/영천BGP_모델링_최종명세_AI인계용.md 에서 기각분을 삭제한 판. 규칙: scripts/clean_source_docs.py · 근거: docs/REJECTED_REGISTRY.md -->

> **정리 알림** — 이 문서에서 R-06 항목이 삭제되었다.

---

---
document_type: modeling_handoff_spec
project: BioGuard-AI / 영천 BGP 혐기소화 모델링
purpose: "다른 AI 에이전트가 이 프로젝트의 모델링을 이어받기 위한 완결 명세"
dataset: 영천BGP_MASTER_2018-2023.csv (2,086행 × 61열)
split: {train: 2018-2021, valid: 2022, test: 2023}
status: "모델 비교 완료. 채택 모델 확정. 미해결 항목 §7에 명시"
key_verdict: "기계론 모델의 성능 이득은 ΔR²<0.01. 해석 가능성을 위해 1차 가수분해 모델 채택"
prior_docs: [영천BGP_AI참조문서.md, 영천BGP_데이터처리기록_및_원본부속정보.md, 영천BGP_2018-2023_데이터취급방법론.md]
---

# 영천 BGP 모델링 최종 명세 (AI 인계용)

이 문서는 데이터 명세가 아니라 **모델링 결과와 결론**입니다. 데이터 자체의 구조·품질은 `영천BGP_AI참조문서.md`를 먼저 읽으십시오.

---

## 0. 최우선 경고 — 이것을 모르면 모든 결과가 틀립니다

```yaml
critical_warnings:
  - id: W1
    title: "persistence R²=0.93은 예측 성능이 아니다"
    fact: "원계열 persistence R²=0.831, 차분계열로 계산하면 -0.000"
    cause: "HRT 40.5일 → 일 교체율 2.47% → ACF(1)=0.916. 반응조 관성이 만든 허위 R²"
    theory_check: "이론 CSTR 감쇠 exp(-1/40.5)=0.976 vs 관측 ACF(1)=0.916 — 정합"
    action: "y_lag1 사용 금지. 사용 시 모델이 공정이 아니라 자기 자신을 설명하게 됨"
    reference: "Granger-Newbold spurious regression (1974)과 동일 구조"

  - id: W2
    title: "소화조 내부 이화학 데이터로 가스량 예측 불가"
    fact: "소화조 pH CV=1.68%, 온도 CV=2.33% — 변동이 없음. 가스는 CV=23.7%"
    evidence: "내부상태 추가 시 test R² 0.688→0.662 (하락)"
    residual_corr: {소화조pH: 0.125, VFA: 0.138, VFA_Alk: 0.042, 알칼리도: 0.190, NH3N: 0.237, 소화액COD: 0.229}
    action: "내부 변수는 예측이 아니라 진단 계층에 배치"

  - id: W3
    title: "무작위 K-fold 금지"
    cause: "ACF(1)=0.916. 인접일이 train/test에 나뉘어 들어가 R² 대폭 부풀림"
    action: "시간순 분할만 사용"

  - id: W4
    title: "df.dropna() 금지"
    cause: "2023 전체 + 모든 연도 주말·실험실 결측일 삭제 → 평일 편향 데이터셋"
    action: "필수 컬럼만 dropna(subset=[...])"
```

---

## 1. 채택 모델

```python
# 최종 채택: 1차 가수분해 모델 (Eastman & Ferguson 형)
# test R² = 0.776, 파라미터 3개, 자기회귀항 없음

import numpy as np, pandas as pd

def hydrolysis_kernel(load_series, k=0.35, K=120):
    """
    1차 가수분해 커널: G(t) = Σ k·exp(-k·τ)·Load(t-τ)
    k     : 겉보기 가수분해 상수 [1/d], 추정값 0.35 (95% CI 0.15-1.50)
    load  : 유기물 부하 [t TS/d] = feed_AB_tpd × acid_TS_pct / 100
    """
    tau = np.arange(K + 1)
    w = k * np.exp(-k * tau)
    w = w / w.sum()
    v = np.nan_to_num(load_series.values)
    out = np.zeros(len(v))
    for i, wv in enumerate(w):
        out[i:] += wv * v[:len(v) - i]
    return pd.Series(out, index=load_series.index)

# 모델식:  biogas(t) = a·H(t) + c
#   H(t) = hydrolysis_kernel(TS_load, k=0.35)
#   a, c 는 학습셋(2018-2021)에서 OLS로 추정
```

```yaml
adopted_model:
  name: "First-order hydrolysis (single pool)"
  formula: "biogas(t) = a·H(t; k) + c"
  parameters: {k_hyd: 0.35, a: OLS추정, c: OLS추정}
  n_parameters: 3
  performance: {valid_R2: 0.550, test_R2: 0.776, test_RMSE: 962}
  reproduction_note: >
    §9 최소 재현 코드(단일 특징, k=0.35 고정)로 실행 시 test R²=0.772.
    위 0.776은 k를 valid 기준으로 최적화(k=0.45)한 값. 둘 다 정상이며
    k=0.35는 부트스트랩 중앙값, k=0.45는 valid 최적값이라는 차이.
  rationale: >
    ADM1-R4 축약(파라미터 6개, test 0.778)과 성능이 동일하나 파라미터가 절반.
    통계 FIR(0.768) 대비 +0.008로 성능 이득은 미미하나,
    k_hyd라는 물리적으로 해석 가능한 파라미터를 산출한다는 점에서 채택.

  optional_extension:
    feature: "COD/VS 비 (기질 환원도)"
    gain: "test 0.775 → 0.784"
    caution: "COD 결측 48%로 학습 표본 1,461→1,224일 감소"
```

---

## 2. 모델 비교 결과 (전수)

동일 데이터·동일 분할·동일 평가로 검증했습니다.

| 모델 | 출처 | 파라미터 | valid R² | **test R²** | RMSE | n_train |
|---|---|---|---|---|---|---|
| M4 ADM1-R4 축약 | Weinrich & Nelles | 6 | 0.632 | **0.778** | 959 | 1,424 |
| **M1 1차 가수분해** | Eastman & Ferguson | **3** | 0.550 | **0.776** | 962 | 1,461 |
| B1 통계 FIR (기준선) | — | 4 | 0.547 | 0.768 | 979 | 1,458 |
| M3 AM2 (2단계) | Bernard et al. 2001 | 4 | 0.512 | 0.733 | 1,051 | 1,461 |
| M2 Chen & Hashimoto | 1978 | 4 | 0.486 | 0.692 | 1,128 | 1,460 |

**핵심 판정: 기계론 모델의 성능 이득은 ΔR² < 0.01. 복잡도 증가가 정당화되지 않습니다.**

### 2.1 AM2가 부진한 이유 (중요)

AM2는 2상 구조에 이론적으로 가장 적합한데도 0.733에 그쳤습니다. 두 가지 원인입니다.

**① 원 용도 불일치** — AM2는 용해성 탄수화물 기반 와인 폐수용으로 개발되어 가수분해가 무관한 경우를 대상으로 합니다. 영천은 고형물 기질이라 가수분해가 지배적입니다.

**② 파라미터 식별 불가** — 생물학적으로 불가능한 조합과 구분되지 않습니다.

| k1 (산생성) | k2 (메탄생성) | valid R² | 생물학적 타당성 |
|---|---|---|---|
| 2.0 | 0.60 | 0.512 | ✓ |
| **0.5** | **1.50** | **0.510** | **✗ 모순** |
| 1.2 | 0.60 | 0.510 | ✓ |
| **0.8** | **1.00** | **0.510** | **✗ 모순** |

R² 차이 0.002로는 `k1 > k2`(산생성균이 메탄생성균보다 빠름)라는 필수 조건을 강제할 수 없습니다. **일별 샘플링으로는 2단계 분해 불가.**

### 2.2 Chen & Hashimoto가 최하위인 이유

HRT 기반 정상상태 수율식인데, 영천은 HRT가 40.5일로 거의 고정되어 변별력이 없습니다. 배치/BMP 예측에는 적합하나 정상상태 연속 운전에는 부적합합니다.

---

## 3. 추정된 물리 파라미터

```yaml
k_hyd:
  point_estimate: 0.35   # /d
  ci_95: [0.15, 1.50]    # 블록 부트스트랩 (30일 블록, 300회)
  half_life_days: 2.0    # CI 0.5 - 4.6
  literature:
    ADM1_food_waste: [0.2, 1.0]
    ADM1_sewage_sludge: [0.1, 0.5]
    Batstone_2009_fullscale: 0.122
    Batstone_2009_post_maintenance: 0.045
  bootstrap_in_literature_range: "75%"
  interpretation: >
    영천 0.35는 Batstone(2009) 실규모 값 0.122의 약 3배.
    2상 구조에 의한 가수분해 촉진 가능성을 시사하나,
    단상 대조군 부재로 인과 확정 불가 (§7 참조).
  caution: "CI가 [0.15, 1.50]으로 넓음. 점추정 0.35를 단정적으로 인용하지 말 것"

impulse_response:
  finding: "유기물 부하 → 가스 응답은 지연 0~3일 구간에 집중"
  evidence_1: "검증 R² 증분이 τ=3에서 꺾임 (τ≤3: +0.002, τ≤4: -0.001, 이후 계속 음수)"
  evidence_2: "EWMA 반감기 그리드 최적 2일, 40일(HRT)에서 R² 0.028로 붕괴"
  evidence_3: "플라시보 테스트 통과 (실제 0.642 vs 셔플30회 최대 -0.239)"
  NOT_proven:
    - "3일 미만 분해 불가 (일별 샘플링 한계)"
    - "h(τ) 개별값 불안정 (K=3일 때 h(0)=239, K=20일 때 375)"
    - "'2상이라서 빠르다'는 인과 미확정 (단상 대조군 없음)"
  scope: >
    Σh(477 m³/tVS)는 관측 총수율(1,155)의 41%.
    나머지 59%는 기저항(c≈3,000-6,000 m³/d)에 흡수되는 난분해성 분획.
    즉 '3일 완결'은 易분해성 분획에 한정된 서술.
```

---

## 4. 3층 구조 (설계 및 검증 결과)

```
[Layer 1] 기질 기반 예측     → 오늘 이만큼 나와야 한다     test R² 0.776
              ↓ 잔차 (σ=1,281 m³/d)
[Layer 2] 내부 이화학 진단   → 왜 그만큼 안 나왔나         보조 지표만
              ↓
[Layer 3] FAN 근본원인       → 암모니아 저해인가          ★검증 실패
```

### Layer 1 — 채택

| 모델 | valid | test |
|---|---|---|
| NNLS-FIR (지연 0~3) | 0.546 | 0.763 |
| OLS 기질 전체 | 0.673 | 0.781 |
| LightGBM | 0.695 | 0.782 |
| **RandomForest** | 0.706 | **0.808** |
| NNLS 스태킹 (FIR+OLS+RF) | — | **0.813** |

스태킹 이득은 0.808 → 0.813으로 미미합니다. **RF 단독으로 충분합니다.**

잔차 상관행렬 (test 2023):

| | FIR | OLS | LGBM | RF |
|---|---|---|---|---|
| FIR | 1.000 | 0.868 | 0.776 | 0.808 |
| LGBM | 0.776 | 0.889 | 1.000 | **0.964** |

LGBM-RF는 0.964로 붙어 있고(둘 다 트리), FIR만 0.78로 갈라집니다. **구조가 다른 모델을 섞어야 앙상블이 의미를 갖습니다.**

### Layer 2 — 축소 채택

Layer 1 잔차(σ=1,281 m³/d)와의 상관:

| 변수 | r | p | 판정 |
|---|---|---|---|
| **NH3N_A_mgL** | 0.237 | 6.1e-04 | 유의 |
| **dig_CODcr_A_mgL** | 0.229 | 2.2e-13 | 유의 |
| ALK_A_mgL | 0.190 | 1.3e-10 | 약함 |
| dig_pH_A | 0.125 | 2.6e-05 | 약함 |
| **VFA_ALK_A** | **0.042** | 0.15 | **무의미** |

r=0.23이면 분산의 5%입니다. 진단 보조는 되지만 예측 변수는 아닙니다.

> **VFA/Alk는 관리 지표에서 제외하십시오.** r=0.042로 완전 무의미하며, 별도로 이 지표는 6년간 0.21→0.16으로 "개선"되는 동안 메탄은 −8.1% 감소했습니다(§6).

### Layer 3 — 검증 실패 (중요)

**당초 제안했던 "FAN > 300 mg/L 조기경보"는 이 데이터로 지지되지 않습니다. 철회합니다.**

| 시간 척도 | FAN vs 성능 | n | p |
|---|---|---|---|
> **[R-06] 삭제됨** — 2개 줄. 연 단위 r=−0.999 (n=4). 반입톤당·투입톤당 어느 정의로도 재현되지 않는다(최대 −0.847).
| 일 단위 (일별 수율) | r = +0.056 | 229 | 0.399 |
| 일 단위 (연도 고정효과 통제) | (+) | 229 | 0.138 |


```yaml
layer3_conclusion:
  finding: "암모니아 저해는 만성(chronic) 현상이며 일별 변동 요인이 아님"
  implication: "FAN 기반 실시간 경보 구현 불가"
  chronic_hypothesis_still_alive:
    - "연 단위 추세: FAN 306→585, 반입톤당메탄 29.8→25.8"
    - "TAN +104% (τ=0.571, p=5e-38), pH 7.85→8.02 (τ=0.261)"
    - "OLR 1.14(여유), HRT 40.5일(충분) → 소거법으로 암모니아 저해 유력"
  verification_requires:
    - "NH3-N 주 1회 이상 연속 측정 (현재 2022년 3건, 2023년 0건)"
    - "16S 군집 데이터로 메탄생성 경로 전환 직접 관찰"
```

---

## 5. 특징(feature) 명세

```yaml
features_include:
  tier1_core:
    - name: TS_load
      formula: "feed_AB_tpd × acid_TS_pct / 100"
      unit: "t TS/d"
      note: "부피(feed)가 아니라 질량 부하. 단독 r=0.54 → 0.61로 상승"
    - name: H_kernel
      formula: "hydrolysis_kernel(TS_load, k=0.35)"
      note: "채택 모델의 유일한 주 특징"

  tier2_optional:
    - name: CODVS
      formula: "acid_CODcr_mgL / (acid_VS_pct × 10000)"
      meaning: "기질 환원도. ADM1의 지질 분획 프록시"
      gain: "test 0.775 → 0.784"
      cost: "COD 결측 48%"

features_exclude:
  - name: y_lag1
    reason: "W1 참조. 자기참조로 공정 설명력 상실"
  - name: acid_VS_pct
    reason: "2020년 VS/TS 계단 불연속(0.80→0.70). TS로 대체"
  - name: acid_pH
    reason: "★이전 추천 철회. valid에선 유용했으나 test에서 0.784→0.574로 붕괴. 연도간 불안정(4.84→5.14 추세)"
  - name: dig_T_A_C / dig_T_B_C
    reason: "단독 R²=0.001, CV<2.6%. 센서 결함 구간 존재"
  - name: intake_manure_tpd / intake_food_tpd
    reason: "r≈0.00"
  - name: intake_total_tpd
    reason: "r=0.138. 저류조에서 완전 평활화(반입 CV 46.3% → 투입 CV 17.3%)"
  - name: lag_12_to_15_features
    reason: "차분 후 소멸하는 위상관"
  - name: 개별 VFA 6종
    reason: "6년간 미측정 (헤더만 존재)"
```

---

## 6. 부수 발견 — 논문 가치가 있는 항목

```yaml
finding_1_VFA_Alk_paradox:
  observation: "VFA/Alk가 개선되는 동안 성능은 저하"
  data:
    2018: {VFA_Alk: 0.21, CH4_m3d: 7080}
    2019: {VFA_Alk: 0.18, CH4_m3d: 7346}
    2020: {VFA_Alk: 0.16, CH4_m3d: 6937}
    2021: {VFA_Alk: 0.17, CH4_m3d: 6568}
    2022: {VFA_Alk: 0.16, CH4_m3d: 6430}
  mechanism: "분모(알칼리도)가 암모니아 때문에 팽창"
  regression: "ALK_A = 14089 + 0.640 × NH3N_A  (r=0.587, R²=0.345, p=1.0e-22, n=230)"
  theoretical_check:
    theoretical_slope: 3.57   # (50/14), TAN 전량 중탄산 가정
    observed_slope: 0.640
    ratio: 0.18
    interpretation: "TAN 증가분의 18%만 알칼리도로 직접 설명. 나머지는 기저 탄산완충계(절편 14,089)"
  significance: "고암모니아 시설에서 VFA/Alk 단독 관리의 구조적 실패를 정량 제시"

finding_2_ADM1_default_params_fail:
  observation: "ADM1 기본 K_I,NH3 = 0.0018M ≈ 31 mg N/L"
  problem: "영천 FAN 평균 410 → I_NH3 = 0.081 (저해율 92%)"
  reality: "공정은 정상 운전 중"
  conclusion: "ADM1 기본 파라미터는 순치된 고암모니아 시설에 부적합"

finding_3_response_time_vs_HRT:
  claim: "가스 응답 시간상수(반감기 2일)는 HRT(40.5일)와 무관"
  caution: "'2상이라서'는 미확정. §7 참조"
```

---

## 7. 미해결 항목 (다음 작업자가 해야 할 것)

```yaml
open_items:
  - id: O1
    priority: HIGHEST
    title: "단상 소화조 대조군 확보"
    why: "'2상이라서 빠르다'는 인과 주장의 유일한 검증 경로"
    how: "KECO 데이터셋(30-40개 국내 AD 시설)에서 단상 시설 선별 → 동일 FIR 적용"
    expected: "단상에서 k_hyd < 0.2 /d, h(τ) 더 길게 늘어짐"
    if_fails: "주장을 '이 시설에서 관측됨'으로 축소"

  - id: O2
    priority: HIGH
    title: "신규성 확인"
    action: "실규모 AD의 FIR/임펄스응답 동정 선행연구 검색"
    keywords: ["impulse response identification anaerobic digester full-scale",
               "distributed lag model biogas", "apparent hydrolysis constant full-scale"]
    note: "선행연구 있어도 '2상 vs 단상 비교'는 남을 가능성 높음"

  - id: O3
    priority: HIGH
    title: "2022년 이상 규명"
    observation: "valid 2022 R²(0.55-0.71) < test 2023 R²(0.76-0.81) — 통상과 반대"
    related: "2022년 Δfeed→Δgas 반응계수 22.51 (6개년 최저, 전체평균 33.16)"
    action: "시설에 2022년 정비·설비변경 이력 문의. Event Story 시트가 공란이라 자체 확인 불가"

  - id: O4
    priority: MEDIUM
    title: "VS/TS 2020년 불연속 원인 확정"
    observation: "VS/TS 0.806(2019) → 0.696(2020) 계단 하락 후 유지"
    caution: "★'측정법 변경'은 추론이며 근거 없음. TS도 6.60→5.81로 함께 하락했으므로 기질 변화 가능성 배제 못함"
    action: "시설에 VS 분석 SOP·회화조건·담당자 변경 이력 문의"
    until_then: "'VS/TS에 2020년 불연속 관측됨, 원인 미상'으로 기술"

  - id: O5
    priority: MEDIUM
    title: "기저항 c의 정체 분해"
    observation: "c ≈ 3,000-6,000 m³/d, 평균 가스의 28-59%"
    hypothesis: "난분해성 분획의 정상상태 기여"
    blocker: "투입을 장기간 크게 바꾼 구간이 없어 검증 불가"
    action: "논문에서는 추정 상수항으로 보고, 기작은 가설로 제시"

  - id: O6
    priority: LOW
    title: "계수 시변성 대응"
    data:
      2018: 34.49  # Δfeed→Δgas 계수, 95% CI ±3.64
      2019: 33.38
      2020: 24.10
      2021: 40.84
      2022: 22.51
      2023: 43.77
    note: "2020 CI 상한(27.70)과 2021 CI 하한(34.58) 미중첩 → 우연 아님"
    options: ["rolling window 재추정(180일)", "d_feed × acid_TS_pct 상호작용항", "주기적 재학습"]
```

---

## 8. 검증 프로토콜 (필수 준수)

```yaml
validation_protocol:
  split:
    train: "2018-01-01 ~ 2021-12-31"
    valid: "2022-01-01 ~ 2022-12-31"
    test:  "2023-01-01 ~ 2023-09-17"
    forbidden: "무작위 K-fold (ACF₁=0.916으로 인접일 누출)"

  mandatory_reporting:
    - "persistence 기준선 R²/RMSE/MAE/MAPE (동일 표본에서)"
    - "차분계열 R² (원계열 R²의 허위성 확인용)"
    - "전체 평균 모델 R² (하한선)"
    - "학습셋 feed_AB_tpd의 p1/p99 (입력공간 커버리지)"
    - "추정 계수와 신뢰구간 (물리 타당성 확인)"

  sanity_checks:
    - "Δfeed→Δgas 계수가 20~45 밖이면 학습셋 구성 의심 (6개년 실측 22.5~43.8)"
    - "학습셋 feed p1 > 130 t/d면 저부하 커버리지 부족"
    - "Σh 기반 메탄수율이 600 m³CH4/tVS 초과면 이론 상한 위반 검토"

  decision_rule:
    rule: "배제/추가로 인한 검증 R² 개선이 0.01 미만이면 채택하지 않는다"
    evidence: "2018 전체 배제 시 R² +0.0011 (365일 손실), 기계론 모델 전환 시 +0.008"

  year_handling:
    2018: {regression: INCLUDE, anomaly_training: EXCLUDE}
    2023: {regression: INCLUDE, annual_aggregation: EXCLUDE, trend_test: EXCLUDE}
    detail: "영천BGP_2018-2023_데이터취급방법론.md 참조"
```

---

## 9. 재현 코드 (최소 실행 단위)

```python
import numpy as np, pandas as pd

M = pd.read_csv('영천BGP_MASTER_2018-2023.csv', parse_dates=['date']).set_index('date')

# 1) 유기물 부하 (질량 기준 — 부피 아님)
TS = M.acid_TS_pct.interpolate(limit=3)
Q  = M.feed_AB_tpd.ffill(limit=2)
TS_load = (Q * TS / 100).ffill(limit=3)          # t TS/d
y = M.biogas_AB_m3d

# 2) 1차 가수분해 커널
def kernel(s, k=0.35, K=120):
    tau = np.arange(K+1); w = k*np.exp(-k*tau); w /= w.sum()
    v = np.nan_to_num(s.values); out = np.zeros(len(v))
    for i, wv in enumerate(w): out[i:] += wv * v[:len(v)-i]
    return pd.Series(out, index=s.index)

H = kernel(TS_load, k=0.35)

# 3) 시간순 분할 + OLS
D = pd.concat([H.rename('H'), y.rename('y')], axis=1).dropna()
tr = D[D.index.year <= 2021]; te = D[D.index.year == 2023]
A = np.c_[np.ones(len(tr)), tr[['H']].values]
b = np.linalg.lstsq(A, tr['y'].values, rcond=None)[0]
P = np.c_[np.ones(len(te)), te[['H']].values] @ b
t = te['y'].values
print('test R² =', 1 - ((t-P)**2).sum()/((t-t.mean())**2).sum())   # 실행 확인값 0.772

# 4) 필수 기준선 병기
yl = y.shift(1).reindex(te.index); m = yl.notna()
print('persistence R² =', 1 - ((t[m]-yl[m])**2).sum()/((t[m]-t[m].mean())**2).sum())
print('차분 R² =', 1 - ((y.diff()-0)**2).sum()/((y.diff()-y.diff().mean())**2).sum())
```

---

## 10. 이 프로젝트의 서사 (논문 방향)

```yaml
recommended_framing:
  primary_claim: >
    실규모 2상 혐기소화조의 가스 응답 시간상수는 HRT와 무관하다.
    HRT 40.5일임에도 유기물 부하 변동에 대한 응답은 반감기 2일 규모로 완결된다.

  supporting_negative_result: >
    ADM1 축약, AM2, Chen-Hashimoto, 1차 가수분해를 동일 조건 비교한 결과
    모델 복잡도 증가가 예측 성능 개선으로 이어지지 않았다(ΔR²<0.01).
    AM2의 2단계 파라미터는 식별 불가능했으며(생물학적 모순 조합과 R² 차이 0.002),
    이는 일별 샘플링의 근본 한계를 시사한다.

  secondary_finding: >
    통상 안정성 지표(VFA/Alk)가 개선되는 동안 실제 성능은 저하되었다.
    원인은 암모니아 축적에 의한 알칼리도 팽창이며,
    이는 VFA/Alk가 구조적으로 탐지할 수 없는 실패 모드다.

  bridge_to_microbiome: >
    운영 데이터는 만성 성능 저하를 보여주지만 기작을 규명하지 못한다.
    일별 해상도에서 FAN은 성능과 무관하며(p=0.399),
    암모니아 저해가 급성이 아닌 만성 현상임을 시사한다.
    따라서 미생물 군집 수준의 검증이 필요하다.

  do_not_frame_as:
    - "바이오가스 예측 정확도 향상 연구 (persistence가 R²=0.93을 이미 가져감)"
    - "딥러닝 적용 연구 (n=2,086으로 LSTM/Transformer 불가)"
    - "ADM1 적용 연구 (26개 상태변수 중 8개 그룹만 관측)"
```

---

## 부기 — 이 문서에서 철회·정정된 이전 판단

| 이전 주장 | 정정 |
|---|---|
| "FAN>300 조기경보(L1)" | **철회.** 일 단위 p=0.399, 부호 반대 |
| "산발효조 pH를 가수분해 대리지표로 사용" | **철회.** test에서 0.784→0.574 붕괴 |
| "가스 응답 96% 완결" | **정정.** 관측 변동의 96%이며, 유기물 전체로는 41% |
| "h(0)=60.9%, h(3)=0" | **정정.** K에 민감(K=20이면 h(0)=375). 개별값 신뢰 불가 |
| "2020년 VS 측정법 변경" | **정정.** 추론이며 근거 없음. TS도 함께 하락해 기질 변화 가능성 배제 못함 |
| "k_hyd = 0.5 /d" | **정정.** 부트스트랩 중앙값 0.35, 95% CI [0.15, 1.50] |

정정 이력을 남기는 이유는 이 문서가 **판단 근거까지 포함해야 재검증 가능**하기 때문입니다. 위 항목들은 초기 분석에서 표면적 관찰로 내린 판정이었고, 추가 검증으로 확정 또는 철회했습니다.
