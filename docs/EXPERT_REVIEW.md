# 영천 BGP 모델링 설계 전문가 검토서

> `docs/MODELING_OVERVIEW.md` (v2)에 대한 독립 검토. 국내외 문헌 중 **영천과 공정이 유사한
> 사례**(중온·습식·음식물류/가축분뇨 병합·고암모니아·전규모)를 우선 채택해 대조했다.
>
> 검토 원칙: 문헌 주장을 그대로 옮기지 않고 **영천 데이터로 재계산해 검증**했다.
> 검증 코드는 본 문서의 각 절에 수치와 함께 명시한다.
>
> **결론 요약: v2의 진단(암모니아 저해)은 옳다. 그러나 v2가 제시한 관제 임계와 예측 지평은
> 둘 다 작동하지 않는다.** 아래 A1·A2가 반드시 수정되어야 한다.

---

## 0. 검토 결과 요약

| # | 구분 | 내용 | 심각도 |
|---|------|------|:---:|
| **A1** | 설계 결함 | **FAN>300 경보가 전체 일수의 75.1%에서 발화** — 경보로 기능 불가. 비순치 문헌 임계를 순치된 시설에 적용한 내부 모순 | 🔴 치명 |
| **A2** | 설계 결함 | **예측 지평 오설정.** h=1일은 persistence가 R²=0.932로 지배. ML 여지는 h≥7일에만 존재(h=7 R²=0.566 → h=30 R²=−0.564) | 🔴 치명 |
| **A3** | 설계 누락 | VS 기반 수율을 폐기했으나 **대체 정규화 지표를 정의하지 않음** | 🟠 중대 |
| **B1** | 신규 확립 | **COD 기준 수율 Y_COD = 0.298 m³CH₄/kg COD제거 = 이론치의 85%** — v1/v2의 "수율이 Buswell 상한 초과" 미해결 항목을 종결 | ✅ 해결 |
| **B2** | 신규 확립 | **Y_COD와 FAN의 일별 역상관 ρ=−0.410 (p=2.7e−08)** — 암모니아 저해를 일 단위로 정량화 | ✅ 강화 |
| **B3** | 신규 도구 | **열역학 상한(0.35)을 QC 필터로 사용** → 물리적으로 불가능한 관측 14.2% 검출. 참조문서 DQ-01~12에 없는 검증 규칙 | ✅ 추가 |
| **B4** | 기전 정밀화 | FAN 410~430(2021년 595)은 문헌상 **아세트산분해 → SAO 경로 전환 구간(200~500)** 에 정확히 위치 | ✅ 강화 |
| **C1** | 음성 결과 | **TAN 소프트센서 실패** (시간순 검증 R²=−0.80~−1.69). NH₃-N 측정 복원의 *대안이 아님* | ⚠️ 기각 |
| **C2** | 부분 성공 | ALK 역산 FAN 프록시: 절대값 불가(MAE 1,002), **추세 지표로는 유효**(표본 5.4배) | 🟡 조건부 |
| **D1** | 방법론 정정 | 참조문서 추세 p값이 자기상관 미보정으로 과대. 보정 후에도 **결론은 생존** | 🟡 경미 |
| **D2** | 변수 재분류 | `소화조_MAlk`는 독립 측정이 아니라 **중탄산알칼리도 계산열**(= TAlk − 0.71×VFA, σ=0) | 🟡 경미 |

---

## A. v2 설계의 치명적 결함

### A1. FAN 절대임계 경보는 작동하지 않는다 (v2 §5 L1 폐기)

v2는 관제 최상위 규칙으로 `FAN > 300 mg/L`를 제시했다. 영천 데이터로 발화율을 계산했다.

```
n=229 (NH₃-N 측정일)
FAN > 150 mg/L  →  발화율 98.7%
FAN > 300 mg/L  →  발화율 75.1%
```

**3일 중 2~3일 울리는 경보는 경보가 아니다.** 근본 원인은 임계값의 출처다. 150/300/600
밴드는 **비순치(unacclimated) 접종원** 기준이며, 문헌은 이 임계의 편차가 극단적으로 크다는
점을 명시한다:

- 보고된 저해 개시 농도가 **27 ~ 1,450 mg NH₃/L** 범위로 분산 ([Capson-Tojo et al.,
  *Unraveling the literature chaos around free ammonia inhibition in anaerobic
  digestion*](https://www.researchgate.net/publication/336617877_Unraveling_the_literature_chaos_around_free_ammonia_inhibition_in_anaerobic_digestion)).
- **비순치계는 TAN 1,700~1,800 mg/L에서 완전 저해**되지만, **순치계는 TAN 5,000 mg/L 초과도
  내성** ([*Ammonia inhibition and toxicity in anaerobic digestion: A critical
  review*](https://www.sciencedirect.com/science/article/abs/pii/S2214714419302107),
  Cranfield).
- 저해 수준은 온도·pH에 의해 지배적으로 좌우되며, 활성 군집과 순치 군집 간 차이가
  보고 편차의 주요 원인 ([Capson-Tojo et al., *Unifying ammonia inhibitory limits in
  anaerobic digestion*](https://www.researchgate.net/profile/Gabriel-Capson-Tojo/publication/333908267_Unifying_ammonia_inhibitory_limits_in_anaerobic_digestion_link_with_operational_conditions_and_microbial_communities/links/5d0c1f5c299bf1547c7157b1/Unifying-ammonia-inhibitory-limits-in-anaerobic-digestion-link-with-operational-conditions-and-microbial-communities.pdf)).

영천은 TAN 3,832~4,913 mg/L에서 6년간 붕괴 없이 운전됐다 — **정의상 순치된 계**다.
v2는 "순치로 붕괴 미발생을 설명"하면서 동시에 "비순치 임계로 경보"하는 내부 모순을 범했다.

#### 대안: 시설 자체 기준선 대비 상대편차 경보

절대임계를 버리고 **시설의 순치된 기준선 대비 이탈**로 전환한다. 성능 지표는 §B1의
`Y_COD`를 쓴다(부하 변동에 오염되지 않음).

| 규칙 | 발화 일수 | 발화율 | 판정 |
|------|:---:|:---:|------|
| Y_COD < 90일 기준선 −10% | 149 | 17.1% | 과민 |
| **Y_COD < 90일 기준선 −15%** | **81** | **9.3%** | **권고** |
| Y_COD < 90일 기준선 −20% | 44 | 5.0% | 보수적 |
| (대조) FAN > 300 절대임계 | — | **75.1%** | 사용 불가 |

FAN은 **경보 임계가 아니라 원인 진단 변수**로 재배치한다: 상대편차 경보가 울렸을 때
"FAN이 동반 상승했는가"를 확인하는 2차 판별에 쓴다.

### A2. 예측 지평이 잘못 설정되었다 (v2 §4.2 수정)

v2는 "Δy(t) = y(t) − y(t−1) 예측"으로 전환했다. 그러나 **h=1일은 persistence가 이길 수 없는
구간**이고, 차분은 백색잡음에 근접(diff ACF(1)=−0.059)하므로 예측 가능성 자체가 낮다.
지평별로 persistence 성능을 계산했다(검증구간 2022~2023, n=625).

| 예측 지평 | persistence R² | RMSE (㎥/d) | ML 여지 |
|:---:|:---:|:---:|------|
| h = 1일 | **0.932** | 460 | 거의 없음 |
| h = 3일 | 0.800 | 789 | 제한적 |
| **h = 7일** | **0.566** | 1,161 | **실질적** |
| **h = 14일** | **0.252** | 1,524 | **큼** |
| h = 30일 | **−0.564** | 2,204 | persistence 무효 |

**h ≥ 7일에서 persistence가 무너진다.** 그리고 이 구간이 운영상 실제로 필요한 지평이다 —
가스홀더 운용, 열병합 정비 계획, 기질 수급 계약은 모두 주 단위 의사결정이다. 1일 예측은
정확해도 쓸 데가 없다.

이는 문헌 관행과도 부합한다. 전규모 AD 대상 ML 연구들이 보고하는 R² 0.60~0.92
([Chemosphere 2023, 산업규모 AD의 ML 예측](https://www.sciencedirect.com/science/article/abs/pii/S0045653523012432);
[*ACS ES&T Engineering* 2024, 도시 병합소화 예측 — MLP R²=0.78, MAPE 13.4%](https://pubs.acs.org/doi/10.1021/acsestengg.3c00435))는
**대체로 persistence 베이스라인을 병기하지 않는다.** 자기상관이 강한 계열에서 절대량 R² 0.8은
persistence 미달일 수 있으므로, 본 프로젝트는 **지평별 skill score**를 필수 보고 항목으로 둔다.

```
Skill(h) = 1 − MSE_model(h) / MSE_persistence(h)      # >0 이어야 의미 있음
```

### A3. 정규화 성능 지표의 공백 → §B1로 해결

v2는 `in_VS`를 DQ-08(측정법 계단변화)로 폐기하고 `in_TS` 사용을 지시했으나, **"성능"을 무엇으로
정규화해 볼 것인지**를 정의하지 않았다. 절대 메탄량은 부하 변동과 섞이므로 저해 진행을
가릴 수 있다(2018→2022 메탄 −8.1%인데 부하도 함께 감소).

---

## B. 새로 확립한 정량 기반

### B1. COD 기준 메탄수율 — v1부터 미해결이던 수율 문제의 종결

v1은 "MY 중위 0.74 m³CH₄/kg VS가 Buswell 이론(탄수화물 0.37/지질 1.01)에 비해 높다"를
**단위 확인 필요 항목으로 남겼다.** v2도 이를 승계했다. 이번에 COD 기준으로 다시 계산해
원인을 규명했다.

COD 물질수지 (n=1,072):

```
COD 부하   = feed_AB × acid_CODcr        = 28,747 kg COD/d (중위)
COD 제거   = feed_AB × (acid − dig) COD  = 23,237 kg/d  (제거율 81.1%)
CH₄ 생산   = 6,879 ㎥/d (중위)

Y_COD = 6,879 / 23,237 = 0.298 m³ CH₄ / kg COD 제거
```

이론 최대는 **0.35 m³ CH₄/kg COD**(COD 4 g ↔ CH₄ 1 g, 표준상태 환산;
[이론 근거](https://www.valorgas.soton.ac.uk/Pub_docs/JyU%20SS%202013/VALORGAS_JyU_2013_Lecture%202.pdf)).

**Y_COD / 0.35 = 85%.** 문헌은 제거 COD의 **10~17%가 균체 합성에 소비**되어 실제 수율이
이론치의 83~90%가 된다고 본다 —
[COD-CH₄ 수율 이론값 정리](https://www.scribd.com/document/267605068/Theoritical-Values-of-Biogas-COD-CH4-Yield-0-35L-CH4-g-COD-1).
**영천의 85%는 이 창 안에 정확히 들어온다.**

> **결론: 수율 이상은 계측 단위 문제가 아니라 VS 측정 문제였다.** DQ-08(VS/TS 0.80→0.70
> 계단 하락)이 VS를 과소평가해 VS 기준 수율을 부풀린 것이다. COD 기준으로 보면 이 시설의
> 전환 효율은 완전히 정상이며 물질수지가 닫힌다.
>
> → **정규화 성능 KPI를 `Y_COD`로 확정한다.** VS 기반 지표(`gas_per_VS`, `CH4_per_VS`)는
> 절대값 보고에서 제외한다.

### B2. Y_COD가 암모니아 저해를 일 단위로 드러낸다

절대 메탄량으로는 부하 변동에 가려 보이지 않던 관계가 정규화하면 드러난다.

| 대조 | Spearman ρ | p | n |
|------|:---:|:---:|:---:|
| **Y_COD vs FAN_A** | **−0.410** | 2.7e−08 | 170 |
| (참고) ALK 역산 FAN 프록시 vs Y_COD | −0.205 | 4.5e−10 | 912 |

연 단위 대조에서 대응이 더 뚜렷하다:

| 연도 | FAN 중위 (mg/L) | Y_COD 중위 | 비고 |
|:---:|:---:|:---:|------|
| 2018 | 317 | 0.281 | |
| 2019 | 374 | 0.310 | |
| 2020 | 347 | 0.304 | |
| **2021** | **595 (최고)** | **0.250 (최저)** | **FAN 정점 ↔ 수율 최저** |
| 2022 | 419 | 0.287 | |
| 2023 | (측정 없음) | 0.297 | |

**2021년 FAN 정점과 Y_COD 최저가 일치한다.** 이는 v2가 연 추세(τ, p값)에만 의존해 제시한
인과 사슬을 **일별 관측 수준의 정량 근거로 격상**시킨다.

### B3. 열역학 상한을 데이터 검증 규칙으로 사용

`Y_COD > 0.35` 또는 `≤ 0`은 **물리적으로 불가능**하다. 영천 데이터에서:

```
물리 불가능 관측: 152 / 1,072 = 14.2%
```

참조문서의 DQ 목록(DQ-01~12)은 범위 필터·센서 결함·수식 오염을 다루지만
**열역학적 일관성 검사는 없다.** 이 규칙은 비용이 0이고, COD·유량·가스가 동시에 틀린 날을
한 번에 잡아낸다. 전처리 파이프라인에 추가할 것을 권고한다(v2 §3 ④ 단계에 삽입).

### B4. 기전 정밀화 — 아세트산분해에서 SAO로의 경로 전환

v2는 "메탄생성 단계 저해"까지만 특정했다. 문헌은 고암모니아 하에서 **무슨 경로로 바뀌는지**를
정량 임계와 함께 제시하며, 영천의 FAN 수준이 정확히 그 전환 구간에 있다.

[FEMS Microbiology Ecology (2015), *Ammonia effect on hydrogenotrophic methanogens and
syntrophic acetate-oxidizing bacteria*](https://academic.oup.com/femsec/article/91/11/fiv130/2467483)
및 관련 연구가 보고하는 구간:

| 암모니아 수준 (mg-N/L) | 우점 경로 | 영천 대응 |
|---|---|---|
| < 200 | *Methanosaeta* 아세트산분해 활성 | — |
| **200 ~ 500** | **SAO-수소영양 경로가 아세트산분해와 경쟁** | **평시 FAN 410~430 = 이 구간** |
| **> 500** | **SAO-HM이 아세트산분해균을 완전히 압도** | **2021년 FAN 595 = 이 구간 진입** |

- 아세트산분해 메탄생성균이 수소영양균보다 암모니아에 민감하므로, 저해 시
  **SAO + 수소영양 메탄생성이 아세트산분해 경로를 대체**한다.
- *Methanoculleus*가 SAO의 주 수소 이용 파트너로 우점하고, *Methanothrix*(구 *Methanosaeta*)·
  *Methanosarcina*가 감소한다.
- 고 TAN 조건에서 수소영양 경로 비중이 **68~75%**(저 TAN 9~23%)에 달한다.
- 전규모 중온 음식물류 소화조에서 SAO 컨소시엄이 실제로 부화됨이 확인됨
  ([*mSystems* 2022](https://journals.asm.org/doi/10.1128/msystems.00339-22)).

**영천과 동일 공정의 국내 사례가 이를 직접 뒷받침한다.** 한국 내 전규모 음식물류 소화조
2기를 분석한 연구에서, **고암모니아 시설은 *Clostridia* 강이 우점하고 메탄생성이 대부분
수소영양균에 의해 수행**되었다
([*Journal of Hazardous Materials* 2020, 한국 전규모 음식물류 AD 2기의 미생물 군집·운전인자 분석](https://www.sciencedirect.com/science/article/abs/pii/S030438942030964X)).
한국에는 약 90기의 전규모 음식물류 AD 시설이 있어 비교군이 확보된다.

> **모델링 함의**: 이는 v2 §6.1의 16S 검증 항목을 **구체적 반증 가능 가설**로 바꾼다.
> 예측: (1) *Methanothrix/Methanosarcina* 상대풍부도 감소, (2) *Methanoculleus* 등
> 수소영양균 우점, (3) SAOB(*Syntrophaceticus* 등) 검출, (4) 2021년 시료가 있다면 가장 극단.
> 이 예측이 틀리면 암모니아 저해 가설은 기각된다.

가축분뇨 혼합비 자체가 설계 변수라는 점도 국내 문헌에 근거가 있다 —
음식물류/가축분뇨 혼합비를 조정해 암모니아·OLR 저해를 회피하는 최적비 연구
([*Sustainability* 2024, 16(17), 7653](https://www.mdpi.com/2071-1050/16/17/7653)),
돼지분뇨·음식물·분뇨·농축슬러지 4종 병합 국내 전규모 사례
([Korea University](https://pure.korea.ac.kr/en/publications/application-of-a-full-scale-horizontal-anaerobic-digester-for-the/)).
**가축분뇨 비중 33.8%는 조정 가능한 운전 레버**이며, 이것이 이 시설에서 유일하게 남은
성능 개선 수단일 가능성이 높다(부하·체류시간·완충능은 모두 여유).

---

## C. 정직한 음성 결과

### C1. TAN 소프트센서는 실패한다 — NH₃-N 측정의 대안이 아니다

문헌은 소프트센서를 적극 권한다. 전규모 건식 음식물류 AD 4기에서 CatBoost로 VFA/ALK
소프트센서를 구축해 R² 0.618~0.768을 얻었고, **FAN과 COD가 VFA/ALK 지표에 미치는 영향이
합쳐 약 50%**로 보고됐다
([*Journal of Environmental Management* 2024](https://www.sciencedirect.com/science/article/abs/pii/S0301479724031761)).
pH 신호만으로 PLS로 TAN을 추정한 사례도 있다
([*Journal of Water Process Engineering* 2022](https://www.sciencedirect.com/science/article/pii/S2214714422001799)).

그래서 영천 데이터로 직접 시험했다. 목표 `NH3N_A`, 입력은 상시 측정 변수 12개
(`dig_pH_A, ALK_A, VFA_A, dig_COD, acid_pH, acid_CODcr, acid_TS, intake_manure,
intake_foodww, feed_AB, CH4_pct, dig_TS_A`), 시간순 분할(학습 2018~2020 n=157 /
검증 2021~2022 n=57).

| 모델 | R² | MAE (mg/L) |
|------|:---:|:---:|
| Ridge | **−0.803** | 908 |
| RandomForest | **−1.686** | 1,123 |
| (기준선) 학습기간 평균 | −3.659 | — |

**실패다.** 두 모델 모두 평균 기준선보다는 낫지만(신호는 존재) 검증구간에서 음의 R²다.
원인은 명확하다: TAN이 3,337 → 4,913 mg/L로 **단조 상승**했으므로 2018~2020으로 학습한
모델은 2021~2022 수준을 **외삽**해야 한다. 게다가 총 n=214, 2022년 3건, 2023년 0건으로
드리프트를 추적할 표본 자체가 없다.

> **결론: 소프트센서는 NH₃-N 측정 복원의 대안이 아니라 그것에 종속된 후속 과제다.**
> 측정이 재개되어 학습 범위 내 보간(interpolation) 문제가 되면 재시도할 가치가 있다.
> v2 §6.1의 "NH₃-N 측정 재개 = 최상 우선순위" 판단은 이 실험으로 **강화**된다 —
> 데이터 과학으로 우회할 수 없음이 실증되었다.

### C2. ALK 역산 프록시 — 절대값은 불가, 추세는 유효

참조문서의 `ALK_A = 14,089 + 0.640 × NH3_A` (r=0.587, R²=0.345)를 역산해 TAN 프록시를
만들어 평가했다.

| 용도 | 결과 | 판정 |
|------|------|------|
| TAN 절대값 추정 | r=0.587, **MAE 1,002 mg/L** | ❌ 사용 불가 |
| Y_COD와의 관계 재현 | ρ=−0.205 (p=4.5e−10), **n=912 (실측 대비 5.4배)** | ✅ 추세 지표로 유효 |

알칼리도는 일별 측정(n=1,257)이므로, NH₃-N 측정이 재개되기 전까지의 **잠정 추세 감시
지표**로 쓸 수 있다. 단 "TAN 값"으로 보고해서는 안 되고 "암모니아 부하 추세 지수"로만
표기해야 한다.

---

## D. 방법론 정정

### D1. 추세 검정의 자기상관 미보정 (결론은 생존)

참조문서는 Kendall τ와 p값을 제시한다(예: ALK_A τ=0.622, **p=4.0e−211**). 그러나
가스 계열 ACF(1)=0.916, 랩 변수도 강한 자기상관을 가지므로 **표준 Mann-Kendall의 독립성
가정이 위반**된다. 양의 자기상관은 없는 추세를 있다고 판정할 확률을 높인다
([Hamed & Rao (1998), *A modified Mann-Kendall trend test for autocorrelated data*,
*Journal of Hydrology* 204:182-196](https://www.sciencedirect.com/science/article/abs/pii/S002216949700125X)).

`pymannkendall`로 Hamed-Rao 보정을 적용했다.

| 변수 | n | τ | 원 MK p | **보정 p** |
|------|:---:|:---:|:---:|:---:|
| ALK_A | 1,257 | 0.625 | ~0 | **1.3e−08** |
| NH3N_A | 230 | 0.571 | ~0 | **5.7e−05** |
| dig_pH_A | 1,259 | 0.280 | ~0 | **4.1e−11** |

> **모든 추세가 보정 후에도 강하게 유의하다 — 실질 결론은 전부 유지된다.**
> 다만 보고된 p=4.0e−211 같은 값은 신뢰할 수 없으므로(약 200자리 과대) 문서에서
> 보정값으로 교체할 것을 권고한다. 이는 암모니아 진단을 약화시키지 않고, 오히려
> 자기상관을 감안해도 살아남는다는 점에서 근거를 견고하게 만든다.

### D2. `소화조_MAlk`는 독립 측정이 아니다 (v1 판단의 근거 교체)

v1은 `소화조_MAlk`를 "TAlk와 corr 0.996으로 중복"이라며 제거했다. 결론은 맞지만 근거가
틀렸다. 실제 관계를 규명했다.

```
(TAlk − MAlk) / VFA  =  0.7100   (중위 = 평균, 표준편차 = 0.000000)
MAlk vs (TAlk − 0.71 × VFA) : 최대 절대오차 0.00 mg/L, corr = 1.000000
```

**`MAlk` = 중탄산(bicarbonate) 알칼리도 계산열**이며, 계수 0.71은 표준식
(중탄산알칼리도 = 총알칼리도 − 0.85 × 0.833 × VFA, 0.85×0.833 = 0.708)과 일치한다.
즉 독립 적정이 아니라 수식열이다. 상관이 아니라 **항등식**이므로 제거가 필수다.

#### 파생 권고: Ripley IA/PA 도입 (신규 측정 항목)

이 확인에는 중요한 부수 결과가 있다. **Ripley의 IA/PA 비는 현 데이터로 산출 불가**하다 —
그것은 pH 5.75와 4.3 두 종점의 실제 2점 적정을 요구한다.

[Ripley et al. (1986), *Improved alkalimetric monitoring for anaerobic digestion of
high-strength waste*](https://www.osti.gov/biblio/5261202)는 가금분뇨 소화조에서
pH 5.75(부분 알칼리도 PA ≈ 중탄산)와 4.3(총) 종점을 사용해, **중간 알칼리도(IA, 5.75→4.3)가
VFA와 강하게 상관**하고 **IA:PA 비가 알칼리도 시험의 민감도를 크게 높여** 공정 이상과 회복을
저렴하게 탐지함을 보였다. 중온 가금분뇨·음식물류에서 **IA/PA < 0.3**이 안정 기준으로 제안된다.
음식물류를 포함한 도시 유기성 폐기물 소화조에서도 알칼리도 비가 불균형 식별에 유효함이
확인됐다
([*Biochemical Engineering Journal* 2013](https://www.sciencedirect.com/science/article/abs/pii/S1369703X13000879)).

> **왜 이것이 영천에 결정적인가**: PA는 중탄산을 근사한다. 영천의 문제는 **암모니아가
> NH₄HCO₃로 중탄산 알칼리도를 부풀려 VFA/총알칼리도 비를 무력화**시킨 것이다(v2 §1.5).
> IA/PA는 분모를 총알칼리도가 아닌 **PA**로 두므로, 원리적으로 이 교란에 더 강건할 수 있다.
> 2점 적정은 추가 시약이 거의 필요 없는 저비용 변경이다.
>
> **단, 이는 검증 대상 가설이다.** PA 자체도 중탄산이므로 암모니아 영향을 받는다.
> 도입 후 IA/PA가 Y_COD와 어떤 관계를 갖는지 6개월 병행 측정으로 확인해야 한다.

---

## E. v3 개정 권고 목록

`docs/MODELING_OVERVIEW.md`에 반영할 변경 사항이다.

| 위치 | 변경 |
|------|------|
| §4.2 타깃 | Δ(h=1) → **다지평 예측 (h = 1/3/7/14일)**, 지평별 skill score 필수 보고. 주 타깃은 **h=7·14일** |
| §4.4 검증 | 단일 분할 → **rolling-origin(확장창) 다중 폴드**. 지평별 persistence 병기 |
| §5 L1 | `FAN>300` 절대임계 **삭제** → **Y_COD 90일 기준선 대비 −15% 상대편차**. FAN은 2차 원인 판별로 재배치 |
| §3 전처리 | ④ 단계에 **열역학 일관성 필터**(0 < Y_COD ≤ 0.35) 추가 |
| §1.4 / KPI | **정규화 성능 KPI = Y_COD** 확정. VS 기반 수율 절대값 보고 금지 |
| §1.4 기전 | SAO 경로 전환(FAN 200~500 경쟁 / >500 SAO 우점) 명시. 16S 예측을 반증 가능 형태로 |
| §1.4 단위 검증 | **"MY 단위 확인 필요" 항목 종결** — VS 측정 문제로 규명됨 (§B1) |
| §6.1 | NH₃-N 복원 근거에 **소프트센서 실패 실증** 추가. **IA/PA 2점 적정 도입** 신규 항목 |
| §2.3 | `MAlk` 제거 근거를 "상관 중복" → **"항등식(TAlk−0.71×VFA)"** 으로 교체 |
| 전반 | 참조문서 인용 p값을 **Hamed-Rao 보정값**으로 교체 |
| §6 신규 | **가축분뇨 혼합비 조정**을 성능 개선 레버로 검토 항목 추가(부하·HRT·완충능 여유 상태) |

### 장기: ADM1 하이브리드의 우선순위 상향

v2는 ADM1을 "장기(7순위)"로 두었다. **상향을 권고한다.**

- v1 시점에는 소화조 용적을 몰라 ADM1 구성이 불가능했다. 이제 **8,000㎥, HRT 40.5일,
  OLR 1.14, COD 물질수지 85% 정합**이 모두 확보되어 **구성 가능한 상태**다.
- ADM1은 **유리암모니아 저해를 명시적으로 모델링**하며 pH·온도 특성을 상세히 기술한다
  ([ADM1 원 논문](https://www.researchgate.net/publication/11198259_Anaerobic_digestion_model_No_1_ADM1);
  [ADM1 수정·응용 종합 리뷰, *Water Research* 2023](https://www.sciencedirect.com/science/article/abs/pii/S0043135423009442)).
  이 시설의 지배적 실패 모드가 바로 그것이다.
- 병합소화 대상 ADM1 보정 선례가 있다
  ([*Chemical Engineering Journal* 2015, 하수슬러지+그리스트랩 폐기물 병합소화 ADM1 보정](https://www.sciencedirect.com/science/article/abs/pii/S1385894714017033)).
- ADM1+ML 하이브리드가 성립한다: ML로 ADM1 동역학 파라미터를 예측하는 M−ADM1이
  파라미터 예측 R²=0.92 달성
  ([*Chemical Engineering Journal* 2023](https://www.sciencedirect.com/science/article/abs/pii/S1385894722058491)),
  ADM1과 ML 결합으로 연속 AD 공정 모델링·최적화
  ([*Biomass and Bioenergy* 2024](https://www.sciencedirect.com/science/article/abs/pii/S0961953424001296)).
- **h≥7일 지평에서 순수 ML의 한계를 메울 유일한 경로**이기도 하다 — 장기 예측에는
  기전 제약이 필요하다.

### 남은 유보 사항

- **2상 공정의 실질 확인 불가**: 유기산화조 유효용적이 공정도·워크북 모두 미기재
  (계통설명서 §8). 진정한 상분리는 산발효조 HRT 1~3일을 요구하므로, pH 분리(4.85 vs 7.95)만으로
  "설계대로 작동"을 단정하기보다 **용적 확보 후 HRT 검증**이 필요하다.
- **인과 추론의 한계**: 가축분뇨↑ → TAN↑ → FAN↑ → 성능↓ 사슬은 모두 시간과 공변한다.
  Y_COD로 부하 교란을 제거한 것(§B2)이 진전이지만, 결정적 검증은 16S 군집분석(§B4) 또는
  혼합비 개입 실험이다.

---

## 참고문헌

**암모니아 저해 · 임계**
1. Capson-Tojo et al. *Unraveling the literature chaos around free ammonia inhibition in anaerobic digestion.* [링크](https://www.researchgate.net/publication/336617877_Unraveling_the_literature_chaos_around_free_ammonia_inhibition_in_anaerobic_digestion)
2. Capson-Tojo et al. *Unifying ammonia inhibitory limits in anaerobic digestion: link with operational conditions and microbial communities.* [PDF](https://www.researchgate.net/profile/Gabriel-Capson-Tojo/publication/333908267_Unifying_ammonia_inhibitory_limits_in_anaerobic_digestion_link_with_operational_conditions_and_microbial_communities/links/5d0c1f5c299bf1547c7157b1/Unifying-ammonia-inhibitory-limits-in-anaerobic-digestion-link-with-operational-conditions-and-microbial-communities.pdf)
3. *Ammonia inhibition and toxicity in anaerobic digestion: A critical review.* [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2214714419302107) / [Cranfield 전문](https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams/059ff1dc-5ae5-4463-9508-9e69b0b468f1/content)

**SAO 경로 전환 · 미생물 군집 (영천 유사 공정)**
4. *Ammonia effect on hydrogenotrophic methanogens and syntrophic acetate-oxidizing bacteria.* FEMS Microbiol Ecol 91(11):fiv130, 2015. [링크](https://academic.oup.com/femsec/article/91/11/fiv130/2467483)
5. *Syntrophic Acetate-Oxidizing Microbial Consortia Enriched from Full-Scale Mesophilic Food Waste Anaerobic Digesters.* mSystems, 2022. [링크](https://journals.asm.org/doi/10.1128/msystems.00339-22)
6. **한국 전규모 음식물류 AD 2기 — 계절변동 및 암모니아 영향 (군집 분석).** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S030438942030964X)

**국내 음식물류·가축분뇨 병합소화 (동일 기질 조합)**
7. *Comparison of Anaerobic Co-Digestion of Food Waste and Livestock Manure at Various Mixing Ratios under Mesophilic and Thermophilic Temperatures.* Sustainability 16(17):7653, 2024. [MDPI](https://www.mdpi.com/2071-1050/16/17/7653)
8. *Application of a Full-Scale Horizontal Anaerobic Digester for the Co-Digestion of Pig Manure, Food Waste, Excretion, and Thickened Sewage Sludge.* [Korea University](https://pure.korea.ac.kr/en/publications/application-of-a-full-scale-horizontal-anaerobic-digester-for-the/)

**알칼리도 기반 안정성 지표**
9. Ripley et al. (1986) *Improved alkalimetric monitoring for anaerobic digestion of high-strength waste.* [OSTI](https://www.osti.gov/biblio/5261202)
10. *Alkalinity ratios to identify process imbalances in anaerobic digesters treating source-sorted organic fraction of municipal wastes.* Biochem Eng J, 2013. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1369703X13000879)

**ML 예측 · 소프트센서 (전규모)**
11. *Machine learning for enhancing prediction of biogas production and building a VFA/ALK soft sensor in full-scale dry anaerobic digestion of kitchen food waste.* J Environ Manage, 2024. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0301479724031761)
12. *Prediction of biogas production of industrial scale anaerobic digestion plant by machine learning algorithms.* Chemosphere, 2023. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045653523012432)
13. *Feature Engineering and Supervised Machine Learning to Forecast Biogas Production during Municipal Anaerobic Co-Digestion.* ACS ES&T Eng, 2024. [ACS](https://pubs.acs.org/doi/10.1021/acsestengg.3c00435)
14. *PLS-based soft-sensor to predict ammonium concentration.* J Water Process Eng, 2022. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214714422001799)

**기전 모델 (ADM1)**
15. Batstone et al. *Anaerobic Digestion Model No.1 (ADM1).* [링크](https://www.researchgate.net/publication/11198259_Anaerobic_digestion_model_No_1_ADM1)
16. *Modifications to ADM1 for enhanced understanding and application — a comprehensive review.* Water Research, 2023. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0043135423009442)
17. *Calibration of ADM1 for steady-state anaerobic co-digestion of municipal wastewater sludge with restaurant grease trap waste.* Chem Eng J, 2015. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894714017033)
18. *Modification of ADM1 with Machine learning models (M−ADM1).* Chem Eng J, 2023. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894722058491)
19. *A hybrid approach of ADM1 and machine learning to model and optimize continuous anaerobic digestion processes.* Biomass Bioenergy, 2024. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0961953424001296)

**통계 방법**
20. Hamed, K.H. & Rao, A.R. (1998) *A modified Mann-Kendall trend test for autocorrelated data.* J Hydrol 204:182-196. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S002216949700125X)

**이론 수율**
21. Zhang, Y. *Anaerobic digestion fundamentals II: Thermodynamics.* VALORGAS. [PDF](https://www.valorgas.soton.ac.uk/Pub_docs/JyU%20SS%202013/VALORGAS_JyU_2013_Lecture%202.pdf)
22. *Theoretical values of biogas COD–CH₄ yield (0.35 L CH₄/g COD).* [링크](https://www.scribd.com/document/267605068/Theoritical-Values-of-Biogas-COD-CH4-Yield-0-35L-CH4-g-COD-1)

---

*검토 수행: 영천 저장소 데이터(`data/영천BGP_MASTER_2018-2023.csv`, `data/master.xlsx`) 직접
재계산. 모든 수치는 본 문서 각 절에 계산 근거와 표본수를 병기했다.*
