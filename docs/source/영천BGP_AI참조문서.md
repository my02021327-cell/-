---
document_type: dataset_reference_spec
facility: 영천 BGP (Yeongcheon Biogas Plant)
process_type: two_phase_anaerobic_digestion
process_stages: [산발효조(acidogenic_reactor), 혐기성소화조_A(anaerobic_digester_A), 혐기성소화조_B(anaerobic_digester_B), SBR_A, SBR_B, 연계처리수조]
substrate_mix: [음폐수(food_waste_leachate), 가축분뇨(livestock_manure), 음식물(food_waste)]
coverage_period: 2018-01-01_to_2023-09-17
data_frame_shape: [2191_rows, 154_cols_full / 40_cols_model]
source_workbooks: 6
project_context: BioGuard-AI (methane_prediction_and_instability_detection)
document_purpose: machine_readable_reference_for_downstream_AI_agents_and_modeling_code
supersedes: 영천BGP_완전분석보고서.md (narrative_version, same_underlying_findings)
status: validated (cross-checked against 사업소운영일지 annual totals, exact match confirmed)
---

# 영천 BGP 데이터셋 참조 문서 (AI 판독용)

이 문서는 산문형 보고서(`영천BGP_완전분석보고서.md`)와 동일한 검증 결과를 담되, 다른 AI 에이전트나 코드가 직접 파싱해 사용하기 쉬운 구조로 재구성한 것입니다. 서술형 설명을 최소화하고 표·키-값·코드블록 중심으로 작성했습니다.

---

## 0. QUICK REFERENCE (최우선 확인 사항)

```yaml
critical_facts:
  - fact: "2023년 데이터는 09-17까지만 존재. 10~12월 105행은 공백."
    action: "학습/집계 시 2023-09-18 이후 행 제외"
  - fact: "T_A(소화조 A 온도) 센서 2회 표류 후 고착. 실제 온도 아님."
    action: "T_A 사용 금지 구간: 2020-11-11~2020-12-31, 2022-10-06~종료"
  - fact: "T_B 2023-06-16부터 36.10℃ 고착."
    action: "T_B 사용 금지 구간: 2023-06-16~종료"
  - fact: "2023년 ALK_B 컬럼이 VFA_B 값으로 오염(71% 동일값)."
    action: "2023년 ALK_B, VFA_ALK_B 사용 금지"
  - fact: "in_VS(투입 VS%)가 2020년 기점으로 계단식 하락(VS/TS 0.80→0.70). 측정법 변경 추정."
    action: "in_VS 기반 지표는 2020 전후 분리하거나 in_TS로 대체"
  - fact: "바이오가스 시계열은 persistence(전일값)로 R²=0.831(전기간), 검증구간 R²=0.932. 지속성 모델을 반드시 베이스라인으로 사용."
    action: "절대량 예측 대신 Δy_t 예측 또는 persistence+Δfeed 잔차모델 사용"
  - fact: "성능저하 근본원인은 유리암모니아(FAN) 저해로 판정. VFA/Alk는 이 실패모드를 탐지 못함(오히려 개선되는 것처럼 보임)."
    action: "안정성 모니터링에 FAN 추정치를 VFA/Alk와 병행 사용"
  - fact: "개별 VFA 6종(아세트산/프로피온산 등) 전체 기간 미측정. NH3-N도 2022년 3건, 2023년 0건."
    action: "이 변수들은 특징(feature)으로 사용 불가. 측정 공백으로 문서화만 가능"
  - fact: "A/B 반응조는 병렬 동일조건 운전, 상관 r=0.927. 독립표본 아님."
    action: "A,B를 별개 샘플로 취급 금지 (유효표본 2배 착각 방지)"
  - fact: "`분석` 시트는 2020년+ 파일에서 날짜라벨만 2019 고정, 값 자체는 정확. `검토 사항` 시트는 2020년부터 실적값 자체가 손상."
    action: "두 시트 모두 원천 데이터로 사용 금지. Data 시트만 사용."
```

---

## 1. 데이터 소스 구조

### 1.1 원본 파일

```
18년도_영천BGP_운영관리현황_최종본.xlsx   (2018, 최종본)
19년도_영천BGP_운영관리현황_최종본.xlsx   (2019, 최종본)
20년도_영천BGP_운영관리현황_최종본.xlsx   (2020, 최종본)
21년도_영천BGP_운영관리현황_최종본.xlsx   (2021, 최종본)
22년도_영천BGP_운영관리현황_최종본.xlsx   (2022, 최종본)
23년도_영천BGP_운영관리현황_관리본.xlsx   (2023, 관리본=미완결, 09-17까지)
```

### 1.2 시트별 신뢰도 매트릭스

| 시트명 | 용도 | 신뢰도 | 사용 여부 | 비고 |
|---|---|---|---|---|
| `Data` | 일별 원자료 | **HIGH** | **주 분석 대상** | 결측 낮음(4.8~99.8% 변동), 년계 교차검증 완전 일치 |
| `Data 계산` | 부하율·수율 파생계산 | **HIGH** | 설계값·HRT·OLR 소스로 사용 | 소화조 용량(4,000+4,000㎥), 설계반입(209.05t/d) 등 |
| `사업소운영일지` | 일일 종합운영일지 | **HIGH** | 년계·에너지수지 소스로 사용 | `Data`와 년계 완전 일치 확인(2022: 가스 3,559,846㎥ 일치) |
| `차트` | 임베디드 산점도 16개 | **N/A** | 사용 안 함(불필요) | 전부 `Data 계산` 컬럼을 그대로 시각화. 신규 정보 없음(검증완료) |
| `Event Story` | 일별 특이사항 | **EMPTY** | 사용 불가 | 6개 파일 전부 공란(781~1174행 전부 placeholder) |
| `분석` | Data 시트 미러 | **MEDIUM** | **사용 안 함**(권장, 대체정보 있음) | 비REF값=Data와 정확히 일치. 단 2020+ 날짜라벨 2019고정, REF는 2018-폐지필드 참조. 상세: §5 |
| `검토 사항` | 주간 운전현황 요약 | **LOW (2020+)** / **MEDIUM (2018-19)** | **사용 안 함** | 2019까지 정상, 2020부터 점진 손상, 2021+ 심각손상(소화조 pH가 산발효조 pH 근사값으로 대체된 정황). 상세: §5 |

### 1.3 `Data` 시트 헤더 구조

```
row2: 공정단위 (반입 / 여액저장조 / 유기산화조 / 혐기성소화조 / 혼합슬러지탈리액 / 유량조정조 / 포기조 / 연계처리수조 / 슬러지수분함량 / 약품투입)
row3: 계열 (#A / #B / A+B)
row4: 측정항목 (pH, SS, COD, TS, VS, VFA, Alk, 메탄함량, 바이오가스발생량 ...)
row5: (blank)
row6~: 일별 데이터 (col1=날짜)
```

키 생성 규칙: `f"{row2_ffill} / {row3_ffill} / {row4}"` (2·3행은 forward-fill 후 결합)

### 1.4 공정 흐름 (2상 AD)

```
반입(음폐수+가축분뇨+음식물)
  → 여액저장조
  → 유기산화조(=산발효조, pH~4.85)
  → 혐기성소화조 #A(4,000㎥)/#B(4,000㎥) 병렬, 중온38℃, pH~7.95
  → 혼합슬러지탈수
  → 유량조정조 → 포기조(SBR #A/#B) → 연계처리수조 → 방류
```

---

## 2. 컬럼 스키마 (raw → clean 매핑)

`영천BGP_통합_분석용데이터_2018-2023.csv`의 실제 컬럼명입니다. 코드에서 직접 참조 가능합니다.

```python
SCHEMA = {
  # 식별자
  "date":            {"dtype": "datetime64", "desc": "관측일"},
  "year":            {"dtype": "int32"},
  "month":           {"dtype": "int32"},
  "dow":             {"dtype": "int32", "desc": "0=월요일 ... 6=일요일"},

  # 반입 (t/d)
  "feed_in_total":   {"raw": "반입 / 반입량",            "unit": "t/d", "missing_pct": 6.5},
  "food_ww":         {"raw": "반입 / 음폐수",             "unit": "t/d", "missing_pct": 8.0},
  "manure":          {"raw": "반입 / 가축분뇨",           "unit": "t/d", "missing_pct": 9.0},
  "food":            {"raw": "반입 / 음식물",             "unit": "t/d", "missing_pct": 8.7},

  # 소화조 투입
  "feed_A":          {"raw": "혐기성소화조 / #A / 투입량", "unit": "t/d", "missing_pct": 5.0},
  "feed_B":          {"raw": "혐기성소화조 / #B / 투입량", "unit": "t/d", "missing_pct": 4.9},
  "feed_AB":         {"derived": "feed_A + feed_B",       "unit": "t/d", "missing_pct": 4.9,
                       "corr_with_biogas_AB": 0.540, "predictive_rank": 1},

  # 투입 기질 성상 (유기산화조=소화조 투입 직전)
  "in_TS":           {"raw": "유기산화조(소화조 투입) / TS(%)", "unit": "%", "corr_with_biogas_AB": 0.522},
  "in_VS":            {"raw": "유기산화조(소화조 투입) / VS(%)", "unit": "%",
                       "WARNING": "2020년 기점 계단식 하락(VS/TS 0.80→0.70). TS와 완전공선(다중회귀 t=-0.2). 사용 비권장"},
  "in_COD":          {"raw": "유기산화조(소화조 투입) / COD(cr)", "unit": "mg/L"},
  "pH_acid":         {"raw": "유기산화조(소화조 투입) / pH",  "unit": "-",
                       "corr_with_biogas_AB": -0.301, "note": "가수분해효율 대리지표"},
  "VS_load_t":       {"derived": "feed_AB * in_VS / 100",  "unit": "t VS/d"},

  # 소화조 상태
  "T_A":             {"raw": "혐기성소화조 / #A / 온도(하)", "unit": "℃",
                       "SENSOR_FAULT": "2020-11-11~2020-12-31, 2022-10-06~종료 사용금지"},
  "T_B":             {"raw": "혐기성소화조 / #B / 온도(하)", "unit": "℃",
                       "SENSOR_FAULT": "2023-06-16~종료 사용금지(36.10℃ 고착)"},
  "pH_A":            {"raw": "혐기성소화조 / #A / pH",       "unit": "-", "mean": 7.945, "cv_pct": 1.69},
  "pH_B":            {"raw": "혐기성소화조 / #B / pH",       "unit": "-", "mean": 7.940},
  "VFA_A":           {"raw": "혐기성소화조 / #A / VFA",      "unit": "mg/L", "mean": 2907, "median": 2836},
  "ALK_A":           {"raw": "혐기성소화조 / #A / Alk",      "unit": "mg/L as CaCO3", "mean": 17026},
  "VFA_ALK_A":       {"derived": "VFA_A / ALK_A",           "mean": 0.172, "threshold_unstable": ">0.3~0.4",
                       "CAVEAT": "TAN 저해 상황에서는 개선되는 것처럼 보이나 실은 저해가 진행 중일 수 있음. §7 참조"},
  "VFA_B":           {"raw": "혐기성소화조 / #B / VFA",      "unit": "mg/L"},
  "ALK_B":           {"raw": "혐기성소화조 / #B / Alk",      "unit": "mg/L",
                       "DATA_CORRUPTION": "2023년 전체는 VFA_B 값으로 오염(71%동일). 2023년 사용금지"},
  "VFA_ALK_B":       {"derived": "VFA_B / ALK_B", "note": "2023년 사용금지(위와 동일 사유)"},
  "NH3_A":           {"raw": "혐기성소화조 / #A / NH3-N",   "unit": "mg/L (TAN)", "n_valid": 230,
                       "missing_note": "2022년 3건, 2023년 0건. 사실상 최근 미측정"},
  "NH3_B":           {"raw": "혐기성소화조 / #B / NH3-N",   "unit": "mg/L", "n_valid": 230},
  "dig_COD_A":       {"raw": "혐기성소화조 / #A / COD(cr)", "unit": "mg/L"},
  "dig_COD_B":       {"raw": "혐기성소화조 / #B / COD(cr)", "unit": "mg/L"},

  # 가스
  "biogas_A":        {"raw": "혐기성소화조 / #A / 바이오가스발생량", "unit": "m3/d", "missing_pct": 4.9},
  "biogas_B":        {"raw": "혐기성소화조 / #B / 바이오가스발생량", "unit": "m3/d", "missing_pct": 4.9},
  "biogas_AB":       {"raw": "혐기성소화조 / A+B / 가스발생량 (fillna biogas_A+biogas_B)",
                       "unit": "m3/d", "missing_pct": 4.8, "mean": 10364, "std": 2456, "cv_pct": 23.7,
                       "PRIMARY_TARGET_CANDIDATE": true},
  "CH4_pct":         {"derived": "mean(소화조#A/#B 메탄함량)", "unit": "%", "mean": 66.0, "cv_pct": 5.2},
  "CH4_m3":          {"derived": "biogas_AB * CH4_pct / 100", "unit": "m3/d"},
  "gas_per_VS":      {"derived": "biogas_AB / VS_load_t", "unit": "m3 biogas/t VS", "median": 1155,
                       "WARNING": "in_VS 계단변화로 절대값 신뢰 낮음. 연도간 상대비교만 사용"},
  "CH4_per_VS":      {"derived": "CH4_m3 / VS_load_t", "unit": "m3 CH4/t VS", "median": 741},
}
```

### 2.1 데이터 스코프 밖 변수 (모델링 불가, 참고용 표기만)

```yaml
never_measured_variables:
  - 혐기성소화조_#A/#B_아세트산       # 6년간 0건
  - 혐기성소화조_#A/#B_프로피온산     # 6년간 0건
  - 혐기성소화조_#A/#B_부틸산_이소부틸산_발레르산_이소발레르산  # 6년간 0건
  - 소화조_SS_BOD_CODmn             # 2018년 헤더만 존재, 이후 폐지
  - 여액저장조_NH3-N                 # 전기간 0건

structural_missing_pattern:
  type: NOT_MCAR (구조적 결측)
  lab_analysis_items:
    weekday_measurement_rate: "83~86%"
    friday_measurement_rate: "~64%"
    weekend_measurement_rate: "~0.3~1.3%"
  flow_meter_items:  # 반입량, 투입량, 바이오가스
    daily_measurement_rate: "93.5~95.2%"
  implication: "결측 무작위 아님. 요일 층화 필수. 주말 선형보간 금지."
```

---

## 3. 데이터 품질 이슈 (ID 태깅)

```yaml
- id: DQ-01
  severity: HIGH
  title: 2023년_데이터_불완전
  affected_range: "2023-09-18 ~ 2023-12-31"
  symptom: "105일 전체 공백"
  root_cause: "23년 파일이 관리본(작업중 사본), 최종본 아님"
  action: "학습/집계 전 해당 구간 제거. 2023년 연간 합계를 타 연도와 비교 금지"

- id: DQ-02
  severity: HIGH
  title: T_A_센서_표류_고착
  affected_range: ["2020-11-11~2020-12-31 (38.3→28.6℃)", "2022-10-06~2023-05-11 (38.0→21.0℃)", "2023-06-16~종료 (30.10℃ 고착, 58일 연속 동일값)"]
  symptom: "하루 약 0.05℃씩 단조감소 후 연초 리셋 또는 특정값 고착"
  root_cause: "온도계 교정 불량 (실제 냉각 아님)"
  evidence: "저온기(T_A<35℃, n=228) vs 정상기(T_A≥37℃, n=1014)에서 A/B 가스분담률 50.7% vs 50.9%로 동일. 실제 21℃라면 활성 60%↓ 필요하나 관측되지 않음"
  action: "해당 구간 T_A를 NaN 처리"

- id: DQ-03
  severity: MEDIUM
  title: T_B_고착
  affected_range: "2023-06-16 ~ 종료"
  symptom: "36.10℃ 값 연속 반복"
  action: "해당 구간 T_B를 NaN 처리"

- id: DQ-04
  severity: MEDIUM
  title: 소화조_투입량_고정값_의심
  affected_range: "2020-01-18 기점 48일 연속"
  symptom: "192 t/d 정확히 동일값 반복 (feed_AB 최장 연속고착)"
  root_cause: "실측 아닌 계획값 입력 의심"
  action: "해당 구간 플래그 처리(제거는 보류)"

- id: DQ-05
  severity: HIGH
  title: 2023년_ALK_B_컬럼_오염
  affected_range: "2023-01-01 ~ 2023-09-17 (n=154)"
  symptom: "ALK_B 값이 VFA_B 값과 71.2% 완전 일치 (양쪽 평균 모두 2,997 mg/L)"
  root_cause: "원본 워크북 수식 참조 오류"
  action: "2023년 ALK_B, VFA_ALK_B 컬럼 사용 금지"

- id: DQ-06
  severity: LOW
  title: 물리범위_이탈값
  total_removed: 544
  breakdown:
    슬러지_TS퍼센트_3종: 535
    소화조A_pH_38.2: 4
    기타(여액저장조_VS132pct/소화조B_Alk_217994/메탄함량637pct/포기조B_pH): 5
  filter_ranges: "pH[3,10] / 온도[20,60]℃ / 메탄함량[30,80]% / TS,VS[0.1,40]% / VFA[50,20000] / Alk[500,40000] mg/L"
  action: "필터 범위 밖 값 NaN 처리 (이미 적용됨)"

- id: DQ-07
  severity: LOW
  title: 실험실항목_0값_이중의미
  rule: "실험실 분석항목(pH,VFA,COD,TS,VS,NH3-N)의 0=미측정→NaN 변환. 유량항목(반입량,투입량,가스발생량)의 0=실제0→유지"
  evidence: "일요일 반입량 22.3t/d(급감)이지만 소화조투입 196.5t/d·가스 10,539m3/d는 평일과 동일(저류조 완충). Kruskal-Wallis p=0.85"

- id: DQ-08
  severity: MEDIUM
  title: VS_측정법_계단변화
  affected_range: "2020년 기점"
  symptom: "VS/TS비 0.799(2019)→0.695(2020)로 한번에 -13%p 하락 후 유지(2021:0.717, 2022:0.688, 2023:0.690)"
  root_cause: "강열감량 분석법/회화조건/분석자 변경 추정 (기질조성 점진변화로는 설명 불가한 계단형)"
  action: "in_VS, VS_load_t, gas_per_VS 등 VS기반 지표는 2020 전후 분리하거나 in_TS로 대체"

- id: DQ-09
  severity: LOW
  title: 가스농도블록_부분오염
  affected_range: "2021~2023년 신설 컬럼 '비중/가스농도(%)' 블록"
  symptom: "452 유효행 중 206행(45.6%) CH4>90% 또는 CO2>50%로 인접 철염컬럼 값 혼입"
  action: "값>90% 또는 CO2>50% 행 제거 후 사용. 정제 후 CH4+CO2=99.99% 확인됨"

- id: DQ-10
  severity: INFO
  title: EventStory_전면공란
  symptom: "6개 파일 전부 781~1174행 placeholder만 존재, 실제 이벤트 기록 0건"
  implication: "지도학습 이상탐지 라벨 생성 불가"

- id: DQ-11
  severity: MEDIUM (문서용도, raw데이터 자체는 무관)
  title: 분석시트_날짜라벨_고정
  affected_files: "20,21,22,23년 파일"
  symptom: "날짜열이 2019-01-01부터 순차 표시(실제연도 아님). 단 값 자체는 해당 파일의 실제 순서 데이터와 정확히 일치 (검증: 22년파일 분석시트 R6 반입량131.55/가스11798 = Data계산시트 2022-01-01 값과 동일)"
  additional_symptom: "표본(500행) 내 REF!오류 11376~12426건. REF발생 컬럼은 정확히 2018년폐지필드(소화조/유기산화조 SS·BOD·CODmn, 연계처리수조 NH3/NO3/NO2-N, 슬러지수분함량+약품투입블록)"
  action: "분석시트 사용 안 함(대체정보 Data시트에 이미 있음). 만약 사용해야 한다면 날짜라벨 무시하고 행순서로만 해석, REF컬럼 제외"

- id: DQ-12
  severity: HIGH (해당시트 자체), N/A (raw Data 무관)
  title: 검토사항시트_실적값_손상
  affected_files: "2020~2023년 파일 (2018,2019는 정상)"
  symptom_by_year:
    2019: "정상 (최근1개월평균 pH_A=7.97 vs Data실측=7.95, 근소한 차이는 기간정의 차이)"
    2020: "손상시작 (pH_A=5.82 vs 7.88, NH4N=0 vs 3931, Alk_A=12429 vs 17206)"
    2021: "심각손상 (pH_A=3.60 vs 실제소화조7.99, but 산발효조pH실측4.70과 근사)"
    2022: "심각손상 (pH_A=4.03 vs 8.00, 산발효조pH실측5.11과 근사)"
    2023: "심각손상 (pH_A=4.38 vs 8.02, 산발효조pH실측5.14과 근사)"
  root_cause_hypothesis: "'기준'열은 2019년 값이 5개파일 모두 동일하게 복사됨(템플릿 재사용, 갱신안됨). 실적값은 고정 열위치 참조 수식이 2021년 Data시트 컬럼이동(§1.4 이동표)을 따라가지 못해 오참조 발생 추정. 2021+년 소화조A 표시pH가 실제로는 산발효조 pH 근처값을 반환하는 정황 있으나 완전일치는 아니라 미확정"
  action: "검토사항시트 실적값 전체 사용 금지. '기준'열(설계고정값)은 참고용으로만 사용"
```

---

## 4. 시설 설계 제원

```yaml
digester_capacity:
  tank_A_m3: 4000
  tank_B_m3: 4000
  total_m3: 8000
  source: "'Data 계산' 시트 col8,9,10 (전 연도 동일)"

design_targets:
  feed_total_t_per_day: 209.05
  feed_A_m3_per_day: 104.525
  feed_B_m3_per_day: 104.525
  influent_TS_pct: 6.02831
  influent_VS_pct: 5.03656
  digester_TS_pct: 1.26021
  digester_VS_pct: 1.05289
  temperature_range_C: [35, 38]
  NH4_N_design_mg_L: 3555.1
  bioreactor_influent_m3_per_day: 418
  source: "'검토 사항' 시트 기준열 (2019~2023 파일 전부 동일값)"

operational_summary_vs_design:
  HRT_median_days: 40.5
  HRT_p5_p95: [32.5, 55.6]
  daily_turnover_rate_pct: 2.47
  OLR_mean_kgVS_per_m3_day: 1.143
  OLR_p5_p95: [0.63, 1.70]
  OLR_literature_range_mesophilic: [1.5, 4.0]
  OLR_assessment: "설계/문헌 하단 → 부하 여력 충분, 과부하 아님"
  feed_vs_design_pct_by_year: {2018: 95.7, 2019: 104.6, 2020: 105.0, 2021: 102.4, 2022: 100.8, 2023: 92.8}

acf_theory_check:
  formula: "exp(-lag/HRT)"
  lag1_theory_vs_observed: [0.976, 0.916]
  lag7_theory_vs_observed: [0.841, 0.670]
  lag30_theory_vs_observed: [0.477, 0.262]
  conclusion: "관측 ACF가 이론 CSTR 감쇠와 형태 일치(잡음으로 하향 편의)"
```

---

## 5. 핵심 정량 관계식

### 5.1 지속성 베이스라인 (반드시 병기해야 하는 기준선)

```yaml
persistence_baseline:
  formula: "y_hat(t) = y(t-1)"
  full_period: {n: 2085, R2: 0.831, RMSE_m3: 1008, MAE_m3: 527, MAPE_pct: 5.35}
  validation_2022_2023: {n: 625, R2: 0.932, RMSE_m3: 460, MAE_m3: 319}

persistence_plus_delta_feed:
  formula: "y_hat(t) = y(t-1) + 33.26*(feed_AB(t)-feed_AB(t-1)) + 1.3"
  coefficient_fit_period: "2018-2021 (학습, 정보누출 없음)"
  validation_2022_2023: {n: 625, R2: 0.942, RMSE_m3: 425, MAE_m3: 293}
  improvement_over_persistence: {R2: "+0.010", RMSE_pct: "-7.4%", MAE_pct: "-8.2%"}

naive_regression_WITHOUT_time_structure:
  formula: "biogas_AB = a + b*feed_AB (시계열무시, 수준회귀)"
  validation_2022_2023: {n: 625, R2: 0.266, RMSE_m3: 1510, MAE_m3: 1312}
  WARNING: "시계열 특징(y_t-1) 없이 IID로 회귀하면 persistence 대비 R² 0.93→0.27로 붕괴. 반드시 lag특징 포함"

autocorrelation:
  biogas_AB_ACF: {lag1: 0.916, lag2: 0.842, lag3: 0.799, lag5: 0.724, lag7: 0.670, lag14: 0.507, lag30: 0.262, lag60: 0.027, lag90: -0.195, lag365: 0.011}
  diff1_ACF: {lag1: -0.059, lag2: -0.183, lag3: -0.014, lag5: -0.032, lag7: 0.046}
  conclusion: "차분후 백색잡음에 근접. 확률보행 구조. 연주기성 없음(lag365=0.011)"
```

### 5.2 다중회귀 (동시점, n=1227)

```yaml
multiple_regression_biogas_AB:
  R2: 0.633
  adj_R2: 0.630
  standardized_coefficients:
    feed_AB:      {coef: 1392.2, t: 33.1, p: "<1e-300", significant: true}
    in_TS:        {coef: 1174.8, t: 15.5, p: "<1e-300", significant: true}
    ALK_A:        {coef: 423.7,  t: 8.2,  p: 4.4e-16,   significant: true}
    pH_acid:      {coef: -422.9, t: -7.4, p: 2.2e-13,   significant: true}
    VFA_ALK_A:    {coef: 300.6,  t: 6.4,  p: 2.2e-10,   significant: true}
    CH4_pct:      {coef: 267.6,  t: 4.9,  p: 1.3e-06,   significant: true}
    T_B:          {coef: -71.1,  t: -1.7, p: 0.097,     significant: false}
    in_VS:        {coef: -13.9,  t: -0.2, p: 0.85,      significant: false, note: "TS와 완전공선성"}
  single_variable_R2:
    feed_AB: 0.292
    in_TS: 0.272
    in_VS: 0.202
    pH_acid: 0.090
    VFA_ALK_A: 0.045
    ALK_A: 0.024
    CH4_pct: 0.003
    T_B: 0.001
```

### 5.3 유리암모니아(FAN) 계산

```python
def calc_FAN(TAN_mgL, pH, T_celsius=38.0):
    """
    TAN_mgL: 총암모니아성질소 (NH3_A 또는 NH3_B 컬럼, mg/L)
    pH: 해당 반응조 pH
    T_celsius: 온도, 결측시 38.0 대입(센서결함 구간 회피)
    반환: FAN (유리암모니아, mg/L as N)
    출처: Anthonisen et al.
    """
    T_kelvin = T_celsius + 273.15
    pKa = 0.09018 + 2729.92 / T_kelvin
    FAN = TAN_mgL / (1 + 10**(pKa - pH))
    return FAN
```

```yaml
thresholds_mgL:
  no_effect: "<150"
  inhibition_onset: "150-300"
  clear_inhibition: "300-600"
  severe_risk: ">600"

observed_2018_2023:
  reactor_A: {n: 229, TAN_mean: 3832, FAN_mean: 410, FAN_median: 380, FAN_max: 831,
              pct_FAN_gt_150: 98.7, pct_FAN_gt_300: 75.1, pct_TAN_gt_3000: 87.8}
  reactor_B: {n: 230, TAN_mean: 3997, FAN_mean: 430, FAN_median: 393, FAN_max: 996,
              pct_FAN_gt_150: 98.3, pct_FAN_gt_300: 77.8}
  yearly_FAN_mgL: {2018: 306, 2019: 370, 2020: 386, 2021: 585, 2022: 428, 2023: null_no_data}
  assessment: "6년 내내 명확한 저해구간(300~600) 운전. 미생물 순치(acclimation) 가설로만 붕괴 미발생 설명 가능"
```

### 5.4 NH3-N ↔ 알칼리도 관계 (기전 확인됨)

```yaml
mechanism: "단백질/요소 분해 → NH4+ 생성 → CO2와 결합 → NH4HCO3(중탄산암모늄) 형성 → 총알칼리도로 측정됨"

empirical_regression:
  formula: "ALK_A = 14089 + 0.640 * NH3_A"
  n: 230
  Pearson_r: 0.587
  R2: 0.345
  p: 1.0e-22
  Spearman_rho: 0.615

theoretical_check:
  theoretical_slope_mg_CaCO3_per_mgN: 3.57   # = (50/14), 전량 중탄산형태 가정시
  observed_slope: 0.640
  observed_to_theoretical_ratio: 0.18
  interpretation: "TAN 증가분의 약 18%만 알칼리도 증가로 직접 설명됨. 나머지는 TAN과 무관한 기저 탄산완충계(절편 14,089 mg/L)"

practical_implication: >
  알칼리도 상승을 '완충능 개선'으로만 해석하면 안 됨. 이 시설처럼 고암모니아 기질 비중이 높은 경우,
  알칼리도 상승분의 일부는 저해물질(암모니아) 자체의 증가 신호. VFA/Alk 비율이 개선되는 것처럼 보이나
  실은 분모(Alk)가 저해원인 때문에 커지는 역설 발생 (§7 참조).
```

### 5.5 성능저하 인과경로 (검증된 상관 사슬)

```yaml
causal_chain:
  step1: {var: manure_share, trend: "70.6→74.6 t/d (2018→2023)", direction: increase}
  step2: {var: NH3_A, trend: "3337→4913 mg/L (2018→2021, +104%)", kendall_tau: 0.571, p: 5.0e-38}
  step3: {var: ALK_A, trend: "15806→18028 mg/L (+19.8%)", kendall_tau: 0.622, p: 4.0e-211}
  step4: {var: pH_A, trend: "7.85→8.02 (+0.17)", kendall_tau: 0.261, p: 3.9e-38}
  step5: {var: FAN, trend: "306→585 mg/L (2018→2021)", derived_from: [step2,step4]}
  step6a: {var: CH4_pct, trend: "-4.5% (2018-2022)", kendall_tau: -0.137, p: 7.9e-12}
  step6b: {var: CH4_production, trend: "-8.1% (7080→6430 m3/d, 2018-2022)", kendall_tau: -0.054, p: 6.7e-03}
  step6c: {var: CH4_per_feed_ton, trend: "36.2→30.1 m3/t (-17%, 2018-2022)"}
  supporting_evidence:
    - "VFA_A 오히려 감소(-9.5%): 산성화가 아니라 메탄생성단계 저해임을 시사"
    - "VFA_ALK_A 오히려 감소(0.21→0.16): 관리지표상 '개선'처럼 보이나 분모(Alk) 증가가 원인이므로 착시"
    - "OLR 1.14 kgVS/m3d로 여유(과부하 아님), HRT 40.5일로 충분(체류시간 부족 아님) → 소거법으로 암모니아 저해가 유력 원인"
  falsifiable_via: "16S rRNA 군집분석. 예상: Methanosaeta 억제, Methanoculleus/Methanobacterium 우점, SAOB(Syntrophaceticus 등) 검출"
```

---

## 6. 모델링 스펙

```yaml
target_definition:
  DO_NOT_predict: "biogas_AB 절대값 (persistence가 이미 R²=0.831~0.932)"
  RECOMMENDED_targets:
    - name: delta_biogas
      formula: "biogas_AB(t) - biogas_AB(t-1)"
      rationale: "persistence의 기준 R²=0이므로 정직한 평가 가능"
    - name: residual_anomaly_flag
      formula: "|residual of persistence+delta_feed model| > 3*sigma, 2일 연속"
    - name: FAN_estimate
      formula: "calc_FAN(NH3_A, pH_A, T=38.0)"
      rationale: "성능저하 근본원인 추적, 조기경보 1순위 후보"

feature_include:
  tier1_required:
    - {name: "biogas_AB_lag1", source: "biogas_AB(t-1)", rationale: "ACF(1)=0.916"}
    - {name: "delta_feed_AB", source: "feed_AB(t)-feed_AB(t-1)", rationale: "잔차R²=0.418, 계수33.26"}
  tier2_useful:
    - {name: "in_TS", rationale: "다중회귀 t=15.5, VS 대신 사용"}
    - {name: "pH_acid", rationale: "가수분해효율 대리지표, 2상AD 고유, t=-7.4"}
    - {name: "ALK_A", rationale: "완충능 상태, t=8.2"}
    - {name: "VFA_ALK_A", rationale: "안정성 상태, t=6.4"}
    - {name: "CH4_pct", rationale: "가스품질, t=4.9"}
  tier3_conditional:
    - {name: "FAN_estimate", condition: "NH3_A 측정 확보시에만", rationale: "§5.5 근본원인 변수"}

feature_exclude:
  - {name: "in_VS", reason: "in_TS와 완전공선성(t=-0.2,p=0.85), 2020년 측정법변경으로 비연속"}
  - {name: "T_A, T_B", reason: "단독R²=0.001, CV<2.6%(변동없음), 센서결함구간 존재(DQ-02,DQ-03)"}
  - {name: "manure, food (기질별 반입량)", reason: "r≈0.00"}
  - {name: "feed_in_total (반입량 합계)", reason: "r=0.138, 저류조에서 완전평활화됨. feed_AB 사용"}
  - {name: "lag_12_to_15_days_features", reason: "탐색상한 부근 위상관, 1차차분후 소멸(§5.1 참고)"}
  - {name: "H2S_column", reason: "Data시트 값은 0.01%(100ppm) 고정 설계가정치, 실측 아님. 실측은 사업소운영일지에 별도 존재(50~80ppm)"}
  - {name: "개별_VFA_6종", reason: "전기간 미측정 (§2.1)"}

validation_protocol:
  split_method: "시간순 분할 (chronological split)"
  recommended_split: "train=2018~2021, test=2022~2023"
  FORBIDDEN: "무작위 K-fold (ACF(1)=0.916로 인접일 누출, R² 과대평가)"
  weekday_handling: "요일 층화 유지 또는 결측마스크 보존 (주말 선형보간 금지, §2.1 structural_missing_pattern)"
  mandatory_baseline_report: "persistence 모델 R²/RMSE/MAE/MAPE를 항상 병기"
  regime_flags:
    - {period: "2018년 전체", flag: "high_variance_regime", note: "가스CV 36.8%(vs 이후 16~21%), ±35%급변일 30건(vs 이후 0~2건). 이상탐지 학습에서 제외 또는 가중축소 권장. 회귀모델에는 포함 무방"}
    - {period: "A vs B reactor", flag: "not_independent_samples", note: "상관r=0.927, 병렬동일조건. 별도 샘플 취급 금지"}

anomaly_detection_reality_check:
  total_days: 2191
  multi_indicator_simultaneous_anomalies: 4
  genuine_process_events: 1  # 2018-09-10
  labeled_events_from_EventStory: 0
  implication: "지도학습 이상탐지 불가능(양성표본 1건). 규칙기반 + 잔차관리도로 설계"
  recommended_rules:
    L1_FAN: {metric: "calc_FAN(NH3_A,pH_A,38.0)", threshold: ">300 mg/L", priority: highest}
    L2_VFA_ALK: {metric: "VFA_ALK_A 3일이동평균", threshold: ">0.30", priority: medium, caveat: "둔감(6년간0.4%만초과)"}
    L3_residual: {metric: "persistence+delta_feed 모델잔차 |z|", threshold: ">3, 2일연속", priority: high}
    L4_combo: {metric: "알칼리도 3일연속하락 + VFA상승 동시", priority: medium}
```

---

## 7. 연도별 핵심 통계 (요약 테이블)

| 변수 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023* |
|---|---|---|---|---|---|---|
| feed_in_total (t/d) | 200.1 | 218.6 | 219.5 | 214.1 | 210.6 | 194.0 |
| feed_AB (t/d) | 194.8 | 220.9 | 195.2 | 188.3 | 190.3 | 190.0 |
| in_TS (%) | 6.77 | 6.60 | 5.81 | 6.45 | 5.93 | 5.91 |
| in_VS (%) [주의:DQ-08] | 5.52 | 5.25 | 4.10 | 4.65 | 4.11 | 4.13 |
| biogas_AB (m3/d) | 10818 | 11020 | 10102 | 10607 | 9753 | 9694 |
| CH4_pct (%) | 67.0 | 66.8 | 68.7 | 62.2 | 65.5 | 65.8 |
| CH4_m3 (m3/d) | 7080 | 7346 | 6937 | 6568 | 6430 | 6374 |
| pH_A | 7.85 | 7.95 | 7.88 | 7.99 | 8.00 | 8.02 |
| pH_acid | 4.84 | 4.52 | 4.91 | 4.70 | 5.11 | 5.14 |
| VFA_A (mg/L) | 3259 | 2773 | 2742 | 2919 | 2796 | 3072 |
| ALK_A (mg/L) | 15806 | 15862 | 17206 | 17452 | 18028 | 18047 |
| VFA_ALK_A | 0.21 | 0.18 | 0.16 | 0.17 | 0.16 | 0.17 |
| NH3_A (mg/L) | 3337 | 3292 | 3931 | 4913 | 4241 | null |
| T_A (℃) [주의:DQ-02] | 38.28 | 38.50 | 38.06 | 38.10 | 36.58 | 26.44 |
| T_B (℃) [주의:DQ-03] | 38.36 | 38.57 | 39.06 | 38.18 | 38.52 | 36.66 |

\* 2023 = 09-17까지, 연간합계 타연도 비교불가 (DQ-01)

---

## 8. 에너지·물질 수지

```yaml
energy_balance_2022:
  source: 사업소운영일지 년계 (cross-validated with Data sheet, exact match)
  power_consumption_MWh: 5576.64
  power_generation_MWh: 6336.595
  self_sufficiency_pct: 113.6
  biogas_production_m3: 3559846
  biogas_utilization_m3: 3259018
  utilization_rate_pct: 91.5
  generation_efficiency_kWh_per_m3: 1.944

energy_balance_2023_partial:
  period: "01-01 ~ 09-17"
  power_consumption_MWh: 3900.47
  power_generation_MWh: 4427.047
  self_sufficiency_pct: 113.5

material_balance_5yr_avg_2018_2022:
  feed_t_per_yr: 76014
  biogas_m3_per_yr: 3819868
  CH4_m3_per_yr: 2515351
  CH4_per_feed_ton_m3: 33.1
  thermal_energy_MWh_per_yr: 25003
  CO2_avoided_tCO2e_per_yr: 45025
  assumptions: {CH4_LHV_kWh_m3: 9.94, CH4_density_kg_m3: 0.716, GWP100_CH4: 25}

removal_efficiency:
  COD_cr_removal_pct_mean: 80.4
  COD_cr_removal_pct_median: 81.1
  VS_removal_pct_mean: 73.5
  gas_yield_m3_per_tVS_median: 1155
  CH4_yield_m3_per_tVS_median: 741
  CAVEAT: "메탄수율 741이 문헌치(400-550)초과 → in_VS 과소평가 가능성(DQ-08). 절대값보다 상대변화 신뢰"

chemical_usage_2022_annual_kg:
  철염(ferric_salt): 293063
  메탄올(methanol): 41017
  폴리머_고상: 33240
  황산: 18572
  가성소다: 14581

SBR_treatment_efficiency:
  TN_removal_pct: 96.9
  COD_mn_removal_pct: 95.1
  influent_TN_mgL: 3032
  effluent_TN_mgL: 94
```

---

## 9. 파일 인벤토리

```yaml
outputs:
  - path: 영천BGP_완전분석보고서.md
    type: narrative_report
    content: "13부+부록2, 서술형 전체 분석 (본 문서와 동일 사실관계, 프레젠테이션용)"
  - path: 영천BGP_AI참조문서.md
    type: structured_reference
    content: "본 문서. 기계판독 최적화"
  - path: 영천BGP_통합_분석용데이터_2018-2023.csv
    type: model_ready_dataset
    shape: [2191, 40]
    schema: "§2 SCHEMA 참조"
  - path: 영천BGP_전체컬럼_원본통합.csv
    type: full_raw_merged
    shape: [2191, 154]
  - path: 부록_전체변수_인벤토리.csv
    type: variable_catalog
    shape: [165, 8]
    columns: [변수, 연도존재여부, 유효n, 결측pct, 평균, 중앙값, 최소, 최대]
  - path: fig/f1.png ~ f9.png
    type: charts
    list:
      f1: "6개년 운전 시계열 개관 (가스/투입/CH4%/온도)"
      f2: "공정안정성 지표 (pH,VFA/Alk,Alk,VFA,NH3-N 히스토그램/막대)"
      f3: "투입-가스 관계 (수준 및 1차차분 산점도)"
      f4: "ACF/차분ACF/베이스라인성능 비교"
      f5: "계절성/A·B대칭성/수율추이"
      f6: "데이터 가용성 맵 (변수x연도 히트맵)"
      f7: "설계대비운전 (HRT/OLR/가동률)"
      f8: "후단처리공정 (SBR질소제거/MLSS/약품)"
      f9: "에너지수지 (전력자립도/가스이용률)"
```

---

## 10. 변경 이력 (검증 이터레이션 기록)

```yaml
change_log:
  - iteration: 1
    date_context: "초기 통합"
    action: "6개 워크북 Data시트 이름기반 매핑 통합, 165변수 → 2191일"
  - iteration: 2
    date_context: "1차 정제"
    action: "센서결함(T_A,T_B) 발견, VS계단변화(DQ-08) 발견, 물리범위필터 적용, 사업소운영일지 년계 교차검증 완료"
  - iteration: 3
    date_context: "전체분석보고서 작성"
    action: "13부 완전판 작성. Data계산/사업소운영일지/검토사항/차트/EventStory 시트 최초 검토. 검토사항·분석시트 초기 문제 발견(당시 '사용금지'로 단순 판정)"
  - iteration: 4
    date_context: "NH3-알칼리도 관계 질의 대응"
    action: "§5.4 기전+실측검증 추가 (Anthonisen 이론기울기 vs 관측기울기 비교)"
  - iteration: 5
    date_context: "분석/검토사항 시트 재검증 요청 대응"
    action: >
      분석시트: 6개년 전수 REF카운트/날짜라벨 정량화. REF컬럼=2018폐지필드로 정확히 매칭 확인.
      비REF값이 Data시트와 완전일치함을 검증(22년파일 예시) → 기존 '사용금지' 판정을 '날짜라벨무시하고 사용가능'으로 정정(DQ-11).
      검토사항시트: 좌표고정 방식으로 5개년 '최근1개월평균' 전수 추출, Data실측 연평균과 정량대조.
      2019=정상, 2020=손상시작, 2021-2023=심각손상(소화조pH 표시값이 산발효조pH 실측과 근사하는 정황 발견) 확인 (DQ-12).
  - iteration: 6
    date_context: "본 문서 (AI참조용 구조화)"
    action: "전체 검증결과를 기계판독 최적화 구조(YAML/table/code block)로 재구성. 서술형 보고서와 병존."
```
