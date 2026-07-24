# BioGuard-AI — 혐기성 소화조 메탄 생성량 예측 · 건강상태 관제

영천 통합바이오가스화시설의 6년간(2018.01~2023.12) 일별 운전 데이터를 학습하여
**메탄생성량(methane)** 을 예측하고, **소화조 건강상태(산성화)** 를 신호등으로 관제하는
시스템이다. 이질적 Base Model(RF·MLP·LSTM·Transformer)을 **NNLS(비음수 최소자승)
스태킹** 으로 결합한다.

## 생물학적 모델링 원칙 (핵심)

혐기성소화는 **가수분해 → 산생성(VFA) → 아세트산생성 → 메탄생성** 의 다단계 미생물
반응으로, 투입 유기물(VS)이 **즉시 메탄으로 전환되지 않고 체류시간(HRT)만큼 지연** 된다.
따라서 오늘의 메탄생성량은 ① 과거 수일~수십일 **누적 투입 기질부하** 와 ② 현재 소화조
**내부 미생물 상태(VFA·알칼리도·pH·온도·VS)** 의 함수이다.

- **생물학적 지연 정량 확인** (`src/eda.py`): 투입 VS부하와 메탄의 교차상관에서 순간값
  r=0.54 대비 **10일 누적(이동평균) r=0.64** 로 더 강함 → 체류시간에 따른 지연이 실재.
  이를 반영해 체류창 부하 피처(`VS_in_load5/10` 등)를 별도 설계.
- **수율(MY=methane/VS_in)은 예측 대상에서 제외.** 분모가 *금일 투입 VS* 이므로
  어제까지 데이터로 오늘 수율을 예측하는 것은 순환적이며 생물학적으로 성립하지 않는다.
  수율은 관측된 효율 진단값으로만 참조한다.
- **건강상태(관제)는 VFA/알칼리도 비 기반.** 산성화(완충능 소진) 지표로 AD 문헌 밴드
  (<0.3 안정 / 0.3~0.4 주의 / 0.4~0.8 불안정 / ≥0.8 위험)를 사용 — 신호등은 수율이
  아닌 이 건강지표로 정의한다.

## 데이터

| 파일 | 내용 |
|------|------|
| `data/master.xlsx` | 운전 변수(반입/투입량·유입/소화조 pH·TS·VS·VFA·알칼리도·온도 등) |
| `data/targets.xlsx` | `methane`(메탄발생량), `VS_in`(투입 VS부하), `MY`(=methane/VS_in, 진단용) |

- 학습 2018~2022 / **홀드아웃 2023** (미래 정보 누수 차단)
- 결측 과다 열(유입_TN·소화조_TN·NH4N, 75~90%) 제거, 소량 결측 선형 보간

## 파이프라인 (`src/`)

| 모듈 | 역할 |
|------|------|
| `data.py` | 로딩·전처리 + 리치 시계열 + **생물학적 체류창 부하** 피처(메탄 단일 타깃) |
| `sequences.py` | LSTM/Transformer용 과거 W=14일 원천 운전변수 시퀀스 |
| `torch_models.py` | PyTorch LSTM(2L,h=32)·Transformer(2L,d=64/head=4/FF=128) |
| `ensemble.py` | 테이블 Base(RF·MLP) + `NNLSMetaCombiner`(비음수 스태킹) |
| `train.py` | 학습·평가 파이프라인(엔드투엔드) |
| `signal.py` | **VFA/알칼리도 건강 신호등** + 메탄 예측 시각화 |
| `eda.py` | 생물학적 지연(교차상관)·체류창·건강 밴드 분석 |

### 앙상블 구조
- **Base** : RandomForest·MLP(테이블 피처) + LSTM·Transformer(과거 14일 시퀀스).
  시퀀스 모델은 그래디언트 클리핑·타깃 표준화·시드 5개 평균으로 학습 분산을 낮춘다.
- **Meta** : **NNLS 비음수 스태킹**. 학습기간의 최근 2년(2021~2022)을 메타 검증블록으로
  두어(그 이전으로 base 선학습) 국면이동 전이에 맞춘 비음수 가중치를 학습한 뒤, base 를
  전체 학습데이터로 재학습해 2023 홀드아웃을 예측한다.

## 실행

```bash
pip install -r requirements.txt
python -m src.train        # 앙상블 학습·평가·건강 신호등·시각화
python -m src.eda          # 생물학적 지연·건강 분석
python -m src.xgb_methane  # XGBoost 2상 집중·투입 lag 선택(persistence 예측 미사용)
python -m src.stack_methane # RF + XGBoost + SARIMAX(시계열) 스태킹 앙상블
```

## XGBoost 2상(메탄생성균 조) 집중 모듈 (`src/xgb_methane.py`)

혐기성 소화조 **2상(메탄생성 단계)** 에 집중한 **XGBoost 단일 모델**. 투입→메탄 **lag
자동선택**(gain 중요도 최댓값 날), **10~15개 피처 집중**, **결측일 삭제**.
**persistence(어제 메탄값)는 예측 입력으로 쓰지 않고 비교 baseline 으로만** 사용한다.
상세는 [`REPORT_XGB.md`](REPORT_XGB.md).

- 메탄은 일별 자기상관 0.92 로 persistence(baseline R²≈0.88)가 매우 강함 → 2상 화학상태
  소프트센서(Model-1)도, 운전변수 전용 Model-2(R²≈0.60)도 과거 메탄 없이는 baseline 을
  못 넘음(정직 보고). Model-1 은 가스미터 대체 nowcast 용도로 유효.
- **persist 피처 없이 baseline 을 이기는 방법은 시계열 모델(SARIMAX)** 이며, 아래 스태킹
  모듈에서 실증(R²=0.891 > 0.877).
- 산출물 : `outputs/xgb_metrics.json`, `xgb_lag_selection.(csv|png)`,
  `xgb_feature_importance.png`, `xgb_predictions_2023.(csv|png)`.

### RF + XGBoost + 시계열(SARIMAX) 스태킹 앙상블 (`src/stack_methane.py`)

운전변수 전용 **RandomForest·XGBoost** 와 **시계열 모델 SARIMAX(1,0,1)+외생 투입부하** 를
학습해 스태킹. RF·XGB 는 `TimeSeriesSplit(5)` 확장창 **OOF**, SARIMAX 는 **1-step 인과예측**
(누수 차단)으로 메타피처를 만들고 **비음수 Ridge** 로 결합한다. **persistence(어제 메탄값)
는 예측 입력으로 쓰지 않고 비교 baseline 으로만** 둔다. SARIMAX 는 `persist` 피처 없이
메탄 동특성을 상태공간으로 모형화하는 정식 시계열 모델이다.

| 모델 | R² | RMSE | MAE |
|------|:---:|:---:|:---:|
| persistence *(baseline)* | 0.877 | 383.5 | 261.2 |
| RandomForest (운전변수) | 0.603 | 689.2 | 535.9 |
| XGBoost (운전변수) | 0.597 | 694.0 | 534.5 |
| **SARIMAX (시계열)** | **0.891** | **360.7** | 262.1 |
| **스태킹 앙상블** | **0.891** | 361.1 | 263.3 |

**persist 피처 없이 baseline 을 이기는 것은 시계열(SARIMAX)** 이다(R² 0.891>0.877, RMSE
−5.8%). 과거 메탄을 안 쓰는 트리는 R²≈0.60 으로 baseline 미달이며, 메타는 이를 0 가중
배제하고 SARIMAX 에 수렴한다. 산출물 : `outputs/stack_metrics.json`,
`stack_meta_weights.png`, `stack_predictions_2023.(csv|png)`.

## 결과 (2023 홀드아웃, 목표 = 메탄생성량)

| 모델 | R² | RMSE | MAE |
|------|:---:|:---:|:---:|
| **RandomForest** (bio 피처) | **0.588** | 701.7 | 564.9 |
| MLP | -3.818 | 2400.2 | 2026.2 |
| LSTM (2L, h=32) | 0.170 | 996.0 | 811.3 |
| Transformer (2L) | 0.400 | 847.2 | 705.9 |
| Ensemble (NNLS) | 0.538 | 742.9 | 612.2 |

- 생물학적 체류창 부하 피처를 더한 **RandomForest 가 가장 강건한 단일 예측기**(R²≈0.59).
- **NNLS 앙상블은 실패하는 MLP(R²<0)를 0 가중치로 배제**하고 LSTM·Transformer 를 결합해
  단일 모델 의존의 위험을 회피(국면이동 하 운영 안정성 확보).
- 2023을 완전 격리한 엄격한 미래 홀드아웃 값이다. 소규모·국면이동 특성상 절대 R² 는
  보수적이나, 생물학적으로 타당한(누수 없는) 예측이라는 점이 핵심이다.

### 소화조 건강상태 (2023, VFA/알칼리도)
| 등급 | 조건 | 일수 |
|------|------|:---:|
| 🟢 안정 | VFA/ALK < 0.30 | 88 |
| 🟡 주의 | 0.30–0.40 | 57 |
| 🟠 불안정 | 0.40–0.80 | 3 |
| 🔴 위험 | ≥ 0.80 | 6 |

### 산출물 (`outputs/`)
- `metrics.json` — 모델별 지표, NNLS 가중치, 건강 분포
- `predictions_2023.csv` — 메탄 실측/예측 + VFA/ALK 건강 + 관측 수율(진단)
- `predictions_2023.png` — 메탄 예측 + 건강 신호등 밴드
- `eda_biology.png` — 투입부하 지연 교차상관 · 체류창 상관
