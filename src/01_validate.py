"""STEP 1 — 무결성 검증 (PROMPT 1 §STEP 1). 결측을 자동으로 채우지 않는다(발견만)."""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from adpipe.config import load_raw, detect_ra_scale, CONFIG

d = load_raw()
X, meta = d["X"], d["meta"]
os.makedirs("reports", exist_ok=True)
L = []
def w(s=""): L.append(s)

w("# 01 · 무결성 검증 (Validation)")
w()
w("> ⚠ 본 데이터는 합성(예시) NGS RA 데이터입니다. TARGET(`ch4_yield`)은 문서화된 "
  "약한 신호로 주입된 합성 값입니다(`src/00_prepare_raw_data.py`).")
w()

# 1. RA 스케일
scale = detect_ra_scale(X)
rs = X.sum(axis=1)
w("## 1. RA 스케일 판별")
w()
w(f"- 판별 결과: **{scale}** (unit≈1 / percent≈100)")
w(f"- 행 합 요약: min={rs.min():.3f}, median={rs.median():.3f}, max={rs.max():.3f}")
if scale == "unknown":
    w("- ❌ 행 합이 1 또는 100 어느 쪽도 아님 → **부분 테이블 의심. 중단하고 사용자 확인 필요.**")
else:
    w(f"- ✅ 스케일 일관성 확인. 이후 파이프라인은 내부적으로 행합=1 비율로 정규화하여 처리.")
w()

# 2. 행 합 편차
tol = 1.0 if scale == "percent" else 0.01
target = 100.0 if scale == "percent" else 1.0
dev = (rs - target).abs()
off = dev[dev > tol]
w("## 2. 행 합 편차 / 재정규화 로그")
w()
if len(off) == 0:
    w(f"- 모든 시료의 행 합이 {target:g}±{tol:g} 이내. 재정규화 불필요(파이프라인 내부 재정규화만 수행).")
else:
    w(f"- {len(off)} 개 시료가 허용편차 초과:")
    for sid, v in off.items():
        w(f"  - `{sid}`: 합={rs[sid]:.3f}, 편차={v:.3f}")
    w("- → transform 시 행합 1 재정규화 적용(조성 폐쇄 유지). 원본은 보존.")
w()

# 3. 음수/NaN/전영 행·열
w("## 3. 음수 · NaN · 전(全)영 행/열")
w()
neg = int((X.values < 0).sum())
nan = int(X.isna().values.sum())
zero_rows = X.index[(X.sum(axis=1) == 0)].tolist()
zero_cols = X.columns[(X.sum(axis=0) == 0)].tolist()
w(f"- 음수 값: {neg} 개")
w(f"- NaN: {nan} 개")
w(f"- 전영 시료(행): {len(zero_rows)} {zero_rows if zero_rows else ''}")
w(f"- 전영 taxa(열): {len(zero_cols)} {zero_cols[:10]}{' ...' if len(zero_cols)>10 else ''}")
w("- (결측/이상은 발견만 보고하며 자동 대체하지 않음 — 사용자 판단 대상.)")
w()

# 4. 시료 ID 1:1
w("## 4. 시료 ID ↔ metadata 대응")
w()
only_ra = sorted(set(X.index) - set(meta.index))
only_meta = sorted(set(meta.index) - set(X.index))
if not only_ra and not only_meta:
    w(f"- ✅ {len(X.index)} 개 시료가 1:1 대응.")
else:
    w(f"- ❌ 불일치: RA에만 있는 시료 {only_ra}; metadata에만 있는 시료 {only_meta}")
w()

# 5. site × season 교차표
w("## 5. 설계 확인 — site × season 교차표")
w()
ct = pd.crosstab(meta["site"], meta["season"])
order = [s for s in CONFIG["SEASON_ORDER"] if s in ct.columns]
ct = ct[order]
w("| site | " + " | ".join(order) + " | 합 |")
w("|---|" + "---|" * (len(order) + 1))
for site, row in ct.iterrows():
    w(f"| {site} | " + " | ".join(str(int(v)) for v in row.values) + f" | {int(row.sum())} |")
balanced = (ct.values == 1).all() and ct.shape == (CONFIG["DESIGN"]["n_sites"], CONFIG["DESIGN"]["n_seasons"])
w()
w(f"- 설계: {CONFIG['DESIGN']['n_sites']} site × {CONFIG['DESIGN']['n_seasons']} season "
  f"→ {'✅ 완전 균형(10×4)' if balanced else '⚠ 불균형/결측 셀 존재'}")
w(f"- 독립 단위 = site 수 = **{meta['site'].nunique()}** (반복측정: 같은 site 4계절). "
  f"→ CV 는 반드시 **site 그룹 분할**.")
w()

# 7. 최소 비영값(검출한계 추정)
nz = X.values[X.values > 0]
w("## 6. 검출한계(최소 비영 RA)")
w()
w(f"- 전체 최소 비영값(RA %): {nz.min():.4f}")
w(f"- 최소 비영값(비율 환산): {nz.min()/ (100 if scale=='percent' else 1):.2e}  "
  f"→ 승법적 영대체 δ 추정 기준(학습 fold별로 재계산).")
w()

# p/n 요약
p = X.shape[1]; n = X.shape[0]
w("## 7. 규모 요약")
w()
w(f"- n(시료)={n}, 독립단위(site)={meta['site'].nunique()}, p(원 taxa)={p}, p/n={p/n:.2f}")
if p/n > 20:
    w("- 🔴 **강한 경고**: p/n>20. 더 상위 분류계급 집계를 강력 권고.")
elif p/n > 5:
    w("- 🟠 경고: p/n>5. 필터링/집계로 특징 수 축소 필요.")
w()

open("reports/01_validation.md", "w").write("\n".join(L))
print("작성: reports/01_validation.md")

# --------------------------- 02 희소성 프로파일 ------------------------------
S = []
def ws(s=""): S.append(s)
P = X.div(X.sum(axis=1), axis=0)          # 행합1
ws("# 02 · 희소성 프로파일 (Sparsity)")
ws()
zero_frac = float((P.values == 0).mean())
det = (P > 0.001)
per_sample = det.sum(axis=1)
prev = det.mean(axis=0)
ws(f"- 전체 영(0) 비율: **{zero_frac:.1%}**")
ws(f"- 시료당 검출 taxa 수(임계 0.1%): min={per_sample.min()}, median={int(per_sample.median())}, max={per_sample.max()}")
ws(f"- 최소 비영 RA(비율): {P.values[P.values>0].min():.2e}")
ws()
ws("## taxa 유병률 분포(임계 0.1%)")
ws()
bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0001]
labels = ["0–10%", "10–25%", "25–50%", "50–75%", "75–100%"]
hist = pd.cut(prev, bins=bins, labels=labels, include_lowest=True, right=False).value_counts().reindex(labels).fillna(0)
ws("| 유병률 구간 | taxa 수 |")
ws("|---|---|")
for lab in labels:
    ws(f"| {lab} | {int(hist[lab])} |")
ws()
ws(f"- 유병률 25% 미만 taxa: {int((prev<0.25).sum())} 개 (기본 필터 `min_prevalence=0.25`에서 제거 후보)")
ws(f"- 유병률 50% 이상 taxa: {int((prev>=0.5).sum())} 개 (안정적 코어)")
ws()
ws("> 판단: 희소성이 높을수록 로그비 변환의 영대체 민감도가 커진다. 따라서 필터 강도"
   "(`min_prevalence`)와 영대체 δ 는 **고정하지 않고** PROMPT 2 내부 CV 에서 튜닝한다. "
   "이 판단이 틀렸다면(예: 코어 taxa 만으로 충분) 필터를 세게 걸어 p 를 줄이는 편이 나을 수 있다.")
open("reports/02_sparsity_profile.md", "w").write("\n".join(S))
print("작성: reports/02_sparsity_profile.md")
print(f"요약: n={n}, sites={meta['site'].nunique()}, p={p}, p/n={p/n:.2f}, zero_frac={zero_frac:.1%}")
