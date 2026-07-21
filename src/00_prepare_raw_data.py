"""
[SETUP] CONFIG 입력 데이터 생성
================================================================================
PROMPT 1 CONFIG 가 기대하는 원시 입력을 `outputs/anaerobic_digester_NGS_RA.xlsx`
(앞 단계에서 생성한 예시 NGS RA 표)로부터 만든다.

산출:
  data/raw/ra_table.csv   행=시료(40), 열=taxa(100 species + unidentified), 값=RA(%)
  data/raw/metadata.csv   sample_id, site, season, season_ko, methane(TARGET)
  data/raw/taxonomy.csv   taxon -> domain/phylum/family/genus/species
  config/domain_taxa.yaml 혐기성소화 기능 길드(genus 단위) 정의

TARGET (실측) — targets.xlsx, 2022년
--------------------------------------------------------------------------------
  name : methane   (메탄생성량; targets.xlsx 'methane' 컬럼)
  task : regression
  사용자 지시: 2022년 3/6/9/12월 1일의 methane 값을 봄/여름/가을/겨울 타깃으로 사용.
  ⚠ 가정: 메탄생성량은 원래 지역별로 다르나(현재 미준비) '모든 지역 동일'로 가정.
     → 타깃은 계절(날짜)에만 의존, 10개 지역 값 동일(예측 시뮬레이션).
     → 계절이 타깃을 완전 결정 → 마이크로바이옴은 계절 이상을 줄 수 없음(가정의 귀결).
  ⚠ 결측: 2022-06-01 전 컬럼 결측 → 최근접일 2022-05-31 대체(자동 아님, 명시 기록).

  ※ 특징(ra_table)에는 site/season 구조 + 검출한계 희소성을 로그공간에서 주입했다
    (그림용 xlsx 와 taxa 목록 동일, 값은 모델링용으로 구조 강화).
"""
import os, sys
# 이 repo 의 기존 src/signal.py 가 표준 signal 모듈을 가리는 것을 방지:
# 스크립트 실행 시 sys.path[0]=src 이므로 repo 루트로 교체한다.
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook

RNG = np.random.default_rng(7)
XLSX = "outputs/anaerobic_digester_NGS_RA.xlsx"
os.makedirs("data/raw", exist_ok=True)
os.makedirs("config", exist_ok=True)

SEASON_MAP = {"봄": "spring", "여름": "summer", "가을": "autumn", "겨울": "winter"}

# ---------------------------------------------------------------------------
# 1. xlsx -> long form 읽기
# ---------------------------------------------------------------------------
wb = load_workbook(XLSX, data_only=True)
ws = wb["RA_Species"]

# 헤더: row1 = site(도시, 병합), row2 = 계절. 데이터 col D(4)~ 40개
FIRST_DATA_COL, FIRST_DATA_ROW = 4, 3
UNID_ROW = 103  # unidentified 행
n_samples = 40

# site 라벨은 병합셀이라 앞 값을 forward-fill
sites, seasons = [], []
last_site = None
for k in range(n_samples):
    c = FIRST_DATA_COL + k
    s = ws.cell(1, c).value
    if s:
        last_site = s
    sites.append(last_site)
    seasons.append(ws.cell(2, c).value)

# taxa (species 전체명) + phylum/genus
taxa, phyla, genera, species_l = [], [], [], []
for r in range(FIRST_DATA_ROW, UNID_ROW):          # 100 species
    phyla.append(ws.cell(r, 1).value)
    genera.append(ws.cell(r, 2).value)
    species_l.append(ws.cell(r, 3).value)
    taxa.append(ws.cell(r, 3).value)
# unidentified
taxa.append("unidentified"); phyla.append("unclassified")
genera.append("unclassified"); species_l.append("unidentified")

# RA 행렬 (시료 × taxa)
ra = np.zeros((n_samples, len(taxa)))
for j, r in enumerate(list(range(FIRST_DATA_ROW, UNID_ROW)) + [UNID_ROW]):
    for k in range(n_samples):
        v = ws.cell(r, FIRST_DATA_COL + k).value
        ra[k, j] = float(v) if v is not None else 0.0

# ---------------------------------------------------------------------------
# 2. sample_id / metadata
# ---------------------------------------------------------------------------
season_en = [SEASON_MAP[s] for s in seasons]
sample_ids = [f"{site}_{se}" for site, se in zip(sites, season_en)]

ra_df = pd.DataFrame(ra, index=sample_ids, columns=taxa)
ra_df.index.name = "sample_id"

# ---------------------------------------------------------------------------
# 1b. 모델링 테이블에 site/season 조성 구조 주입 (반복측정 구조)
# ---------------------------------------------------------------------------
#   근거: 실제 소화조 마이크로바이옴은 '시설(site)'이 최대 변동 요인인 경우가 많다.
#   같은 site 의 4계절이 공통 site 시그니처를 공유 → 반복측정 구조.
#   이 구조 때문에 무작위 K-fold 는 낙관 편향되고 site 그룹 CV 가 필수가 된다.
#   ⚠ 따라서 data/raw/ra_table.csv 는 (그림용 xlsx 조성)에 site/season/잡음 구조를
#     로그공간에서 주입한 '모델링 데이터셋'이다. taxa 목록·taxonomy 는 동일하다.
n_taxa = ra_df.shape[1]
site_sig = {s: RNG.normal(0, 0.55, n_taxa) for s in dict.fromkeys(sites)}   # site 시그니처
season_sig = {se: RNG.normal(0, 0.30, n_taxa) for se in set(season_en)}     # 계절 시그니처

base = np.log(ra_df.values / 100.0 + 1e-4)
logP = base.copy()
for i, (site, se) in enumerate(zip(sites, season_en)):
    logP[i] += site_sig[site] + season_sig[se] + RNG.normal(0, 0.15, n_taxa)
# softmax(행별) → 조성 폐쇄, *100 (percent)
logP -= logP.max(axis=1, keepdims=True)
P = np.exp(logP); P /= P.sum(axis=1, keepdims=True)
# 검출한계 미만은 0 으로(현실적 희소성 복원). 신호 taxa 는 상향되어 대부분 유지됨.
P[P < 8e-4] = 0.0
P /= P.sum(axis=1, keepdims=True)
ra_df = pd.DataFrame(P * 100.0, index=sample_ids, columns=taxa)
ra_df.index.name = "sample_id"

# ---------------------------------------------------------------------------
# 3. 실측 TARGET (methane, 메탄생성량) — targets.xlsx, 2022년
# ---------------------------------------------------------------------------
#   사용자 지시: 2022년 3/6/9/12월 1일의 methane 값을 계절 타깃으로 사용.
#   ⚠ 가정: 메탄생성량은 지역별로 다르지만(현재 미준비) '모든 지역에서 동일'하다고 가정한다.
#     → 타깃은 계절(날짜)에만 의존하며 10개 지역 모두 같은 값(예측 시뮬레이션).
#   ⚠ 결측 처리(자동 대체 아님, 명시적 기록):
#     · 2022-06-01 은 전 컬럼 결측 → 가장 가까운 관측일 2022-05-31 값 사용.
#     · 2022-03-01 은 MY/VS_in 결측이나 methane 은 존재 → methane 사용.
SEASON_TARGET = {          # 메탄생성량(methane), 지역 공통
    "spring": 5486.9115,   # 2022-03-01
    "summer": 7458.858,    # 2022-05-31 (6-01 결측 대체, |Δ|=1일)
    "autumn": 6408.16,     # 2022-09-01
    "winter": 6798.644,    # 2022-12-01
}
TARGET_PROVENANCE = {
    "spring": "2022-03-01", "summer": "2022-05-31(6-01결측 대체)",
    "autumn": "2022-09-01", "winter": "2022-12-01",
}
y = np.array([SEASON_TARGET[s] for s in season_en])

meta = pd.DataFrame({
    "sample_id": sample_ids,
    "site": sites,
    "season": season_en,
    "season_ko": seasons,
    "methane": y,
}).set_index("sample_id")

print("=== 실측 TARGET (methane, 메탄생성량; 지역 공통 가정) ===")
for s in ["spring", "summer", "autumn", "winter"]:
    print(f"  {s:7s} {TARGET_PROVENANCE[s]:26s} methane={SEASON_TARGET[s]:.2f}")
print(f"  target methane  mean={y.mean():.1f}  sd={y.std():.1f}  range=[{y.min():.1f},{y.max():.1f}]")
print("  ⚠ 타깃은 계절에만 의존(지역 공통) → 계절이 타깃을 완전 결정. "
      "마이크로바이옴은 계절을 넘어선 정보를 줄 수 없음(가정의 직접 귀결).")

# ---------------------------------------------------------------------------
# 4. taxonomy.csv  (domain/phylum/family/genus/species)
# ---------------------------------------------------------------------------
GENUS_FAMILY = {
    "Methanosaeta": "Methanotrichaceae", "Methanosarcina": "Methanosarcinaceae",
    "Methanoculleus": "Methanoculleaceae", "Methanobacterium": "Methanobacteriaceae",
    "Methanobrevibacter": "Methanobacteriaceae", "Methanospirillum": "Methanospirillaceae",
    "Methanolinea": "Methanoregulaceae", "Methanomassiliicoccus": "Methanomassiliicoccaceae",
    "Methanoregula": "Methanoregulaceae", "Methanocorpusculum": "Methanocorpusculaceae",
    "Methanothermobacter": "Methanothermobacteraceae",
    "Clostridium": "Clostridiaceae", "Syntrophomonas": "Syntrophomonadaceae",
    "Ruminococcus": "Oscillospiraceae", "Christensenella": "Christensenellaceae",
    "Caldicoprobacter": "Caldicoprobacteraceae", "Sedimentibacter": "Sedimentibacteraceae",
    "Acetobacterium": "Eubacteriaceae", "Syntrophaceticus": "Thermoanaerobacteraceae",
    "Tepidanaerobacter": "Thermoanaerobacteraceae", "Gelria": "Thermoanaerobacteraceae",
    "Pelotomaculum": "Pelotomaculaceae", "Lactobacillus": "Lactobacillaceae",
    "Streptococcus": "Streptococcaceae", "Bacillus": "Bacillaceae",
    "Turicibacter": "Turicibacteraceae", "Romboutsia": "Peptostreptococcaceae",
    "Anaerovorax": "Anaerovoracaceae", "Proteiniclasticum": "Lachnospiraceae",
    "Fastidiosipila": "Ruminococcaceae", "Garciella": "Clostridiaceae",
    "Acetomicrobium": "Synergistaceae", "Anaerobaculum": "Synergistaceae",
    "Aminobacter": "unclassified", "Thermoanaerobacter": "Thermoanaerobacteraceae",
    "Anaerofilum": "Oscillospiraceae",
    "Bacteroides": "Bacteroidaceae", "Prevotella": "Prevotellaceae",
    "Proteiniphilum": "Dysgonomonadaceae", "Petrimonas": "Dysgonomonadaceae",
    "Macellibacteroides": "Dysgonomonadaceae", "Paludibacter": "Paludibacteraceae",
    "Alistipes": "Rikenellaceae", "Dysgonomonas": "Dysgonomonadaceae",
    "Fermentimonas": "Dysgonomonadaceae", "Tannerella": "Tannerellaceae",
    "Anaerolinea": "Anaerolineaceae", "Longilinea": "Anaerolineaceae",
    "Levilinea": "Anaerolineaceae", "Bellilinea": "Anaerolineaceae",
    "Leptolinea": "Anaerolineaceae", "Flexilinea": "Anaerolineaceae",
    "Ornatilinea": "Anaerolineaceae",
    "Syntrophobacter": "Syntrophobacteraceae", "Smithella": "Syntrophaceae",
    "Syntrophorhabdus": "Syntrophorhabdaceae", "Pseudomonas": "Pseudomonadaceae",
    "Desulfovibrio": "Desulfovibrionaceae", "Desulfobulbus": "Desulfobulbaceae",
    "Geobacter": "Geobacteraceae", "Thauera": "Rhodocyclaceae",
    "Aeromonas": "Aeromonadaceae", "Acinetobacter": "Moraxellaceae",
    "Synergistes": "Synergistaceae", "Aminobacterium": "Synergistaceae",
    "Cloacibacillus": "Synergistaceae", "Thermovirga": "Synergistaceae",
    "Aminomonas": "Synergistaceae",
    "Defluviitoga": "Petrotogaceae", "Kosmotoga": "Kosmotogaceae",
    "Petrotoga": "Petrotogaceae", "Mesotoga": "Kosmotogaceae",
    "Treponema": "Spirochaetaceae", "Sphaerochaeta": "Spirochaetaceae",
    "Spirochaeta": "Spirochaetaceae",
    "Actinomyces": "Actinomycetaceae", "Corynebacterium": "Corynebacteriaceae",
    "Bifidobacterium": "Bifidobacteriaceae", "Gordonibacter": "Eggerthellaceae",
    "Olsenella": "Atopobiaceae",
    "Candidatus Cloacimonas": "Cloacimonadaceae",
    "Candidatus Caldatribacterium": "Atribacteraceae", "Fusobacterium": "Fusobacteriaceae",
    "Fibrobacter": "Fibrobacteraceae", "Akkermansia": "Akkermansiaceae",
    "Planctomyces": "Planctomycetaceae", "Elusimicrobium": "Elusimicrobiaceae",
    "Victivallis": "Victivallaceae", "Caldisericum": "Caldisericaceae",
    "Thermodesulfovibrio": "Nitrospiraceae", "Coprothermobacter": "Coprothermobacteraceae",
    "Denitrovibrio": "Deferribacteraceae", "Armatimonas": "Armatimonadaceae",
    "Acidobacterium": "Acidobacteriaceae", "Candidatus Fermentibacter": "Fermentibacteraceae",
}
ARCHAEA_PHYLA = {"Euryarchaeota"}
tax_rows = []
for t, ph, ge, sp in zip(taxa, phyla, genera, species_l):
    if t == "unidentified":
        tax_rows.append(dict(taxon=t, domain="unclassified", phylum="unclassified",
                             family="unclassified", genus="unclassified", species="unidentified"))
        continue
    fam = GENUS_FAMILY.get(ge, "unclassified")
    if fam == "unclassified":
        fam = f"f__unclassified_{ph}"
    dom = "Archaea" if ph in ARCHAEA_PHYLA else "Bacteria"
    tax_rows.append(dict(taxon=t, domain=dom, phylum=ph, family=fam, genus=ge, species=sp))
tax_df = pd.DataFrame(tax_rows).set_index("taxon")

# ---------------------------------------------------------------------------
# 5. domain_taxa.yaml — 혐기성소화 기능 길드(genus 단위)
# ---------------------------------------------------------------------------
domain_taxa = {
    "acetoclastic_methanogens": {
        "level": "genus",
        "members": ["Methanosaeta", "Methanosarcina"],
    },
    "hydrogenotrophic_methanogens": {
        "level": "genus",
        "members": ["Methanoculleus", "Methanobacterium", "Methanobrevibacter",
                    "Methanospirillum", "Methanolinea", "Methanoregula",
                    "Methanocorpusculum", "Methanothermobacter"],
    },
    "methylotrophic_methanogens": {
        "level": "genus",
        "members": ["Methanomassiliicoccus"],
    },
    "syntrophic_bacteria": {
        "level": "genus",
        "members": ["Syntrophomonas", "Syntrophobacter", "Smithella",
                    "Syntrophaceticus", "Syntrophorhabdus", "Pelotomaculum",
                    "Candidatus Cloacimonas"],
    },
    "hydrolytic_acidogenic_bacteria": {
        "level": "genus",
        "members": ["Clostridium", "Bacteroides", "Prevotella", "Ruminococcus",
                    "Christensenella", "Fibrobacter", "Caldicoprobacter",
                    "Proteiniclasticum", "Anaerolinea"],
    },
    "vfa_oxidizing_bacteria": {
        "level": "genus",
        "members": ["Syntrophaceticus", "Tepidanaerobacter", "Syntrophomonas",
                    "Smithella", "Syntrophobacter"],
    },
    "ratios": [
        {"name": "logratio_aceto_hydrogeno_methanogen",
         "numerator": "acetoclastic_methanogens",
         "denominator": "hydrogenotrophic_methanogens"},
    ],
}

# ---------------------------------------------------------------------------
# 6. 저장
# ---------------------------------------------------------------------------
ra_df.round(4).to_csv("data/raw/ra_table.csv")
meta.to_csv("data/raw/metadata.csv")
tax_df.to_csv("data/raw/taxonomy.csv")
with open("config/domain_taxa.yaml", "w") as f:
    yaml.safe_dump(domain_taxa, f, allow_unicode=True, sort_keys=False)

# 매칭 실패(도메인 taxa) 보고
all_guild_genera = set()
for g, spec in domain_taxa.items():
    if g == "ratios":
        continue
    all_guild_genera.update(spec["members"])
present_genera = set(genera)
missing = sorted(all_guild_genera - present_genera)
print("\n=== domain_taxa.yaml 길드 genus 중 데이터에 없는 것 ===")
print("  ", missing if missing else "(없음 — 전부 매칭)")

print("\n저장 완료:")
print("  data/raw/ra_table.csv   ", ra_df.shape)
print("  data/raw/metadata.csv   ", meta.shape)
print("  data/raw/taxonomy.csv   ", tax_df.shape)
print("  config/domain_taxa.yaml ")
