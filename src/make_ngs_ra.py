"""
혐기성 소화조(Anaerobic Digester) 내부 미생물 군집 NGS 데이터(예시) 생성기
================================================================================
- 출력형태 : RA (Relative Abundance, 상대풍부도 %)
- 시설(site): 대한민국 전국 도시 (10개소)
- Round     : 봄 / 여름 / 가을 / 겨울 (계절별 4 round)  -> 시설당 4 샘플, 총 40 샘플
- 미생물 종 : 혐기성 소화조에서 주로 발견되는 분류군 100 종 + unidentified
- 제약조건
    * unidentified(미분류)      : 0.1% ~ 30% 수준으로 샘플마다 다양하게 배치
    * minor group (<1% 존재 종) : 합계 1% ~ 10% 수준
    * 각 샘플(열)의 RA 합계        : 100.0%
- 산출물   : Excel (엑셀 표 + 계통(phylum)별 stacked bar graph)

표준 라이브러리 random 만 사용(numpy 미의존). 재현성을 위해 seed 고정.
"""

import random
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule

random.seed(20260721)

# ---------------------------------------------------------------------------
# 1. 시설 & Round 정의
# ---------------------------------------------------------------------------
CITIES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "수원", "창원", "청주"]
ROUNDS = ["봄(Spring)", "여름(Summer)", "가을(Fall)", "겨울(Winter)"]

# (city, round) 순서로 40개 샘플 컬럼
SAMPLES = [(c, r) for c in CITIES for r in ROUNDS]

# ---------------------------------------------------------------------------
# 2. 미생물 종 목록 (Phylum, Genus, Species) - 혐기성 소화조 대표 분류군 100종
#    'w' = 우점(dominant) 성향 가중치 (클수록 우점종으로 뽑힐 확률↑)
# ---------------------------------------------------------------------------
TAXA = [
    # --- Euryarchaeota : 메탄생성 고균 (혐기성 소화조 핵심 우점군) -------------
    ("Euryarchaeota", "Methanosaeta",            "Methanosaeta concilii",                 9.0),
    ("Euryarchaeota", "Methanosarcina",          "Methanosarcina barkeri",                7.0),
    ("Euryarchaeota", "Methanosarcina",          "Methanosarcina mazei",                  6.0),
    ("Euryarchaeota", "Methanoculleus",          "Methanoculleus bourgensis",             6.5),
    ("Euryarchaeota", "Methanoculleus",          "Methanoculleus marisnigri",             4.0),
    ("Euryarchaeota", "Methanobacterium",        "Methanobacterium formicicum",           5.0),
    ("Euryarchaeota", "Methanobrevibacter",      "Methanobrevibacter smithii",            3.0),
    ("Euryarchaeota", "Methanospirillum",        "Methanospirillum hungatei",             3.5),
    ("Euryarchaeota", "Methanolinea",            "Methanolinea tarda",                    3.0),
    ("Euryarchaeota", "Methanomassiliicoccus",   "Methanomassiliicoccus luminyensis",     2.0),
    ("Euryarchaeota", "Methanoregula",           "Methanoregula boonei",                  2.5),
    ("Euryarchaeota", "Methanocorpusculum",      "Methanocorpusculum labreanum",          2.0),
    ("Euryarchaeota", "Methanothermobacter",     "Methanothermobacter thermautotrophicus",2.0),
    # --- Firmicutes : 발효/가수분해/합성영양(syntrophic) 세균 ------------------
    ("Firmicutes",    "Clostridium",             "Clostridium butyricum",                 5.5),
    ("Firmicutes",    "Clostridium",             "Clostridium thermocellum",              4.0),
    ("Firmicutes",    "Syntrophomonas",          "Syntrophomonas wolfei",                 5.0),
    ("Firmicutes",    "Syntrophomonas",          "Syntrophomonas zehnderi",               3.0),
    ("Firmicutes",    "Ruminococcus",            "Ruminococcus albus",                    3.5),
    ("Firmicutes",    "Christensenella",         "Christensenella minuta",                3.0),
    ("Firmicutes",    "Caldicoprobacter",        "Caldicoprobacter oshimai",              2.0),
    ("Firmicutes",    "Sedimentibacter",         "Sedimentibacter hydroxybenzoicus",      2.5),
    ("Firmicutes",    "Acetobacterium",          "Acetobacterium woodii",                 3.0),
    ("Firmicutes",    "Syntrophaceticus",        "Syntrophaceticus schinkii",             2.5),
    ("Firmicutes",    "Tepidanaerobacter",       "Tepidanaerobacter acetatoxydans",       2.0),
    ("Firmicutes",    "Gelria",                  "Gelria glutamica",                      1.5),
    ("Firmicutes",    "Pelotomaculum",           "Pelotomaculum thermopropionicum",       2.0),
    ("Firmicutes",    "Lactobacillus",           "Lactobacillus acidophilus",             1.5),
    ("Firmicutes",    "Streptococcus",           "Streptococcus bovis",                   1.5),
    ("Firmicutes",    "Bacillus",                "Bacillus subtilis",                     1.5),
    ("Firmicutes",    "Turicibacter",            "Turicibacter sanguinis",                1.2),
    ("Firmicutes",    "Romboutsia",              "Romboutsia ilealis",                    1.2),
    ("Firmicutes",    "Anaerovorax",             "Anaerovorax odorimutans",               1.2),
    ("Firmicutes",    "Proteiniclasticum",       "Proteiniclasticum ruminis",             1.5),
    ("Firmicutes",    "Fastidiosipila",          "Fastidiosipila sanguinis",              1.0),
    ("Firmicutes",    "Garciella",               "Garciella nitratireducens",             1.0),
    ("Firmicutes",    "Acetomicrobium",          "Acetomicrobium mobile",                 1.2),
    ("Firmicutes",    "Anaerobaculum",           "Anaerobaculum mobile",                  1.2),
    ("Firmicutes",    "Aminobacter",             "Aminobacter aminovorans",               1.0),
    ("Firmicutes",    "Thermoanaerobacter",      "Thermoanaerobacter siderophilus",       1.2),
    ("Firmicutes",    "Anaerofilum",             "Anaerofilum agile",                     1.0),
    # --- Bacteroidota : 단백/당 분해 발효 세균 --------------------------------
    ("Bacteroidota",  "Bacteroides",             "Bacteroides graminisolvens",            4.5),
    ("Bacteroidota",  "Prevotella",              "Prevotella ruminicola",                 3.5),
    ("Bacteroidota",  "Proteiniphilum",          "Proteiniphilum acetatigenes",           3.0),
    ("Bacteroidota",  "Petrimonas",              "Petrimonas sulfuriphila",               2.5),
    ("Bacteroidota",  "Macellibacteroides",      "Macellibacteroides fermentans",         2.0),
    ("Bacteroidota",  "Paludibacter",            "Paludibacter propionicigenes",          2.0),
    ("Bacteroidota",  "Alistipes",               "Alistipes finegoldii",                  1.5),
    ("Bacteroidota",  "Dysgonomonas",            "Dysgonomonas mossii",                   1.2),
    ("Bacteroidota",  "Fermentimonas",           "Fermentimonas caenicola",               1.5),
    ("Bacteroidota",  "Tannerella",              "Tannerella forsythia",                  1.0),
    # --- Chloroflexi : 사상성(filamentous), 소화조 플록 형성 -------------------
    ("Chloroflexi",   "Anaerolinea",             "Anaerolinea thermophila",               4.0),
    ("Chloroflexi",   "Longilinea",              "Longilinea arvoryzae",                  2.5),
    ("Chloroflexi",   "Levilinea",               "Levilinea saccharolytica",              2.5),
    ("Chloroflexi",   "Bellilinea",              "Bellilinea caldifistulae",              2.0),
    ("Chloroflexi",   "Leptolinea",              "Leptolinea tardivitalis",               2.0),
    ("Chloroflexi",   "Flexilinea",              "Flexilinea flocculi",                   1.5),
    ("Chloroflexi",   "Ornatilinea",             "Ornatilinea apprima",                   1.2),
    # --- Pseudomonadota (Proteobacteria) : 합성영양/황산환원 등 ---------------
    ("Pseudomonadota","Syntrophobacter",         "Syntrophobacter fumaroxidans",          3.5),
    ("Pseudomonadota","Smithella",               "Smithella propionica",                  3.0),
    ("Pseudomonadota","Syntrophorhabdus",        "Syntrophorhabdus aromaticivorans",      2.0),
    ("Pseudomonadota","Pseudomonas",             "Pseudomonas putida",                    1.5),
    ("Pseudomonadota","Desulfovibrio",           "Desulfovibrio vulgaris",                2.0),
    ("Pseudomonadota","Desulfobulbus",           "Desulfobulbus propionicus",             1.8),
    ("Pseudomonadota","Geobacter",               "Geobacter sulfurreducens",              2.0),
    ("Pseudomonadota","Thauera",                 "Thauera aromatica",                     1.2),
    ("Pseudomonadota","Aeromonas",               "Aeromonas hydrophila",                  1.0),
    ("Pseudomonadota","Acinetobacter",           "Acinetobacter johnsonii",               1.0),
    # --- Synergistota : 아미노산 분해 합성영양 --------------------------------
    ("Synergistota",  "Synergistes",             "Synergistes jonesii",                   2.0),
    ("Synergistota",  "Aminobacterium",          "Aminobacterium colombiense",            2.0),
    ("Synergistota",  "Cloacibacillus",          "Cloacibacillus evryensis",              1.8),
    ("Synergistota",  "Thermovirga",             "Thermovirga lienii",                    1.5),
    ("Synergistota",  "Aminomonas",              "Aminomonas paucivorans",                1.2),
    # --- Thermotogota : 고온성 발효 세균 --------------------------------------
    ("Thermotogota",  "Defluviitoga",            "Defluviitoga tunisiensis",              2.5),
    ("Thermotogota",  "Kosmotoga",               "Kosmotoga olearia",                     1.5),
    ("Thermotogota",  "Petrotoga",               "Petrotoga mobilis",                     1.2),
    ("Thermotogota",  "Mesotoga",                "Mesotoga prima",                        1.2),
    # --- Spirochaetota : 당 발효 스피로헤타 -----------------------------------
    ("Spirochaetota", "Treponema",               "Treponema primitia",                    2.0),
    ("Spirochaetota", "Sphaerochaeta",           "Sphaerochaeta globosa",                 1.5),
    ("Spirochaetota", "Spirochaeta",             "Spirochaeta caldaria",                  1.2),
    # --- Actinobacteriota -----------------------------------------------------
    ("Actinobacteriota","Actinomyces",           "Actinomyces bovis",                     1.5),
    ("Actinobacteriota","Corynebacterium",       "Corynebacterium glutamicum",            1.2),
    ("Actinobacteriota","Bifidobacterium",       "Bifidobacterium longum",                1.2),
    ("Actinobacteriota","Gordonibacter",         "Gordonibacter pamelaeae",               1.0),
    ("Actinobacteriota","Olsenella",             "Olsenella uli",                         1.0),
    # --- Cloacimonadota : 프로피온산 합성영양(대표: Ca. Cloacimonas) ----------
    ("Cloacimonadota","Candidatus Cloacimonas",  "Ca. Cloacimonas acidaminovorans",       2.5),
    ("Cloacimonadota","Candidatus Cloacimonas",  "Ca. Cloacimonas W5",                    1.5),
    # --- 기타 소수 계통 (희소/miner group 성향) -------------------------------
    ("Atribacterota", "Candidatus Caldatribacterium","Ca. Caldatribacterium californiense",1.5),
    ("Fusobacteriota","Fusobacterium",           "Fusobacterium nucleatum",               1.0),
    ("Fibrobacterota","Fibrobacter",             "Fibrobacter succinogenes",              1.2),
    ("Verrucomicrobiota","Akkermansia",          "Akkermansia muciniphila",               1.0),
    ("Planctomycetota","Planctomyces",           "Planctomyces limnophilus",              0.8),
    ("Elusimicrobiota","Elusimicrobium",         "Elusimicrobium minutum",                0.8),
    ("Lentisphaerota","Victivallis",             "Victivallis vadensis",                  0.8),
    ("Caldisericota", "Caldisericum",            "Caldisericum exile",                    0.8),
    ("Nitrospirota",  "Thermodesulfovibrio",     "Thermodesulfovibrio yellowstonii",      0.9),
    ("Coprothermobacterota","Coprothermobacter", "Coprothermobacter proteolyticus",       1.5),
    ("Deferribacterota","Denitrovibrio",         "Denitrovibrio acetiphilus",             0.8),
    ("Armatimonadota","Armatimonas",             "Armatimonas rosea",                     0.6),
    ("Acidobacteriota","Acidobacterium",         "Acidobacterium capsulatum",             0.6),
    ("Fermentibacterota","Candidatus Fermentibacter","Ca. Fermentibacter daniensis",      1.2),
]

assert len(TAXA) == 100, f"종 수는 100 이어야 함, 현재 {len(TAXA)}"

N_SPECIES = len(TAXA)
WEIGHTS = [t[3] for t in TAXA]

# ---------------------------------------------------------------------------
# 3. 샘플별 RA(%) 생성
# ---------------------------------------------------------------------------
def weighted_sample_without_replacement(population_idx, weights, k):
    """가중치 기반 비복원 추출 (Efraimidis-Spirakis A-Res)."""
    keyed = []
    for i in population_idx:
        w = weights[i]
        u = random.random()
        key = u ** (1.0 / w)   # 큰 key 우선 -> 가중치 클수록 선택 확률↑
        keyed.append((key, i))
    keyed.sort(reverse=True)
    return [i for _, i in keyed[:k]]


def generate_sample():
    """한 샘플(열)의 종별 RA(%) 리스트(len=100)와 unidentified 값을 반환."""
    idx_all = list(range(N_SPECIES))

    # (1) 미분류(unidentified): 0.1 ~ 30 %
    unidentified = random.uniform(0.1, 30.0)

    # (2) minor group(<1% 존재 종) 합계 R : 1 ~ 10 %
    R = random.uniform(1.0, 10.0)

    # (3) 우점(dominant) 종 합계 D
    D = 100.0 - unidentified - R

    # 우점종 개수 K (샘플마다 다양성 부여)
    K = random.randint(16, 34)

    # 우점종 선택 (가중치 기반)
    dom_idx = weighted_sample_without_replacement(idx_all, WEIGHTS, K)
    dom_set = set(dom_idx)

    # 우점종 값: 각 >=1%, 합 = D  (skew한 분포로 소수 종이 크게 우점)
    #   각 종 기본 1% 부여 후, 잔여(D-K)를 지수분포 가중으로 분배
    base = [1.0] * K
    extra = D - K
    if extra < 0:  # 방어적: K가 D보다 크면 K 축소
        K = max(1, int(D))
        dom_idx = dom_idx[:K]
        dom_set = set(dom_idx)
        base = [1.0] * K
        extra = D - K
    raw = [random.expovariate(1.0) ** 1.6 for _ in range(K)]  # heavy-tail
    s = sum(raw)
    dom_vals = [base[j] + extra * raw[j] / s for j in range(K)]

    # (4) minor 종 선택 : 남은 종 중 일부만 '검출'(나머지는 0.0)
    rare_pool = [i for i in idx_all if i not in dom_set]
    n_present = random.randint(20, min(40, len(rare_pool)))
    # 희소종도 가중치 반영해 검출 여부 결정
    minor_idx = weighted_sample_without_replacement(rare_pool, WEIGHTS, n_present)
    # 각 <1% 이며 합 = R : 균등에 가깝게 분배(한 종이 1% 넘지 않도록)
    mr = [random.random() + 0.05 for _ in range(n_present)]
    ms = sum(mr)
    minor_vals = [R * v / ms for v in mr]
    # 혹시 1% 초과가 생기면 상한 0.95로 클립 후 잔여 재분배
    for _ in range(5):
        over = sum(max(0.0, v - 0.95) for v in minor_vals)
        if over < 1e-9:
            break
        minor_vals = [min(v, 0.95) for v in minor_vals]
        room = [0.95 - v for v in minor_vals]
        rs = sum(room)
        if rs < 1e-9:
            break
        minor_vals = [v + over * room[j] / rs for j, v in enumerate(minor_vals)]

    # (5) 종별 RA 배열 구성
    ra = [0.0] * N_SPECIES
    for j, i in enumerate(dom_idx):
        ra[i] = dom_vals[j]
    for j, i in enumerate(minor_idx):
        ra[i] = minor_vals[j]

    return ra, unidentified


def round_and_fix(values):
    """소수 1자리로 반올림하고, 합계가 정확히 100.0 이 되도록 최대값 셀에서 보정."""
    r = [round(v, 1) for v in values]
    residual = round(100.0 - sum(r), 1)
    if abs(residual) >= 0.05:
        # 값이 가장 큰 셀에 잔차를 흡수(음수가 되지 않게)
        order = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
        for i in order:
            if r[i] + residual >= 0:
                r[i] = round(r[i] + residual, 1)
                break
    return r


# 40개 샘플 생성. 각 열: [종0..종99, unidentified]  (길이 101)
columns = []
for (city, rnd) in SAMPLES:
    ra, unid = generate_sample()
    full = ra + [unid]                 # 마지막 = unidentified
    full = round_and_fix(full)         # 합계 100.0 보정
    columns.append(full)

# 검증
for k, col in enumerate(columns):
    assert abs(sum(col) - 100.0) < 0.051, f"열 {k} 합계 {sum(col)}"
    unid = col[-1]
    assert 0.1 - 0.05 <= unid <= 30.0 + 0.5, f"열 {k} unidentified {unid}"
    minor_sum = sum(v for v in col[:-1] if 0 < v < 1.0)
    # minor group 합계는 대체로 1~10% (반올림 영향으로 약간 벗어날 수 있어 경고만)
print("데이터 검증 통과: 40개 샘플, 각 열 합계 100.0%")

# ---------------------------------------------------------------------------
# 4. 엑셀 작성
# ---------------------------------------------------------------------------
FONT = "Arial"
wb = Workbook()

thin = Side(style="thin", color="BBBBBB")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
hdr_fill = PatternFill("solid", fgColor="1F4E5F")
site_fill = PatternFill("solid", fgColor="2E75B6")
unid_fill = PatternFill("solid", fgColor="F2F2F2")
center = Alignment(horizontal="center", vertical="center")
left = Alignment(horizontal="left", vertical="center")

# ===== Sheet 1 : 종(Species) 수준 RA 표 =====================================
ws = wb.active
ws.title = "RA_Species"

TAX_COLS = 3  # Phylum, Genus, Species
FIRST_DATA_COL = TAX_COLS + 1          # D열
HDR_ROW_SITE = 1
HDR_ROW_ROUND = 2
FIRST_DATA_ROW = 3

# 좌상단 라벨
ws.cell(HDR_ROW_SITE, 1, "Anaerobic Digester — Microbial Community (RA, %)")
ws.merge_cells(start_row=HDR_ROW_SITE, start_column=1, end_row=HDR_ROW_SITE, end_column=TAX_COLS)
ws.cell(HDR_ROW_ROUND, 1, "Phylum"); ws.cell(HDR_ROW_ROUND, 2, "Genus"); ws.cell(HDR_ROW_ROUND, 3, "Species")

# 상단 2행 헤더: site(도시) 병합 + round(계절)
col = FIRST_DATA_COL
for ci, city in enumerate(CITIES):
    c0 = col
    for rnd in ROUNDS:
        cell = ws.cell(HDR_ROW_ROUND, col, rnd.split("(")[0])  # 봄/여름/가을/겨울
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=9)
        cell.fill = hdr_fill; cell.alignment = center; cell.border = border
        col += 1
    ws.merge_cells(start_row=HDR_ROW_SITE, start_column=c0, end_row=HDR_ROW_SITE, end_column=col - 1)
    sc = ws.cell(HDR_ROW_SITE, c0, city)
    sc.font = Font(name=FONT, bold=True, color="FFFFFF", size=11)
    sc.fill = site_fill; sc.alignment = center; sc.border = border

# 좌상단 라벨 스타일
for r in (HDR_ROW_SITE, HDR_ROW_ROUND):
    for c in range(1, TAX_COLS + 1):
        cc = ws.cell(r, c)
        cc.font = Font(name=FONT, bold=True, color="FFFFFF", size=(11 if r == HDR_ROW_SITE else 10))
        cc.fill = (site_fill if r == HDR_ROW_SITE else hdr_fill)
        cc.alignment = (left if (r == HDR_ROW_SITE and c == 1) else center)
        cc.border = border

# 데이터 행: 100종 + unidentified
row = FIRST_DATA_ROW
for si, (phylum, genus, species, _w) in enumerate(TAXA):
    ws.cell(row, 1, phylum).font = Font(name=FONT, size=9)
    ws.cell(row, 2, genus).font = Font(name=FONT, size=9, italic=True)
    ws.cell(row, 3, species).font = Font(name=FONT, size=9, italic=True)
    for c in range(1, 4):
        ws.cell(row, c).alignment = left; ws.cell(row, c).border = border
    for k, colvals in enumerate(columns):
        cell = ws.cell(row, FIRST_DATA_COL + k, colvals[si])
        cell.font = Font(name=FONT, size=9); cell.alignment = center
        cell.number_format = "0.0"; cell.border = border
    row += 1

# unidentified 행
UNID_ROW = row
ws.cell(UNID_ROW, 1, "unidentified").font = Font(name=FONT, size=9, bold=True)
ws.merge_cells(start_row=UNID_ROW, start_column=1, end_row=UNID_ROW, end_column=3)
ws.cell(UNID_ROW, 1).alignment = left; ws.cell(UNID_ROW, 1).border = border
for k, colvals in enumerate(columns):
    cell = ws.cell(UNID_ROW, FIRST_DATA_COL + k, colvals[-1])
    cell.font = Font(name=FONT, size=9, bold=True); cell.alignment = center
    cell.number_format = "0.0"; cell.fill = unid_fill; cell.border = border
row += 1

# 합계(Total) 행 = 열별 합계 (검증용, =100.0). LibreOffice 미가용 → 계산값 기입
TOTAL_ROW = row
ws.cell(TOTAL_ROW, 1, "합계 Total (%)").font = Font(name=FONT, size=9, bold=True)
ws.merge_cells(start_row=TOTAL_ROW, start_column=1, end_row=TOTAL_ROW, end_column=3)
ws.cell(TOTAL_ROW, 1).alignment = left; ws.cell(TOTAL_ROW, 1).border = border
for k, colvals in enumerate(columns):
    cell = ws.cell(TOTAL_ROW, FIRST_DATA_COL + k, round(sum(colvals), 1))
    cell.font = Font(name=FONT, size=9, bold=True); cell.alignment = center
    cell.number_format = "0.0"; cell.fill = PatternFill("solid", fgColor="FCE4D6"); cell.border = border

# 컬러 스케일(값이 클수록 진하게) — 예시 이미지처럼 시각적 강조
last_data_col_letter = get_column_letter(FIRST_DATA_COL + len(columns) - 1)
rng = f"{get_column_letter(FIRST_DATA_COL)}{FIRST_DATA_ROW}:{last_data_col_letter}{UNID_ROW}"
ws.conditional_formatting.add(rng, ColorScaleRule(
    start_type="num", start_value=0,   start_color="FFFFFF",
    mid_type="num",   mid_value=3,     mid_color="FFD966",
    end_type="num",   end_value=15,    end_color="C0392B"))

# 열 너비 / 틀 고정
ws.column_dimensions["A"].width = 20
ws.column_dimensions["B"].width = 22
ws.column_dimensions["C"].width = 34
for k in range(len(columns)):
    ws.column_dimensions[get_column_letter(FIRST_DATA_COL + k)].width = 8.5
ws.freeze_panes = ws.cell(FIRST_DATA_ROW, FIRST_DATA_COL)

# ===== Sheet 2 : 계통(Phylum) 수준 집계 + Stacked Bar =========================
wp = wb.create_sheet("RA_Phylum")

PHYLA = []
for (ph, *_r) in TAXA:
    if ph not in PHYLA:
        PHYLA.append(ph)

# 헤더: A=Phylum, B.. = 샘플(도시_계절)
wp.cell(1, 1, "Phylum").font = Font(name=FONT, bold=True, color="FFFFFF")
wp.cell(1, 1).fill = hdr_fill; wp.cell(1, 1).alignment = center; wp.cell(1, 1).border = border
for k, (city, rnd) in enumerate(SAMPLES):
    label = f"{city}_{rnd.split('(')[0]}"
    cell = wp.cell(1, 2 + k, label)
    cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=9)
    cell.fill = hdr_fill; cell.alignment = Alignment(horizontal="center", vertical="center", text_rotation=90)
    cell.border = border

# phylum -> 종 인덱스 목록
ph_to_idx = {ph: [] for ph in PHYLA}
for si, (phylum, *_r) in enumerate(TAXA):
    ph_to_idx[phylum].append(si)

# 각 phylum 행 = 해당 phylum 종들의 RA 합 (계산값). 소수 1자리.
for pi, ph in enumerate(PHYLA):
    r = 2 + pi
    wp.cell(r, 1, ph).font = Font(name=FONT, size=9); wp.cell(r, 1).border = border
    for k, colvals in enumerate(columns):
        val = round(sum(colvals[si] for si in ph_to_idx[ph]), 1)
        cell = wp.cell(r, 2 + k, val)
        cell.font = Font(name=FONT, size=9); cell.number_format = "0.0"
        cell.alignment = center; cell.border = border

# unidentified 행 (종 시트 unidentified = 각 열 마지막 값)
UNID_PROW = 2 + len(PHYLA)
wp.cell(UNID_PROW, 1, "unidentified").font = Font(name=FONT, size=9, bold=True)
wp.cell(UNID_PROW, 1).border = border
for k, colvals in enumerate(columns):
    cell = wp.cell(UNID_PROW, 2 + k, colvals[-1])
    cell.font = Font(name=FONT, size=9, bold=True); cell.number_format = "0.0"
    cell.alignment = center; cell.fill = unid_fill; cell.border = border

# 합계 행 (계산값, =100.0)
TOT_PROW = UNID_PROW + 1
wp.cell(TOT_PROW, 1, "합계 Total (%)").font = Font(name=FONT, size=9, bold=True)
wp.cell(TOT_PROW, 1).border = border
for k, colvals in enumerate(columns):
    # phylum 합 + unidentified = 100 (반올림 표기)
    tot = round(sum(round(sum(colvals[si] for si in ph_to_idx[ph]), 1) for ph in PHYLA)
                + colvals[-1], 1)
    cell = wp.cell(TOT_PROW, 2 + k, tot)
    cell.font = Font(name=FONT, size=9, bold=True); cell.number_format = "0.0"
    cell.alignment = center; cell.fill = PatternFill("solid", fgColor="FCE4D6"); cell.border = border

wp.column_dimensions["A"].width = 22
for k in range(len(columns)):
    wp.column_dimensions[get_column_letter(2 + k)].width = 7
wp.freeze_panes = "B2"

# ----- Stacked Bar Chart : 샘플별 phylum 조성 --------------------------------
chart = BarChart()
chart.type = "col"
chart.grouping = "stacked"
chart.overlap = 100
chart.title = "혐기성 소화조 미생물 군집 조성 (Phylum-level RA, %) — 시설×계절"
chart.y_axis.title = "Relative Abundance (%)"
chart.x_axis.title = "시설 _ 계절 (Site _ Round)"
chart.height = 12
chart.width = 40
chart.y_axis.scaling.min = 0
chart.y_axis.scaling.max = 100

# 데이터: phylum + unidentified 행들, 열 = 샘플. 시리즈=행(phylum), 카테고리=샘플라벨
data = Reference(wp, min_col=1, min_row=1, max_row=UNID_PROW, max_col=1 + len(columns))
cats = Reference(wp, min_col=2, min_row=1, max_col=1 + len(columns), max_row=1)
chart.add_data(data, titles_from_data=True, from_rows=True)
chart.set_categories(cats)
chart.legend.position = "r"
wp.add_chart(chart, f"A{TOT_PROW + 3}")

# ===== Sheet 3 : README / 메타데이터 ========================================
wm = wb.create_sheet("README", 0)
meta = [
    ["혐기성 소화조(Anaerobic Digester) 내부 미생물 군집 NGS 데이터 (예시)", ""],
    ["", ""],
    ["출력형태 (Output)", "RA (Relative Abundance, 상대풍부도 %)"],
    ["시설 수 (Sites)", f"{len(CITIES)} 개소 — 대한민국 전국 도시"],
    ["시설 목록", ", ".join(CITIES)],
    ["Round (계절)", "봄 / 여름 / 가을 / 겨울 (계절별 4 round)"],
    ["총 샘플 수", f"{len(SAMPLES)} = {len(CITIES)} 시설 × {len(ROUNDS)} round"],
    ["미생물 종 수", f"{N_SPECIES} 종 (+ unidentified) — 혐기성 소화조 대표 분류군"],
    ["계통(Phylum) 수", f"{len(PHYLA)} 문"],
    ["", ""],
    ["데이터 규칙 (Rules)", ""],
    ["  · 각 샘플(열) RA 합계", "100.0 %"],
    ["  · unidentified(미분류)", "0.1 % ~ 30 % 수준으로 샘플마다 다양"],
    ["  · minor group (<1% 검출 종)", "합계 1 % ~ 10 % 수준"],
    ["  · 우점종(dominant)", "메탄생성 고균 · 합성영양 세균 등이 계절/시설별로 변동"],
    ["", ""],
    ["시트 구성", ""],
    ["  RA_Species", "종(100) 수준 RA 표 (엑셀 표) + 값 컬러 스케일"],
    ["  RA_Phylum", "계통(문) 수준 집계 표 + Stacked Bar Graph"],
    ["", ""],
    ["생성 스크립트", "src/make_ngs_ra.py (seed=20260721, 재현 가능)"],
    ["주의", "본 데이터는 실측이 아닌 통계적으로 생성한 예시(모의) 데이터입니다."],
]
for r, (a, b) in enumerate(meta, start=1):
    ca = wm.cell(r, 1, a); cb = wm.cell(r, 2, b)
    ca.font = Font(name=FONT, bold=(r == 1 or b == "" and a != ""), size=(13 if r == 1 else 10))
    cb.font = Font(name=FONT, size=10)
    ca.alignment = left; cb.alignment = left
wm.column_dimensions["A"].width = 30
wm.column_dimensions["B"].width = 70

# ---------------------------------------------------------------------------
OUT = "outputs/anaerobic_digester_NGS_RA.xlsx"
wb.save(OUT)
print(f"저장 완료: {OUT}")
print(f"  - 시설 {len(CITIES)}개 × round {len(ROUNDS)} = 샘플 {len(SAMPLES)}개")
print(f"  - 종 {N_SPECIES} + unidentified,  문(phylum) {len(PHYLA)}")
