"""
반입 → 혐기소화조 소요시간, 문헌 기반 추정
================================================================================
공정흐름도(영천 BGP)에서 읽은 경로 — 기질마다 전단 경로가 다르다.

  음식물류폐기물 : 수거차 → 저장호퍼 → 파봉파쇄기 → 종합처리기
                   → 미세비중물제거기 → 여액분리기 → 여액저장조 → 유기산발효조 → 소화조
  음폐수         : 수거차 → 스트레이너 → 여액저장조 → 유기산발효조 → 소화조
  가축분뇨       : 수거차 → 협잡물종합처리기 → 여액저장조 → 유기산발효조 → 소화조

핵심: 음식물류폐기물만 저장호퍼(2~3일)를 거친다. 나머지 둘은 우회한다.

물질수지에서 부피/반입량으로 SRT를 구하는 방식은 사용하지 않는다.
모든 체류시간은 문헌·설계지침의 단위공정별 값이며, 각 조를 완전혼합(CSTR)으로
보아 지수 RTD 를 합성곱한다.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from gpu_backend import BK

DT, NT = 0.25, 4000          # 0.25일 격자, 1000일 지평
t = np.arange(NT) * DT

# --------------------------------------------------------------------------
# 문헌 근거 (에이전트 웹조사로 실제 취득한 출처만 기재)
# --------------------------------------------------------------------------
REFS = {
    "이동진2015": {"저자": "이동진·강준구·이수영·김기헌·배지수 (2015)",
                 "제목": "음식물류폐기물의 고효율 바이오가스화를 위한 설계 및 운전 기술지침 마련 연구(Ⅲ)",
                 "출처": "유기물자원화 23(3):11–22",
                 "url": "https://koreascience.kr/article/JAKO201534851987510.pdf"},
    "이동진2017": {"저자": "이동진·문희성·손지환·배지수 (2017)",
                 "제목": "가축분뇨 병합처리 바이오가스화를 위한 설계 및 운전 기술지침 마련 연구(III)",
                 "출처": "유기물자원화 25(3):99–111",
                 "url": "https://koreascience.kr/article/JAKO201730049612435.pdf"},
    "SL공사2021": {"저자": "수도권매립지관리공사 자원관리처 음폐수시설부 (2021.3)",
                 "제목": "수도권광역 음폐수 바이오가스화시설 운영·관리기술 지침서",
                 "출처": "산발효조 813㎥×2조, 체류 3일, 35~38℃, pH 3.5~4",
                 "url": "https://www.slc.or.kr/upload/pdf/5.%EC%9D%8C%ED%8F%90%EC%88%98%20%EB%B0%94%EC%9D%B4%EC%98%A4%EA%B0%80%EC%8A%A4%ED%99%94%20%EC%8B%9C%EC%84%A4%20%EC%9A%B4%EC%98%81%EA%B4%80%EB%A6%AC%20%EA%B8%B0%EC%88%A0.pdf"},
    "이은영2012": {"저자": "이은영·전덕우·이상화·배재호·김정환·김영오 (2012)",
                 "제목": "음폐수 혐기성 소화 — 완전혼합형 산발효조(TAR) 4일",
                 "출처": "상하수도학회지 26(1)",
                 "url": "https://koreascience.kr/article/JAKO201219074277204.pdf"},
    "Menzel2020": {"저자": "Menzel, Neubauer & Junne (2020)",
                   "제목": "Role of Microbial Hydrolysis in Anaerobic Digestion",
                   "출처": "Energies 13(21):5555 — 가수분해조 CSTR, HRT 2~3일 (음식물 1일 내)",
                   "url": "https://www.mdpi.com/1996-1073/13/21/5555"},
    "EPA2006": {"저자": "US EPA (2006)",
                "제목": "Multi-Stage Anaerobic Digestion, EPA 832-F-06-031",
                "출처": "전규모 산발효조 ~1일(Waterloo IA), 2.5~3.5일(Inland Empire CA)",
                "url": "https://www.epa.gov/sites/default/files/2018-11/documents/multistage-anaerobic-digestion-factsheet.pdf"},
    "Kabaivanova2025": {"저자": "Kabaivanova et al. (2025)",
                        "제목": "Two-stage AD review — 산발효 1~3일 일반 설계범위",
                        "출처": "Processes 13(2):294",
                        "url": "https://www.mdpi.com/2227-9717/13/2/294"},
    "Demirel2002": {"저자": "Demirel & Yenigün (2002)",
                    "제목": "Two-phase anaerobic digestion processes: a review",
                    "출처": "J Chem Technol Biotechnol — 산발효 4~48시간, 대부분 CSTR",
                    "url": "https://www.academia.edu/28861802/Two_phase_anaerobic_digestion_processes_a_review"},
    "Dareioti2021": {"저자": "Dareioti, Vavouraki, Tsigkou & Kornaros (2021)",
                     "제목": "액상 우분 2상 소화 — 산발효 3일 CSTR",
                     "출처": "Energies 14(17):5423",
                     "url": "https://www.mdpi.com/1996-1073/14/17/5423"},
    "Sillero2024": {"저자": "Sillero, Pérez & Solera (2024)",
                    "제목": "하수슬러지+와인부산물+계분 — 산발효 최적 5일(고온)",
                    "출처": "Fuel 365:131186",
                    "url": "https://rodin.uca.es/bitstream/handle/10498/32689/OA_2024_0371.pdf?sequence=1"},
    "Lin2021": {"저자": "Lin et al. (2021)",
                "제목": "돈분 산발효 최적 1.5일 (35℃)",
                "출처": "Processes 9(8):1324",
                "url": "https://www.mdpi.com/2227-9717/9/8/1324"},
    "Cavinato2011": {"저자": "Cavinato, Bolzonella, Fatone, Cecchi & Pavan (2011)",
                     "제목": "분리배출 유기성폐기물 2상 — 산발효 3.3일 (55℃)",
                     "출처": "Bioresour Technol 102(18):8605–8611",
                     "url": "https://valorgas.soton.ac.uk/Pub_docs/Cavinato%20et%20al%202011%20Biores%20Tech.pdf"},
    "BioCycle": {"저자": "BioCycle — Food Depackaging: The Systems",
                 "제목": "파봉·펄퍼형 전처리 배치 10~12분 / 연속식 2~40 t/hr",
                 "출처": "BioCycle",
                 "url": "https://www.biocycle.net/food-depackaging-the-systems/"},
    "Toson2019": {"저자": "Toson, Doshi & Jajcevic (2019)",
                  "제목": "일반화 n-CSTR RTD — n<1 단회로, n=1 CSTR, n>1 직렬조",
                  "출처": "Processes 7(9):615",
                  "url": "https://www.mdpi.com/2227-9717/7/9/615"},
}

# --------------------------------------------------------------------------
# 단계별 체류시간 (일) — [최소, 표준, 최대], 모델, 근거
# --------------------------------------------------------------------------
STAGES = {
    "저장호퍼": {"tau": [2.0, 2.5, 3.0], "model": "CSTR*",
              "근거": ["이동진2015"],
              "설명": "지침: 최소 2일, 권장 3일 이상 저장용량. 그랩크레인 인출이라 "
                    "완전혼합도 FIFO도 아니다 — CSTR을 보수적 기준으로 쓰고 별도 표기."},
    "기계전처리": {"tau": [0.007, 0.015, 0.03], "model": "지연",
               "근거": ["BioCycle"],
               "설명": "파봉파쇄기·종합처리기·미세비중물제거기·여액분리기 합계 "
                     "10~30분. 총 소요의 0.5% 미만 — 순수 지연으로 처리."},
    "스트레이너": {"tau": [0.002, 0.005, 0.01], "model": "지연",
               "근거": [],
               "설명": "음폐수 전용. 초 단위 — 사실상 무시 가능."},
    "협잡물종합처리기": {"tau": [0.005, 0.01, 0.02], "model": "지연",
                  "근거": [],
                  "설명": "가축분뇨 전용. 분 단위."},
    "여액저장조": {"tau": [2.0, 2.5, 3.0], "model": "CSTR",
               "근거": ["이동진2015", "이동진2017", "SL공사2021"],
               "설명": "지침: 「완전혼합을 바탕으로 체류시간 최소 2일, 권장 3일, 2조 이상」. "
                     "병합지침 중간저장조 3일 이상. SL공사 실적 2일."},
    "유기산발효조": {"tau": [2.0, 3.0, 4.0], "model": "CSTR",
                "근거": ["SL공사2021", "Menzel2020", "Kabaivanova2025",
                       "EPA2006", "Dareioti2021", "이은영2012", "Demirel2002",
                       "Cavinato2011", "Lin2021", "Sillero2024"],
                "설명": "국내 전규모 실적 3일(SL공사). 국제문헌 합의 2~3일, "
                      "분뇨·섬유질 우세시 4~5일. 저부하 10~25일 사례는 "
                      "상분리 목적이 소실된 조건이라 설계근거에서 제외."},
}

PATHS = {
    "음식물": ["저장호퍼", "기계전처리", "여액저장조", "유기산발효조"],
    "음폐수": ["스트레이너", "여액저장조", "유기산발효조"],
    "가축분뇨": ["협잡물종합처리기", "여액저장조", "유기산발효조"],
}
SCEN = {"최소": 0, "표준": 1, "최대": 2}


# --------------------------------------------------------------------------
# RTD 합성
# --------------------------------------------------------------------------
def cstr_pdf(tau):
    e = np.exp(-t / tau) / tau
    return e / e.sum()


def delta_shift(tau):
    k = np.zeros(NT)
    i = int(round(tau / DT))
    k[min(i, NT - 1)] = 1.0
    return k


def conv_norm(a, b):
    c = np.convolve(a, b)[:NT]
    return c / c.sum()


def path_kernel(sub, scen):
    g = np.zeros(NT); g[0] = 1.0
    for st in PATHS[sub]:
        tau = STAGES[st]["tau"][SCEN[scen]]
        g = conv_norm(g, cstr_pdf(tau) if STAGES[st]["model"].startswith("CSTR")
                      else delta_shift(tau))
    return g


def kstats(g):
    g = g / g.sum(); cdf = np.cumsum(g)
    q = lambda p: float(np.interp(p, cdf, t))
    return {"평균": float((t * g).sum()), "t10": q(.10), "t50": q(.50),
            "t80": q(.80), "t90": q(.90), "t95": q(.95), "t99": q(.99)}


R = {"backend": BK.info(), "참고문헌": REFS,
     "단계": {k: {"체류일_최소표준최대": v["tau"], "모델": v["model"],
                "근거": v["근거"], "설명": v["설명"]} for k, v in STAGES.items()},
     "경로": PATHS}

R["기질별_RTD"] = {}
for sub in PATHS:
    R["기질별_RTD"][sub] = {sc: {k: round(v, 2) for k, v in kstats(path_kernel(sub, sc)).items()}
                          for sc in SCEN}

# 단계별 기여 (표준 시나리오)
R["단계별_기여_표준"] = {}
for sub in PATHS:
    acc = {}
    for st in PATHS[sub]:
        acc[st] = STAGES[st]["tau"][1]
    acc["합계_평균일"] = round(sum(acc.values()), 2)
    R["단계별_기여_표준"][sub] = acc

# 투입량 가중 통합 커널 (표준)
d01 = pd.read_csv("3c6b62de-01_________.csv", parse_dates=["날짜"])
share = {s: float(d01[f"투입톤_{s}_t"].fillna(0).mean()) for s in PATHS}
tot = sum(share.values())
share = {s: v / tot for s, v in share.items()}
R["투입비중"] = {s: round(v, 4) for s, v in share.items()}
for sc in SCEN:
    g = sum(share[s] * path_kernel(s, sc) for s in PATHS)
    R.setdefault("통합_RTD", {})[sc] = {k: round(v, 2) for k, v in kstats(g).items()}

g_std = sum(share[s] * path_kernel(s, "표준") for s in PATHS)
R["통합커널_표준_0to30d"] = [round(float(x), 6) for x in
                        (g_std / g_std.sum())[:int(30 / DT):int(1 / DT)]]

# 기존 모델(τ₀=0, τ_mix=3일 단일 커널)과 비교
old = cstr_pdf(3.0)
R["기존가정_비교"] = {
    "기존": {"설명": "τ₀=0, τ_mix=3일 단일 지수커널 (전 기질 공통)",
           **{k: round(v, 2) for k, v in kstats(old).items()}},
    "문헌표준": {"설명": "기질별 경로 반영, 단계별 문헌값 합성",
             **{k: round(v, 2) for k, v in kstats(g_std).items()}},
    "평균_차이_일": round(kstats(g_std)["평균"] - kstats(old)["평균"], 2),
    "배수": round(kstats(g_std)["평균"] / kstats(old)["평균"], 2),
}

# 기질별 커널 (그래프용, 0~30일 일단위)
R["커널_일단위"] = {"tau": list(range(31))}
for sub in PATHS:
    for sc in SCEN:
        g = path_kernel(sub, sc)
        R["커널_일단위"][f"{sub}_{sc}"] = [round(float(x), 6)
                                        for x in (g / g.sum())[:int(31 / DT):int(1 / DT)]]
R["커널_일단위"]["기존_τmix3"] = [round(float(x), 6)
                            for x in (old / old.sum())[:int(31 / DT):int(1 / DT)]]

with open("results_frontend.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)

print("=== 반입 → 소화조 유입 소요시간 (문헌 기반) ===")
for sub in PATHS:
    print(f"\n[{sub}]  경로: {' → '.join(PATHS[sub])}")
    for sc in SCEN:
        v = R["기질별_RTD"][sub][sc]
        print(f"   {sc:3s}  평균 {v['평균']:5.2f}일  t10 {v['t10']:5.2f}  t50 {v['t50']:5.2f}  "
              f"t90 {v['t90']:6.2f}  t95 {v['t95']:6.2f}")
print("\n=== 투입량 가중 통합 ===")
for sc in SCEN:
    v = R["통합_RTD"][sc]
    print(f"   {sc:3s}  평균 {v['평균']:5.2f}일  t50 {v['t50']:5.2f}  t90 {v['t90']:6.2f}  t95 {v['t95']:6.2f}")
print("\n=== 기존 가정과 비교 ===")
print(json.dumps(R["기존가정_비교"], ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------
# 실증 검증 — 문헌 커널이 실제 메탄 예측을 개선하는가
#   rolling-origin CV (90일 앞 예측) + 폴드별 대응 t검정
# --------------------------------------------------------------------------
from scipy.optimize import lsq_linear
from scipy import stats as _st

K_HYD = {"음폐수": 0.40, "가축분뇨": 0.08, "음식물": 0.30}
SRT_FIX = 25.0
NV = 600
ch4 = d01["실측CH4_m3d"].to_numpy(float)
NN = len(ch4)
recv = {s: (d01["반입_총량_tpd"].fillna(0) * d01[f"반입비율_{s}"].fillna(0)).to_numpy(float)
        for s in PATHS}
FOLDS = [(np.arange(0, st), np.arange(st, min(st + 90, NN)))
         for st in range(365, NN - 90, 90)]


def _cstr(tau, n=NV):
    e = np.exp(-np.arange(n) / tau); return e / e.sum()


def _conv(a, b, n=NV):
    c = np.convolve(a, b)[:n]; return c / c.sum()


def _h(srt, k, n=NV):
    return k * np.exp(-(1.0 / srt + k) * np.arange(n, dtype=float))


def _front_v(sub, scen):
    if scen == "기존":
        return _cstr(3.0)
    i = SCEN[scen]
    g = np.zeros(NV); g[0] = 1
    if sub == "음식물":
        g = _conv(g, _cstr(STAGES["저장호퍼"]["tau"][i]))
    g = _conv(g, _cstr(STAGES["여액저장조"]["tau"][i]))
    return _conv(g, _cstr(STAGES["유기산발효조"]["tau"][i]))


def _fold_rmse(W):
    out = []
    for tr, te in FOLDS:
        tr = tr[np.isfinite(ch4[tr])]; te = te[np.isfinite(ch4[te])]
        if len(tr) < 200 or len(te) < 20:
            continue
        th = lsq_linear(W[tr], ch4[tr],
                        bounds=([0] * W.shape[1], [np.inf] * W.shape[1])).x
        r = ch4[te] - W[te] @ th
        out.append(float(np.sqrt(r @ r / len(r))))
    return np.array(out)


def _W(fronts, ic=True):
    cols = [BK.conv_causal(BK.conv_causal(recv[s], fronts[s]), _h(SRT_FIX, K_HYD[s]))
            for s in PATHS]
    return np.column_stack(cols + [np.ones(NN)]) if ic else np.column_stack(cols)


scen_r = {}
for sc in ["기존", "최소", "표준", "최대"]:
    fr = {s: _front_v(s, sc) for s in PATHS}
    scen_r[sc] = {"무절편": _fold_rmse(_W(fr, False)), "절편": _fold_rmse(_W(fr, True))}

ver = {"검증": f"확장창 rolling-origin, 90일 앞 예측, 폴드 {len(scen_r['기존']['절편'])}개, SRT {SRT_FIX}일 고정",
       "시나리오별_CV_RMSE": {sc: {m: round(float(v.mean()), 1) for m, v in r.items()}
                        for sc, r in scen_r.items()}, "대응검정": []}
for sc in ["최소", "표준", "최대"]:
    for m in ["무절편", "절편"]:
        dd = scen_r[sc][m] - scen_r["기존"][m]
        _t, _p = _st.ttest_rel(scen_r[sc][m], scen_r["기존"][m])
        ver["대응검정"].append({"시나리오": sc, "사양": m,
                             "ΔRMSE": round(float(dd.mean()), 2),
                             "SE": round(float(dd.std(ddof=1) / np.sqrt(len(dd))), 2),
                             "p": round(float(_p), 4), "유의": bool(_p < 0.05)})

# 전단 평균지연 스캔 (2단 직렬 CSTR)
scan = []
for te_ in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
    g = _conv(_cstr(te_), _cstr(te_))
    a = _fold_rmse(_W({s: g for s in PATHS}))
    scan.append({"조당τ": te_, "평균지연": 2 * te_, "CV_RMSE": round(float(a.mean()), 1)})
bi = int(np.argmin([r["CV_RMSE"] for r in scan]))
lit = _fold_rmse(_W({s: _front_v(s, "표준") for s in PATHS}))
gbest = _conv(_cstr(scan[bi]["조당τ"]), _cstr(scan[bi]["조당τ"]))
bst = _fold_rmse(_W({s: gbest for s in PATHS}))
none = _fold_rmse(_W({s: (lambda a: a)(np.eye(NV)[0]) for s in PATHS}))
_t1, _p1 = _st.ttest_rel(lit, bst)
_t2, _p2 = _st.ttest_rel(lit, none)
ver["전단지연_스캔"] = scan
ver["CV최적_평균지연_일"] = scan[bi]["평균지연"]
ver["문헌표준_평균지연_일"] = round(kstats(path_kernel("음폐수", "표준"))["평균"], 2)
ver["문헌_vs_CV최적"] = {"문헌_RMSE": round(float(lit.mean()), 1),
                     "CV최적_RMSE": round(float(bst.mean()), 1),
                     "Δ": round(float(lit.mean() - bst.mean()), 2), "p": round(float(_p1), 3),
                     "판정": "구별되지 않음 — 문헌값 채택이 데이터와 모순되지 않는다"
                     if _p1 >= 0.05 else "유의한 차이"}
ver["문헌_vs_전단없음"] = {"문헌_RMSE": round(float(lit.mean()), 1),
                     "전단없음_RMSE": round(float(none.mean()), 1),
                     "Δ": round(float(lit.mean() - none.mean()), 2),
                     "p": float(f"{_p2:.2e}")}
R["실증검증"] = ver

with open("results_frontend.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("\n=== 실증 검증 ===")
print(json.dumps({k: v for k, v in ver.items() if k != "전단지연_스캔"},
                 ensure_ascii=False, indent=1))
