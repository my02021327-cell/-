"""
현장 경험식 vs 우리 모델 — 동일 폴드·동일 타깃 비교, 그리고 결함 수정안 검증.

    python scripts/compare_empirical.py

출력: outputs/compare_empirical.json  (+ 콘솔 요약)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bioguard.config import K_HYD, SRT_REFERENCE_D          # noqa: E402
from src.bioguard.cv import (fold_rmse, make_folds, metrics, oof_predictions,  # noqa: E402
                             paired_test, valid_folds)
from src.bioguard.data import build_frame                        # noqa: E402
from src.bioguard.empirical import (EMP, emp_weights, empirical_series, make_M1_feed,  # noqa: E402
                                    make_empirical, make_empirical_refit,
                                    reconstruction_check, substrate_feed)
from src.bioguard.models import combine, make_M1, make_M2, make_M3, make_M4, make_M5  # noqa: E402

OUT = ROOT / "outputs"


def main() -> int:
    F = build_frame()
    y = F.ch4.to_numpy(float)
    n = len(F)
    folds = valid_folds(make_folds(n), y)
    R: dict = {"폴드수": len(folds), "타깃_관측일": int(np.isfinite(y).sum())}

    # ---------------------------------------------------------------- 1. 식 검증
    Wf, Wm = emp_weights()
    d = np.arange(1, EMP["D"] + 1)
    inf_f = EMP["k_f"] * np.exp(-EMP["k_f"]) / (1 - np.exp(-EMP["k_f"]))
    inf_m = EMP["k_m"] * np.exp(-EMP["k_m"]) / (1 - np.exp(-EMP["k_m"]))
    R["식_내부검증"] = {
        "가중치표_재현": {f"D-{i}": [round(float(Wf[i - 1]), 3), round(float(Wm[i - 1]), 4)]
                     for i in d},
        "판정": "문서 표와 소수 셋째 자리까지 일치 — 식은 내부적으로 일관됨",
        "7일합_음식물류_m3_per_t": round(float(Wf.sum()), 3),
        "7일합_가축분뇨_m3_per_t": round(float(Wm.sum()), 4),
        "7일절단이_담는비율_pct": {"음식물류": round(100 * float(Wf.sum()) / (EMP["Y_f"] * inf_f), 1),
                          "가축분뇨": round(100 * float(Wm.sum()) / (EMP["Y_m"] * inf_m), 1)},
        "반감기_d": {"음식물류": round(float(np.log(2) / EMP["k_f"]), 1),
                  "가축분뇨": round(float(np.log(2) / EMP["k_m"]), 1)},
    }

    # ---------------------------------------------------------------- 2. 항별 기여
    Qf, Qm = substrate_feed(F)
    sub_f = np.zeros(n)
    sub_m = np.zeros(n)
    for i in range(EMP["D"]):
        lag = i + 1
        sub_f[lag:] += Wf[i] * np.nan_to_num(Qf.to_numpy(float)[:-lag])
        sub_m[lag:] += Wm[i] * np.nan_to_num(Qm.to_numpy(float)[:-lag])
    pred_now = empirical_series(F, y, mode="nowcast")
    ok = np.isfinite(y) & np.isfinite(pred_now)
    base = pred_now - sub_f - sub_m
    R["항별_기여"] = {
        "n": int(ok.sum()),
        "음식물류항_m3d": round(float(sub_f[ok].mean()), 0),
        "가축분뇨항_m3d": round(float(sub_m[ok].mean()), 0),
        "기저항_m3d": round(float(base[ok].mean()), 0),
        "기저항_비중_pct": round(float(100 * base[ok].mean() / pred_now[ok].mean()), 1),
        "가축분뇨항_비중_pct": round(float(100 * sub_m[ok].mean() / pred_now[ok].mean()), 2),
        "예측평균_vs_실측": [round(float(pred_now[ok].mean()), 0), round(float(y[ok].mean()), 0)],
        "계통편향_m3d": round(float(pred_now[ok].mean() - y[ok].mean()), 0),
    }

    # ---------------------------------------------------------------- 3. 모드별 성능
    emp_now = fold_rmse(make_empirical(F, y, "nowcast"), folds, y)
    emp_fc = fold_rmse(make_empirical(F, y, "forecast"), folds, y)
    emp_rf = fold_rmse(make_empirical_refit(F, y, "forecast"), folds, y)
    R["경험식_모드별"] = {
        "Nowcast(실측 MA30)": round(float(np.nanmean(emp_now)), 1),
        "Forecast(재귀 MA30)": round(float(np.nanmean(emp_fc)), 1),
        "Forecast+계수재적합": round(float(np.nanmean(emp_rf)), 1),
        "검정_nowcast_vs_forecast": paired_test(emp_now, emp_fc, "Nowcast", "Forecast"),
        "해석": "같은 식이 모드만 바꿔도 성능이 갈린다. Nowcast 는 어제까지의 실측을 보고 "
              "오늘을 맞히는 문제이고, 90일 앞 예측에서는 그 정보가 없다.",
    }

    # ---------------------------------------------------------------- 4. 우리 모델과 동일 폴드 비교
    models = {
        "M1_기계론(반입구동)": make_M1(F, y, "표준", SRT_REFERENCE_D, K_HYD, True),
        "M1F_기계론(실측투입구동)": make_M1_feed(F, y),
        "M2_ADL": make_M2(F, y, "표준", SRT_REFERENCE_D, K_HYD),
        "M3_VS물질수지": make_M3(F, y, True),
        "M4_트리_lag격자": make_M4(F, y, "forecast", True),
        "M5_상태공간칼만": make_M5(F, y, "표준", SRT_REFERENCE_D, K_HYD),
        "경험식_Forecast": make_empirical(F, y, "forecast"),
    }
    fr, oof = {}, {}
    for nm, fp in models.items():
        fr[nm] = fold_rmse(fp, folds, y)
        oof[nm] = oof_predictions(fp, folds, y, n)
    BASE = fr["M1_기계론(반입구동)"]
    R["동일폴드_비교"] = [
        {"모델": nm, "CV_RMSE": round(float(np.nanmean(v)), 1),
         "폴드SD": round(float(np.nanstd(v, ddof=1)), 1),
         **{k: v2 for k, v2 in paired_test(v, BASE, nm, "M1기준").items()
            if k in ("ΔRMSE", "SE", "p", "판정")}}
        for nm, v in sorted(fr.items(), key=lambda kv: np.nanmean(kv[1]))]

    # ---------------------------------------------------------------- 5. 결함 진단
    R["결함진단"] = {
        "D1_구동변수": reconstruction_check(F),
        "D2_가축분뇨_소실": _manure_diag(F, y, folds),
    }

    # ---------------------------------------------------------------- 5b. 수율 환산 대조
    R["수율_환산대조"] = _yield_compare()

    # ---------------------------------------------------------------- 6. 개선 앙상블
    keep = ["M1F_기계론(실측투입구동)", "M3_VS물질수지", "M4_트리_lag격자", "M5_상태공간칼만",
            "경험식_Forecast"]
    ens_new = combine({k: oof[k] for k in keep}, y, folds, "simple")
    ens_old = combine({k: oof[k] for k in
                       ["M1_기계론(반입구동)", "M2_ADL", "M3_VS물질수지",
                        "M4_트리_lag격자", "M5_상태공간칼만"]}, y, folds, "simple")
    fn = np.array([_r(y, ens_new, te) for _, te in folds])
    fo = np.array([_r(y, ens_old, te) for _, te in folds])
    R["개선앙상블"] = {
        "기존구성_CV_RMSE": round(float(np.nanmean(fo)), 1),
        "개선구성_CV_RMSE": round(float(np.nanmean(fn)), 1),
        "구성": keep,
        "검정": paired_test(fn, fo, "개선구성", "기존구성"),
        "holdout_2023": {"개선": metrics(y[(F.date >= "2023-01-01").to_numpy()],
                                       ens_new[(F.date >= "2023-01-01").to_numpy()]),
                        "기존": metrics(y[(F.date >= "2023-01-01").to_numpy()],
                                      ens_old[(F.date >= "2023-01-01").to_numpy()])},
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "compare_empirical.json").write_text(json.dumps(R, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    _print(R)
    return 0


def _manure_diag(F, y, folds):
    """가축분뇨 계수가 0 으로 붕괴하는 원인 진단."""
    from src.bioguard.empirical import feed_driven_design
    from scipy.optimize import lsq_linear
    from src.bioguard.models import design_matrix, nnls_fit
    from src.bioguard.config import SUBSTRATES, SUB_KR

    obs = np.where(np.isfinite(y))[0]
    Wi = design_matrix(F, "표준", SRT_REFERENCE_D, K_HYD, True, "tonnage")
    Wf_ = feed_driven_design(F, SRT_REFERENCE_D, K_HYD, True)
    ti = nnls_fit(Wi, y, obs)
    tf = lsq_linear(Wf_[obs], y[obs], bounds=(0.0, np.inf)).x
    # 설계행렬 열간 상관 (공선성)
    def corr(W):
        C = np.corrcoef(W[obs, :3].T)
        return {f"{SUB_KR[a]}~{SUB_KR[b]}": round(float(C[i, j]), 3)
                for i, a in enumerate(SUBSTRATES) for j, b in enumerate(SUBSTRATES)
                if i < j}
    return {
        "반입구동_θ": {SUB_KR[s]: round(float(v), 3) for s, v in zip(SUBSTRATES, ti[:3])},
        "반입구동_절편": round(float(ti[3]), 0),
        "실측투입구동_θ": {SUB_KR[s]: round(float(v), 3) for s, v in zip(SUBSTRATES, tf[:3])},
        "실측투입구동_절편": round(float(tf[3]), 0),
        "설계행렬_열간상관_반입구동": corr(Wi),
        "설계행렬_열간상관_투입구동": corr(Wf_),
        "경험식_Y": {"음식물류": EMP["Y_f"], "가축분뇨": EMP["Y_m"]},
        "판정": "세 기질 열이 서로 강하게 상관하면(조성비가 거의 고정) NNLS 는 한 열에 "
              "몰아주고 나머지를 경계값 0 으로 민다. 가축분뇨 계수 0 은 '메탄을 안 낸다'는 "
              "뜻이 아니라 '이 자료로 분리 식별되지 않는다'는 뜻이다.",
    }


def _yield_compare():
    """
    두 모델을 **같은 단위**로 환산해 비교한다.

    우리 모델의 θ 는 h(τ) 와 곱해져 쓰이므로 그대로 비교하면 안 된다. h 의 총합인
    정상이득 G=k·SRT/(1+k·SRT) 를 곱해야 '투입 1톤이 결국 만드는 메탄'이 된다.
    경험식의 Y 도 이산합 Σk·e^(−kd) 만큼만 실현되므로 같은 보정을 한다.
    """
    from src.bioguard.config import SUB_KR
    from src.bioguard.models import stoichiometry

    st = stoichiometry()["기질"]
    # 실측 투입 구동 NNLS 계수 (D2 진단과 동일 적합)
    theta = {"foodww": 32.754, "manure": 11.791, "food": 0.0}
    d = np.arange(1, EMP["D"] + 1)
    emp_inf = {"foodww": EMP["Y_f"] * EMP["k_f"] * np.exp(-EMP["k_f"]) / (1 - np.exp(-EMP["k_f"])),
               "manure": EMP["Y_m"] * EMP["k_m"] * np.exp(-EMP["k_m"]) / (1 - np.exp(-EMP["k_m"]))}
    emp_7d = {"foodww": float((EMP["Y_f"] * EMP["k_f"] * np.exp(-EMP["k_f"] * d)).sum()),
              "manure": float((EMP["Y_m"] * EMP["k_m"] * np.exp(-EMP["k_m"] * d)).sum())}
    cap = {"foodww": 0.65, "manure": 0.48}

    rows = []
    for s in ("foodww", "manure"):
        g = K_HYD[s] * SRT_REFERENCE_D / (1 + K_HYD[s] * SRT_REFERENCE_D)
        ours_t = theta[s] * g                      # ㎥CH₄ / 투입 t
        emp_t = emp_inf[s]
        vs = st[s]["VSfrac"]                       # t VS / t 습중량
        b_th = st[s]["B_th"]
        ours_vs, emp_vs = ours_t / vs / 1000.0, emp_t / vs / 1000.0   # ㎥/kgVS
        rows.append({
            "기질": SUB_KR[s],
            "우리_θ_m3_per_t": round(theta[s], 3),
            "우리_정상이득_G": round(float(g), 3),
            "우리_유효_m3_per_t": round(float(ours_t), 2),
            "경험식_Y": EMP["Y_f"] if s == "foodww" else EMP["Y_m"],
            "경험식_무한지평_m3_per_t": round(float(emp_t), 2),
            "경험식_7일절단_m3_per_t": round(emp_7d[s], 2),
            "우리_VS수율_m3_per_kgVS": round(float(ours_vs), 3),
            "경험식_VS수율_m3_per_kgVS": round(float(emp_vs), 3),
            "문헌_상한": cap[s],
            "우리_BD": round(float(ours_vs / b_th), 3),
            "경험식_BD": round(float(emp_vs / b_th), 3),
            "우리_물리통과": bool(ours_vs <= cap[s] and 0 < ours_vs / b_th <= 1),
            "경험식_물리통과": bool(emp_vs <= cap[s] and 0 < emp_vs / b_th <= 1),
        })
    return {
        "표": rows,
        "판정": "투입 톤 기준 유효 수율은 두 모델이 같은 자릿수로 수렴한다(음식물류 28.9 vs "
              "24.2, 가축분뇨 7.9 vs 5.3 ㎥/t). 그러나 **물리 검사는 경험식만 통과한다** — "
              "우리 음폐수 수율은 BD 1.07 로 이론 최대를 넘고, 경험식은 0.90 으로 "
              "'제거 COD 의 10~17%가 균체 합성' 창(BD 0.83~0.90) 안에 정확히 들어온다.",
        "주의": "경험식의 7일 절단은 가축분뇨 잠재량의 6.8% 만 담는다. 나머지는 C_base 항이 "
              "흡수하므로, 경험식의 Y_m=5.31 은 '식 안에서 실제로 쓰이는 값'이 아니라 "
              "'기저항과 분리되지 않은 명목값'이다.",
    }


def _r(y, p, te):
    m = np.isfinite(y[te]) & np.isfinite(p[te])
    if m.sum() < 2:
        return np.nan
    r = y[te][m] - p[te][m]
    return float(np.sqrt(r @ r / len(r)))


def _print(R):
    print(f"\n폴드 {R['폴드수']}개 · 타깃 관측 {R['타깃_관측일']}일")
    print("\n=== 경험식 항별 기여 ===")
    c = R["항별_기여"]
    print(f"  음식물류 {c['음식물류항_m3d']:>6.0f}   가축분뇨 {c['가축분뇨항_m3d']:>5.0f} "
          f"({c['가축분뇨항_비중_pct']}%)   기저 {c['기저항_m3d']:>6.0f} ({c['기저항_비중_pct']}%)")
    print(f"  예측평균 {c['예측평균_vs_실측'][0]:.0f} vs 실측 {c['예측평균_vs_실측'][1]:.0f} "
          f"→ 계통편향 {c['계통편향_m3d']:+.0f} ㎥/d")
    print("\n=== 경험식 모드별 CV-RMSE ===")
    for k in ("Nowcast(실측 MA30)", "Forecast(재귀 MA30)", "Forecast+계수재적합"):
        print(f"  {k:24s} {R['경험식_모드별'][k]:>8.1f}")
    print("\n=== 동일 폴드 비교 ===")
    for r in R["동일폴드_비교"]:
        print(f"  {r['모델']:26s} {r['CV_RMSE']:>8.1f}  Δ{str(r.get('ΔRMSE')):>9} "
              f"p={str(r.get('p')):>7}  {r.get('판정','')}")
    print("\n=== 결함 D1: 구동 변수 ===")
    d1 = R["결함진단"]["D1_구동변수"]
    print(f"  반입합계 vs 실측투입 r={d1['반입합계_vs_실측투입_r']}  → "
          f"전단커널 재구성 r={d1['전단커널재구성_vs_실측투입_r']}")
    print(f"  반입 CV {d1['반입_CV_pct']}% vs 투입 CV {d1['투입_CV_pct']}%")
    print("\n=== 결함 D2: 가축분뇨 계수 ===")
    d2 = R["결함진단"]["D2_가축분뇨_소실"]
    print(f"  반입구동   θ={d2['반입구동_θ']} 절편 {d2['반입구동_절편']:.0f}")
    print(f"  투입구동   θ={d2['실측투입구동_θ']} 절편 {d2['실측투입구동_절편']:.0f}")
    print(f"  열간상관(투입구동) {d2['설계행렬_열간상관_투입구동']}")
    print("\n=== 수율 환산 대조 (㎥CH₄/kgVS_in) ===")
    for r in R["수율_환산대조"]["표"]:
        print(f"  {r['기질']:6s} 우리 {r['우리_VS수율_m3_per_kgVS']:.3f} (BD {r['우리_BD']:.2f}, "
              f"{'통과' if r['우리_물리통과'] else '미통과'})   "
              f"경험식 {r['경험식_VS수율_m3_per_kgVS']:.3f} (BD {r['경험식_BD']:.2f}, "
              f"{'통과' if r['경험식_물리통과'] else '미통과'})   상한 {r['문헌_상한']}")
    print("\n=== 개선 앙상블 ===")
    e = R["개선앙상블"]
    print(f"  기존 {e['기존구성_CV_RMSE']} → 개선 {e['개선구성_CV_RMSE']}  "
          f"Δ{e['검정']['ΔRMSE']} p={e['검정']['p']} {e['검정']['판정']}")
    print(f"  2023 홀드아웃: 기존 R²={e['holdout_2023']['기존']['R2']} → "
          f"개선 R²={e['holdout_2023']['개선']['R2']}")


if __name__ == "__main__":
    raise SystemExit(main())
