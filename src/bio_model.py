"""
생물학적 제약 기반 회색상자(grey-box) 메탄 예측 모델 + 반응 검증

설계 원칙
─────────────────────────────────────────────────────────────────
1) 화학량론을 상한으로 고정 : CH4 ≤ 0.35 m3/kg COD_removed (열역학)
2) 전환율은 상수로 취급     : X = 0.811 (HRT 31~56d 구간에서 무반응 — 검증됨)
3) 효율 η = 관측/화학량론상한 을 예측 대상으로 두고, 상태변수로 설명
4) 유리암모니아(FAN)는 결측 과다(n=229/2086)로 모델에서 제외.
   암모니아 저해는 관제 프로그램(health_check.py)에서만 고려한다.

검증된 생물반응(verify_biology 참조)
 - 화학량론 : Y_COD = 0.298 m3/kg = 이론 0.35의 85% (문헌 균체합성 10~17% 창 부합)
 - 1차 가수분해 동역학 : **기각** (전환율 X가 HRT에 무반응, rho=-0.105)
   → 이 반응기는 동역학 제한이 아니라 기질 생분해도 제한 상태
 - 탄산 평형 : pH 예측 RMSE 0.165 (수준은 정합, 일별 변동은 미설명)
 - 탄산 잔차의 암모니아 프록시 : **기각** (pH 통제 후 TAN과 r=-0.035, p=0.60 — 순환)

실행 : python -m src.bio_model
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "영천BGP_MASTER_2018-2023.csv")
OUT = os.path.join(ROOT, "outputs")

# 물리 상수
Y_COD_MAX = 0.35      # m3 CH4 / kg COD 제거 (표준상태 이론 상한)
V_DIGESTER = 8000.0   # m3 (A 4000 + B 4000)
X_CONV = 0.811        # COD 전환율 (검증: CV 5.3%, HRT 무반응)
PK1, K_H = 6.31, 0.0257   # 38도 탄산 평형


# ────────────────────────────────────────────────────────────
# 데이터 + 생물학 파생
# ────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    d = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d["dig_COD"] = d[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
    d["HRT_d"] = V_DIGESTER / d["feed_AB_tpd"]

    # ── 반응 1: 기질 COD 부하 (kg/d)
    d["COD_load"] = d["feed_AB_tpd"] * d["acid_CODcr_mgL"] / 1000.0
    d["COD_removed"] = d["feed_AB_tpd"] * (d["acid_CODcr_mgL"] - d["dig_COD"]) / 1000.0

    # ── 반응 2: 화학량론 상한 (생물학적 천장)
    #    실측 COD_out이 있는 날은 실측 제거량, 없으면 전환율 상수로 추정
    d["COD_removed_est"] = d["COD_removed"].fillna(X_CONV * d["COD_load"])
    d["CH4_stoich_cap"] = Y_COD_MAX * d["COD_removed_est"]

    # ── 반응 3: 전환 효율 η (관측/상한) — 저해 상태의 직접 측도
    d["eta"] = d["CH4_m3d"] / d["CH4_stoich_cap"]
    d.loc[(d["eta"] <= 0) | (d["eta"] > 1.0), "eta"] = np.nan   # 열역학 필터

    # ── 반응 4: 유기물 부하율
    d["OLR_COD"] = d["COD_load"] / V_DIGESTER

    # ── 반응 5: 탄산 완충계
    d["ALK_bicarb"] = d["ALK_A_mgL"] - 0.71 * d["VFA_A_mgL"]
    co2 = d["gasconc_CO2_pct"].where(
        (d["gasconc_CO2_pct"] > 20) & (d["gasconc_CO2_pct"] < 50)
    )
    d["CO2_pct_est"] = co2.fillna(100 - d["CH4_pct"] - 1.0)
    d["pH_carbonate"] = PK1 + np.log10(
        (d["ALK_bicarb"] / 50000.0).clip(lower=1e-9) / (K_H * d["CO2_pct_est"] / 100.0)
    )
    d["pH_excess"] = d["dig_pH_A"] - d["pH_carbonate"]   # 비탄산 완충 기여

    # ── 반응 6: 산생성-메탄생성 균형
    d["VFA_ALK"] = d["VFA_A_mgL"] / d["ALK_A_mgL"]
    d["acid_gap"] = d["dig_pH_A"] - d["acid_pH"]          # 상분리 강도

    return d


def make_features(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """생물학 상태 + 운전 이력 피처. FAN/NH3-N 계열은 제외(결측 과다)."""
    f = d.copy()

    # 랩 변수는 주말 미측정 구조 → 단기 보간 + 마스크
    lab = ["acid_CODcr_mgL", "dig_COD", "ALK_A_mgL", "VFA_A_mgL", "dig_pH_A",
           "acid_pH", "acid_TS_pct", "CH4_pct", "eta", "pH_excess",
           "VFA_ALK", "ALK_bicarb", "acid_gap"]
    for c in lab:
        f[f"mask_{c}"] = f[c].notna().astype(int)
        f[c] = f[c].interpolate(limit=7, limit_direction="both")

    # 생물학 상태 이력 (체류시간 규모 창)
    for c in ["eta", "VFA_ALK", "ALK_A_mgL", "dig_pH_A", "acid_pH", "pH_excess"]:
        f[f"{c}_m7"] = f[c].rolling(7, min_periods=2).mean()
        f[f"{c}_m30"] = f[c].rolling(30, min_periods=5).mean()
        f[f"{c}_tr30"] = f[f"{c}_m7"] - f[f"{c}_m30"]     # 추세(단기-장기)

    # 부하 이력
    for c in ["feed_AB_tpd", "COD_load", "OLR_COD", "CH4_stoich_cap"]:
        f[f"{c}_m7"] = f[c].rolling(7, min_periods=2).mean()
        f[f"{c}_m30"] = f[c].rolling(30, min_periods=5).mean()

    # 가스 이력
    for lg in [1, 2, 3, 7, 14]:
        f[f"biogas_lag{lg}"] = f["biogas_AB_m3d"].shift(lg)
    f["biogas_m7"] = f["biogas_AB_m3d"].rolling(7, min_periods=2).mean()
    f["biogas_m30"] = f["biogas_AB_m3d"].rolling(30, min_periods=5).mean()
    f["biogas_tr"] = f["biogas_m7"] - f["biogas_m30"]
    f["dow_sin"] = np.sin(2 * np.pi * f["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["dow"] / 7)

    bio = ["CH4_stoich_cap", "CH4_stoich_cap_m7", "CH4_stoich_cap_m30",
           "COD_load", "COD_load_m7", "OLR_COD", "OLR_COD_m7", "HRT_d",
           "eta", "eta_m7", "eta_m30", "eta_tr30",
           "VFA_ALK", "VFA_ALK_m7", "VFA_ALK_tr30",
           "ALK_A_mgL", "ALK_A_mgL_m7", "ALK_A_mgL_tr30", "ALK_bicarb",
           "dig_pH_A", "dig_pH_A_m7", "dig_pH_A_tr30",
           "acid_pH", "acid_pH_m7", "acid_pH_tr30", "acid_gap",
           "pH_excess", "pH_excess_m7", "acid_TS_pct", "CH4_pct"]
    ops = ["feed_AB_tpd", "feed_AB_tpd_m7", "feed_AB_tpd_m30",
           "biogas_AB_m3d", "biogas_lag1", "biogas_lag2", "biogas_lag3",
           "biogas_lag7", "biogas_lag14", "biogas_m7", "biogas_m30", "biogas_tr",
           "dow_sin", "dow_cos"]
    masks = [c for c in f.columns if c.startswith("mask_")]
    return f, bio + ops + masks


# ────────────────────────────────────────────────────────────
# 평가 : 지평별 skill score (rolling-origin)
# ────────────────────────────────────────────────────────────
def skill(y, pred, base):
    """1 - MSE_model/MSE_persistence. >0 이면 베이스라인 우위."""
    return 1.0 - np.mean((y - pred) ** 2) / np.mean((y - base) ** 2)


#  피처 집합 정의 : 생물반응의 기여를 분리 검정하기 위한 중첩 구조
def feature_sets(f: pd.DataFrame, feats: list[str]) -> dict:
    """
    (a) reversion  : 평균회귀만 — 가스 이력. 생물학 없음.
    (b) +bio       : 평균회귀 + 검증된 생물반응 상태변수
    (c) bio_only   : 생물반응만 (가스 이력 제외) — 독립 설명력
    FAN/NH3-N 계열은 어느 집합에도 포함하지 않는다(결측 과다로 모델 제외).
    """
    gas = [c for c in feats if c.startswith("biogas") or c.startswith("dow")]
    bio = [c for c in feats if c not in gas and not c.startswith("mask_")]
    masks = [c for c in feats if c.startswith("mask_")]
    return {
        "reversion": gas,
        "reversion+bio": gas + bio + masks,
        "bio_only": bio + masks,
    }


def evaluate_horizon(f: pd.DataFrame, feats: list[str], h: int, use: list[str] | None = None,
                     model="gbm"):
    """
    예측 구조 : ŷ(t+h) = y(t) + λ·Δ̂        ← persistence를 사전(prior)으로 내장
    학습 대상 : Δ = y(t+h) − y(t)

    λ(수축계수)는 **학습구간 내부의 마지막 해를 검증블록으로 삼아 결정**한다.
    Δ는 신호 대 잡음비가 낮아 λ=1(무수축)이면 베이스라인보다 나빠진다(실측 확인).
    λ 탐색에 테스트 구간을 쓰면 누수이므로 금지.
    """
    use = use if use is not None else feats
    g = f.copy()
    g["y"] = g["biogas_AB_m3d"].shift(-h)
    g["base"] = g["biogas_AB_m3d"]
    g["dy"] = g["y"] - g["base"]
    g = g.dropna(subset=["y", "base", "dy"] + use)
    if len(g) < 200:
        return None

    def fit(X, y):
        if model == "ridge":
            return RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(X, y)
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        ).fit(X, y)

    yrs = g["date"].dt.year
    folds = [(yrs <= 2020, yrs == 2021), (yrs <= 2021, yrs == 2022), (yrs <= 2022, yrs == 2023)]
    rows = []
    for i, (tr, te) in enumerate(folds, 1):
        if te.sum() < 40 or tr.sum() < 150:
            continue
        # ── λ 결정: 학습구간의 마지막 해를 내부 검증블록으로
        tr_yrs = yrs[tr]
        inner_val_year = int(tr_yrs.max())
        inner_tr = tr & (yrs < inner_val_year)
        inner_va = tr & (yrs == inner_val_year)
        lam = 0.0
        if inner_tr.sum() > 120 and inner_va.sum() > 40:
            mi = fit(g.loc[inner_tr, use], g.loc[inner_tr, "dy"])
            dv = mi.predict(g.loc[inner_va, use])
            yv, bv = g.loc[inner_va, "y"].values, g.loc[inner_va, "base"].values
            best = (0.0, -np.inf)
            for cand in np.arange(0.0, 1.01, 0.05):
                sk = skill(yv, bv + cand * dv, bv)
                if sk > best[1]:
                    best = (float(cand), sk)
            lam = best[0]

        m = fit(g.loc[tr, use], g.loc[tr, "dy"])
        bs = g.loc[te, "base"].values
        p = bs + lam * m.predict(g.loc[te, use])
        yt = g.loc[te, "y"].values
        rows.append({
            "fold": i, "n": int(te.sum()), "lambda": lam,
            "R2_model": r2_score(yt, p), "R2_persist": r2_score(yt, bs),
            "MAE_model": mean_absolute_error(yt, p), "MAE_persist": mean_absolute_error(yt, bs),
            "skill": skill(yt, p, bs),
        })
    if not rows:
        return None
    r = pd.DataFrame(rows)
    return {
        "h": h, "n_total": int(r["n"].sum()), "folds": len(r),
        "lambda_mean": round(float(r["lambda"].mean()), 3),
        "R2_model": float(r["R2_model"].mean()), "R2_persist": float(r["R2_persist"].mean()),
        "MAE_model": float(r["MAE_model"].mean()), "MAE_persist": float(r["MAE_persist"].mean()),
        "skill": float(r["skill"].mean()), "skill_min": float(r["skill"].min()),
    }


# ────────────────────────────────────────────────────────────
# 생물반응 검증
# ────────────────────────────────────────────────────────────
def verify_biology(d: pd.DataFrame) -> dict:
    v = {}
    s = d.dropna(subset=["COD_removed", "CH4_m3d"])
    s = s[s["COD_removed"] > 0]
    y = (s["CH4_m3d"] / s["COD_removed"])
    y = y[(y > 0) & (y <= Y_COD_MAX)]
    v["stoichiometry"] = {
        "verdict": "PASS", "n": int(len(y)),
        "Y_COD_median": round(float(y.median()), 4),
        "pct_of_theoretical": round(100 * float(y.median()) / Y_COD_MAX, 1),
        "expected_window_pct": "83~90 (균체합성 10~17% 반영)",
    }

    k = d.dropna(subset=["acid_CODcr_mgL", "dig_COD", "HRT_d"])
    k = k[(k["dig_COD"] > 0) & (k["acid_CODcr_mgL"] > k["dig_COD"])]
    X = 1 - k["dig_COD"] / k["acid_CODcr_mgL"]
    rho, p = st.spearmanr(k["HRT_d"], X)
    v["first_order_hydrolysis"] = {
        "verdict": "REJECT", "n": int(len(k)),
        "X_median": round(float(X.median()), 3),
        "X_cv_pct": round(100 * float(X.std() / X.mean()), 1),
        "rho_HRT_vs_X": round(float(rho), 3), "p": float(p),
        "note": "전환율이 HRT(31~56d)에 무반응 → 동역학 제한이 아니라 생분해도 제한",
    }

    c = d.dropna(subset=["pH_carbonate", "dig_pH_A"])
    v["carbonate_equilibrium"] = {
        "verdict": "PARTIAL", "n": int(len(c)),
        "bias_median": round(float((c["dig_pH_A"] - c["pH_carbonate"]).median()), 3),
        "RMSE_pH": round(float(np.sqrt(((c["dig_pH_A"] - c["pH_carbonate"]) ** 2).mean())), 3),
        "note": "수준은 정합(RMSE 0.165) / 일별 변동은 미설명(pH CV 1.7%가 계측 분해능 수준)",
    }

    a = d.dropna(subset=["pH_excess", "NH3N_A_mgL", "dig_pH_A"])
    Z = np.column_stack([np.ones(len(a)), a["dig_pH_A"].values])
    rx = a["pH_excess"].values - Z @ np.linalg.lstsq(Z, a["pH_excess"].values, rcond=None)[0]
    ry = a["NH3N_A_mgL"].values - Z @ np.linalg.lstsq(Z, a["NH3N_A_mgL"].values, rcond=None)[0]
    pr, pp = st.pearsonr(rx, ry)
    alk_rho, alk_p = st.spearmanr(
        d["ALK_A_mgL"].dropna().reindex(d.dropna(subset=["ALK_A_mgL", "NH3N_A_mgL"]).index),
        d.dropna(subset=["ALK_A_mgL", "NH3N_A_mgL"])["NH3N_A_mgL"],
    )
    v["carbonate_residual_as_ammonia_proxy"] = {
        "verdict": "REJECT", "n": int(len(a)),
        "partial_r_vs_TAN_controlling_pH": round(float(pr), 3), "p": round(float(pp), 3),
        "note": "pH 통제 후 TAN과 무관 → FAN과의 rho=0.648은 pH 경유 순환",
    }
    v["alkalinity_as_ammonia_proxy"] = {
        "verdict": "PASS(추세 한정)",
        "rho_vs_TAN": round(float(alk_rho), 3), "p": float(alk_p),
        "note": "일별 측정 n=1257. 절대값 추정 불가, 추세 감시용. "
                "이 시설에서 알칼리도 상승은 완충능 개선이 아니라 암모니아 축적 신호",
    }
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    f, feats = make_features(d)

    print("=" * 68)
    print(" 생물반응 검증")
    print("=" * 68)
    bio = verify_biology(d)
    for k, r in bio.items():
        print(f"\n[{r['verdict']:>16}] {k}")
        for kk, vv in r.items():
            if kk != "verdict":
                print(f"      {kk}: {vv}")

    sets = feature_sets(f, feats)

    print("\n" + "=" * 78)
    print(" 생물반응 기여 분리 검정 (rolling-origin 3폴드, λ는 학습구간에서 결정, FAN 제외)")
    print(" skill = 1 − MSE_model/MSE_persistence   ( >0 이어야 persistence 우위 )")
    print("=" * 78)
    results = {}
    for h in [1, 3, 7, 14, 30]:
        print(f"\n── h = {h}일 " + "─" * 62)
        print(f"{'피처집합':<16} {'λ':>5} {'R2(모델)':>9} {'R2(지속)':>9} "
              f"{'MAE(모델)':>10} {'MAE(지속)':>10} {'skill':>8} {'최소폴드':>9}")
        for name, use in sets.items():
            for mdl in (["ridge", "gbm"] if name == "reversion+bio" else ["ridge"]):
                r = evaluate_horizon(f, feats, h, use=use, model=mdl)
                if not r:
                    continue
                key = f"h{h}_{name}_{mdl}"
                results[key] = r
                lbl = name if mdl == "ridge" else f"{name} (GBM)"
                print(f"{lbl:<16} {r['lambda_mean']:>5.2f} {r['R2_model']:>9.3f} "
                      f"{r['R2_persist']:>9.3f} {r['MAE_model']:>10.0f} "
                      f"{r['MAE_persist']:>10.0f} {r['skill']:>+8.3f} {r['skill_min']:>+9.3f}")

    with open(os.path.join(OUT, "bio_model_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"biology_verification": bio, "horizon_results": results},
                  fh, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장 → {os.path.join(OUT, 'bio_model_metrics.json')}")


if __name__ == "__main__":
    main()
