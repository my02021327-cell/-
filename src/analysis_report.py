# -*- coding: utf-8 -*-
"""영천 BGP 메탄생성 예측·소화조 상태진단 리포트용 계산 스크립트.

MODELING_OVERVIEW v2 / EXPERT_REVIEW 권고를 따른다:
  - 예측 계층: 1차 가수분해 커널(채택 모델) + persistence 기준선 병기
  - 진단 계층: Y_COD 90일 기준선 상대편차, 잔차 관리도, FAN/알칼리도 추세
  - 시간순 분할(train 2018-2021 / valid 2022 / test 2023), K-fold 금지
결과는 outputs/report_data.json 으로 저장하며 HTML 리포트가 이를 내장한다.
"""
import json
import numpy as np
import pandas as pd

CSV = "data/영천BGP_MASTER_2018-2023.csv"

M = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]

# ---------------------------------------------------------------- Layer 1: 예측
TS = M.acid_TS_pct.interpolate(limit=3)
Q = M.feed_AB_tpd.ffill(limit=2)
TS_load = (Q * TS / 100).ffill(limit=3)
y = M.biogas_AB_m3d


def kernel(s, k=0.35, K=120):
    tau = np.arange(K + 1)
    w = k * np.exp(-k * tau)
    w /= w.sum()
    v = np.nan_to_num(s.values)
    out = np.zeros(len(v))
    for i, wv in enumerate(w):
        out[i:] += wv * v[: len(v) - i]
    return pd.Series(out, index=s.index)


H = kernel(TS_load, k=0.35)
D = pd.concat([H.rename("H"), y.rename("y")], axis=1).dropna()
tr = D[D.index.year <= 2021]
va = D[D.index.year == 2022]
te = D[D.index.year == 2023]

A = np.c_[np.ones(len(tr)), tr[["H"]].values]
coef = np.linalg.lstsq(A, tr["y"].values, rcond=None)[0]
c0, a1 = float(coef[0]), float(coef[1])


def metrics(sub):
    p = c0 + a1 * sub["H"].values
    t = sub["y"].values
    ss = ((t - p) ** 2).sum()
    r2 = 1 - ss / ((t - t.mean()) ** 2).sum()
    rmse = float(np.sqrt(ss / len(t)))
    mae = float(np.abs(t - p).mean())
    return {"n": int(len(t)), "R2": round(float(r2), 3), "RMSE": round(rmse), "MAE": round(mae)}


pred_all = c0 + a1 * D["H"]

# persistence 기준선 (동일 표본)
def persistence_metrics(sub):
    yl = y.shift(1).reindex(sub.index)
    m = yl.notna()
    t = sub["y"].values[m.values]
    p = yl.values[m.values]
    ss = ((t - p) ** 2).sum()
    return {
        "n": int(m.sum()),
        "R2": round(float(1 - ss / ((t - t.mean()) ** 2).sum()), 3),
        "RMSE": round(float(np.sqrt(ss / m.sum()))),
        "MAE": round(float(np.abs(t - p).mean())),
    }


# persistence + Δfeed (train 2018-2021 로 계수 추정)
dfeed = M.feed_AB_tpd.diff()
resid_tr = (y - y.shift(1)).loc["2018":"2021"]
X_tr = dfeed.loc["2018":"2021"]
mm = resid_tr.notna() & X_tr.notna()
beta = np.polyfit(X_tr[mm].values, resid_tr[mm].values, 1)
beta1, beta0 = float(beta[0]), float(beta[1])

pd_pred = y.shift(1) + beta1 * dfeed + beta0


def pd_metrics(idx_year):
    sub = pd.concat([y.rename("y"), pd_pred.rename("p")], axis=1).dropna()
    sub = sub[sub.index.year == idx_year] if idx_year else sub
    t, p = sub["y"].values, sub["p"].values
    ss = ((t - p) ** 2).sum()
    return {
        "n": int(len(t)),
        "R2": round(float(1 - ss / ((t - t.mean()) ** 2).sum()), 3),
        "RMSE": round(float(np.sqrt(ss / len(t)))),
        "MAE": round(float(np.abs(t - p).mean())),
    }


# 지평별 persistence skill (h=1/3/7/14/30, 2022~2023)
horizons = {}
vv = y.loc["2022":"2023-09-17"]
for h in [1, 3, 7, 14, 30]:
    yl = y.shift(h).reindex(vv.index)
    m = yl.notna() & vv.notna()
    t, p = vv[m].values, yl[m].values
    ss = ((t - p) ** 2).sum()
    horizons[h] = {
        "R2": round(float(1 - ss / ((t - t.mean()) ** 2).sum()), 3),
        "RMSE": round(float(np.sqrt(ss / m.sum()))),
    }

# ---------------------------------------------------------- Layer 2: 진단 지표
# Y_COD: 제거 COD당 메탄수율 (m3 CH4 / kg COD removed), 열역학 상한 0.35
dig_COD = M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)
COD_removed_kg = M.feed_AB_tpd * (M.acid_CODcr_mgL - dig_COD) / 1000.0
Y_COD = M.CH4_m3d / COD_removed_kg
qc_total = int(Y_COD.notna().sum())
bad = (Y_COD <= 0) | (Y_COD > 0.35)
qc_bad = int((bad & Y_COD.notna()).sum())
Yq = Y_COD.where(~bad)

base90 = Yq.rolling(90, min_periods=30).median().shift(1)
alarm_ycod = Yq < base90 * 0.85
ycod_alarm_days = int(alarm_ycod.sum())
ycod_alarm_rate = round(100 * alarm_ycod.sum() / Yq.notna().sum(), 1)

# 잔차 관리도 (persistence+Δfeed 잔차 z, train σ 기준)
resid = y - pd_pred
sigma = float(resid.loc["2018":"2021"].std())
z = resid / sigma
z2 = (z.abs() > 3) & (z.abs().shift(1) > 3)
resid_alarms = z2[z2].index.strftime("%Y-%m-%d").tolist()

# FAN·알칼리도·안정성 연도별 표
ann = {}
for yr, g in M.groupby(M.index.year):
    ann[int(yr)] = {
        "feed": round(float(g.feed_AB_tpd.mean()), 1),
        "biogas": round(float(g.biogas_AB_m3d.mean())),
        "ch4": round(float(g.CH4_m3d.mean())),
        "ch4pct": round(float(g.CH4_pct.mean()), 1),
        "pH_A": round(float(g.dig_pH_A.mean()), 2),
        "VFA_A": round(float(g.VFA_A_mgL.mean())),
        "ALK_A": round(float(g.ALK_A_mgL.mean())),
        "VFA_ALK": round(float(g.VFA_ALK_A.mean()), 2),
        "NH3N_A": (round(float(g.NH3N_A_mgL.mean())) if g.NH3N_A_mgL.notna().sum() > 0 else None),
        "FAN_A": (round(float(g.FAN_A_mgL.median())) if g.FAN_A_mgL.notna().sum() > 0 else None),
        "Y_COD": (round(float(Yq[Yq.index.year == yr].median()), 3) if Yq[Yq.index.year == yr].notna().sum() > 0 else None),
    }

# 최근 상태 스냅샷 (관측 마지막 90일)
tail = M.loc["2023-06-20":"2023-09-17"]
last_valid = {
    "date_range": "2023-06-20 ~ 2023-09-17",
    "biogas": round(float(tail.biogas_AB_m3d.mean())),
    "ch4": round(float(tail.CH4_m3d.mean())),
    "feed": round(float(tail.feed_AB_tpd.mean()), 1),
    "pH_A": round(float(tail.dig_pH_A.mean()), 2),
    "VFA_ALK": round(float(tail.VFA_ALK_A.mean()), 3),
    "ALK_A": round(float(tail.ALK_A_mgL.mean())),
    "Y_COD": round(float(Yq.loc["2023-06-20":"2023-09-17"].median()), 3),
    "Y_COD_base": round(float(base90.loc["2023-09-01":"2023-09-17"].median()), 3),
}

# ------------------------------------------------------------------ 시계열 내보내기
def ser(s, r=0):
    return [None if pd.isna(v) else round(float(v), r) for v in s.values]


TS_out = pd.DataFrame(
    {
        "y": y,
        "pred": pred_all.reindex(y.index),
        "feed": M.feed_AB_tpd,
        "ycod": Yq,
        "ycod_base": base90,
        "z": z,
        "alk": M.ALK_A_mgL,
        "vfaalk": M.VFA_ALK_A,
        "ph": M.dig_pH_A,
        "fan": M.FAN_A_mgL,
    }
)

out = {
    "dates": TS_out.index.strftime("%Y-%m-%d").tolist(),
    "series": {
        "biogas": ser(TS_out.y),
        "pred": ser(TS_out.pred),
        "feed": [None if pd.isna(v) else round(float(v), 1) for v in TS_out.feed.values],
        "ycod": [None if pd.isna(v) else round(float(v), 4) for v in TS_out.ycod.values],
        "ycod_base": [None if pd.isna(v) else round(float(v), 4) for v in TS_out.ycod_base.values],
        "z": [None if pd.isna(v) else round(float(v), 2) for v in TS_out.z.values],
        "alk": ser(TS_out.alk),
        "vfaalk": [None if pd.isna(v) else round(float(v), 3) for v in TS_out.vfaalk.values],
        "ph": [None if pd.isna(v) else round(float(v), 2) for v in TS_out.ph.values],
        "fan": [None if pd.isna(v) else round(float(v), 1) for v in TS_out.fan.values],
    },
    "model": {
        "a": round(a1, 2),
        "c": round(c0, 1),
        "k": 0.35,
        "train": metrics(tr),
        "valid": metrics(va),
        "test": metrics(te),
        "persistence_test": persistence_metrics(te),
        "persistence_valid": persistence_metrics(va),
        "pd_beta": round(beta1, 2),
        "pd_2022": pd_metrics(2022),
        "pd_2023": pd_metrics(2023),
        "horizons": horizons,
    },
    "diag": {
        "ycod_n": qc_total,
        "ycod_qc_excluded": qc_bad,
        "ycod_alarm_days": ycod_alarm_days,
        "ycod_alarm_rate": ycod_alarm_rate,
        "resid_sigma": round(sigma),
        "resid_alarms": resid_alarms,
        "snapshot": last_valid,
    },
    "annual": ann,
}

with open("outputs/report_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("=== 1차 가수분해 모델  biogas(t) = a·H(t) + c ===")
print(f"a={a1:.2f}, c={c0:.1f}")
print("train:", metrics(tr))
print("valid 2022:", metrics(va))
print("test  2023:", metrics(te))
print("persistence test:", persistence_metrics(te))
print("persistence+Δfeed 2022:", pd_metrics(2022), " 2023:", pd_metrics(2023))
print("β(Δfeed) =", round(beta1, 2))
print("horizons:", horizons)
print("Y_COD n:", qc_total, " QC 제외:", qc_bad, f"({100*qc_bad/qc_total:.1f}%)")
print("Y_COD 경보일:", ycod_alarm_days, f"({ycod_alarm_rate}%)")
print("잔차 경보일:", len(resid_alarms), resid_alarms[:10])
print("snapshot:", last_valid)
