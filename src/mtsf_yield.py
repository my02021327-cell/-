# -*- coding: utf-8 -*-
"""메탄수율(MY) 다단계 직접예측 — Choi et al. (Water Research 286:124276, 2025) 방법론 이식.

■ 원 논문
   S. Choi, S.I. Kim, C. Chairattanawat, S. Hwang (POSTECH),
   "Transformer-based multi-step time series forecasting of methane yield in
   full-scale anaerobic digestion", Water Research 286 (2025) 124276.
   전규모 AD 2기(FW:PS = 60:40), 2015.1~2024.6, 3,451일.
   타깃은 **메탄수율 MY [㎥CH₄/kg VS_added]**, 1~14일 다단계 **직접(direct)** 예측.
   구성: 시계열 패칭(PatchTST 변형, 패치 내 변수 혼합) + 인스턴스 정규화(RevIN)
         + ANOVA 기반 시간 임베딩. 비교군 RNN·LSTM·vanilla Transformer.

■ 이 이식이 §2.10(iTransformer)과 결정적으로 다른 점 — 타깃을 바꾼다
   §2.10은 **메탄 발생량(㎥/d)** 을 예측했다. 이 양은 투입량에 지배되어 자기상관이
   극도로 높고(ACF₁ 0.906), 그래서 persistence를 아무도 이기지 못했다.
   본 논문은 **수율(투입 VS 1 kg당 메탄)** 을 예측한다. 분모로 부하를 나눠버리므로
   '어제 값을 옮기는' 전략이 통하지 않고, 소화조 내부 상태가 설명해야 할 몫만 남는다.
   §2 전체가 요구해 온 '소화조 내부에서 기인한 메탄'에 방법론적으로 가장 가깝다.

■ 영천 이식 시 자율 판단 (사용자 위임)
   · 타깃 MY = CH₄[㎥/d] / (투입량[t/d] × 산생성조 VS[%] × 10)  [㎥CH₄/kg VS]
   · 논문의 VS_eff·TS_eff(유출 실측)는 영천에 없다(§2.11). CSTR 가정하에 조내 VS·TS로 대체.
   · §2.11에서 유일하게 정보가 있던 **VS/TS 비**를 피처로 추가한다.
   · 논문의 vol은 계란형 소화조라 일별 변동(11,884~14,342㎥)하지만 영천은 고정 8,000㎥ → 제외.
   · swell(= vol − eff)은 구성 불가 → 탈수량을 유출 대리변수로 쓰되 혼합 스트림임을 명시.
   · H₂S 미측정 → AD 안정성 지표인 **VFA/ALK 비**로 대체.
   · 문맥길이는 논문과 같이 체류시간의 배수: 0.5/1/1.5/2 × SRT. SRT는 §2.11의 12일 채택.
   · 논문에 없는 **persistence 기준선**을 항상 함께 싣는다(§2.10의 교훈).
   · MY의 분모에 투입량이 들어가므로 Q_in 계열을 피처로 쓰면 준순환이다.
     논문은 그대로 쓴다. 여기서는 **논문충실판 / 분모배제판을 모두** 돌려 함께 보고한다.

■ 구간 (2023년은 온도·VFA 붕괴로 제외)
   학습 2018-01-01~2020-12-31 / 사고학습 2021 / 평가 2022   (≈ 6:2:2)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

SRT = 12.0
V_DESIGN = 8000.0
TR = ("2018-01-01", "2020-12-31"); VA = ("2021-01-01", "2021-12-31")
TE = ("2022-01-01", "2022-12-31")
HORIZONS = [1, 3, 7, 14]
CTX = {"0.5·SRT": 6, "1·SRT": 12, "1.5·SRT": 18, "2·SRT": 24}
SEEDS = [0, 1, 2]
OUT = {"srt_used": SRT, "splits": {"train": TR, "valid": VA, "test": TE},
       "note_2023": "온도·VFA 붕괴로 제외"}

# ================================================================ 데이터
M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M.loc["2018-01-01":"2022-12-31"]
F = pd.date_range(M.index.min(), M.index.max(), freq="D")
M = M.reindex(F)

feed = M.feed_AB_tpd.ffill(limit=2)
VS_added = feed * M.acid_VS_pct / 100 * 1000                     # kg VS/d
CH4 = M.biogas_AB_m3d * M.CH4_pct.ffill(limit=3) / 100
MY = (CH4 / VS_added).replace([np.inf, -np.inf], np.nan)
MY = MY.where((MY > 0.05) & (MY < 3.0))                          # 물리 범위 밖 제거

# 기질별 VS 부하 — §2.8의 전단 커널(τ₀=0, τ_mix=3)을 통과시킨 조성 사용
KS = ["foodww", "manure", "food"]
I = M[["intake_foodww_tpd", "intake_manure_tpd", "intake_food_tpd"]].clip(lower=0)
I.columns = KS
I = I.where(I.sum(axis=1) > 5)
R = I.div(I.sum(axis=1), axis=0).ffill(limit=7)
w = np.exp(-np.arange(41) / 3.0); w /= w.sum()


def conv(s, wv):
    v = np.nan_to_num(s.values); o = np.zeros(len(v))
    for i, x in enumerate(wv):
        o[i:] += x * v[: len(v) - i]
    return pd.Series(o, index=s.index)


Rm = pd.DataFrame({c: conv(R[c], w) for c in KS})
Rm = Rm.div(Rm.sum(axis=1), axis=0)

X = pd.DataFrame(index=F)
X["SRT_d"] = V_DESIGN / feed                                     # 논문 HRT_d 대응
for c in KS:
    X[f"Qin_{c}"] = feed * Rm[c] * M.acid_VS_pct / 100           # ton-VS/d
X["temp"] = M[["dig_T_A_C", "dig_T_B_C"]].mean(axis=1)
X["TS_dig"] = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)   # 논문 TS_eff 대응
X["VS_dig"] = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)   # 논문 VS_eff 대응
X["VSTS_dig"] = X.VS_dig / X.TS_dig                              # ★§2.11 추가
X["VFA_ALK"] = M.VFA_ALK_A                                       # 논문 H₂S 대응
X["dewater"] = M.dewater_tpd                                     # 논문 swell 대리
X["pH_dig"] = M[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
DENOM_FEATS = ["SRT_d"] + [f"Qin_{c}" for c in KS]                # MY 분모 성분 포함
FEATS = list(X.columns)
X = X.ffill(limit=7)

print("=== 데이터 개요 ===")
print(f"  기간 {F.min().date()}~{F.max().date()} ({len(F)}일), MY 관측 {int(MY.notna().sum())}일")
print(f"  MY 평균 {MY.mean():.3f}, 중앙 {MY.median():.3f}, IQR {MY.quantile(.75)-MY.quantile(.25):.3f}, "
      f"범위 {MY.min():.3f}~{MY.max():.3f} ㎥CH₄/kg VS")
print(f"  (논문 D_main: 평균 0.5, IQR 0.1, 범위 0.3~1.4)")
OUT["MY_stats"] = {"mean": round(float(MY.mean()), 3), "median": round(float(MY.median()), 3),
                   "iqr": round(float(MY.quantile(.75) - MY.quantile(.25)), 3),
                   "min": round(float(MY.min()), 3), "max": round(float(MY.max()), 3),
                   "n": int(MY.notna().sum()),
                   "paper": {"mean": 0.5, "iqr": 0.1, "min": 0.3, "max": 1.4}}

# ================================================================ 2.2 데이터 분석
print("\n=== 정상성·시간의존성 진단 (논문 §2.2) ===")
from statsmodels.tsa.stattools import kpss, acf

mi = MY.interpolate(limit=7).dropna()
kp = kpss(mi.values, regression="c", nlags="auto")
a = acf(mi.values, nlags=40, fft=True)
OUT["stationarity"] = {"kpss_stat": round(float(kp[0]), 3), "kpss_p": round(float(kp[1]), 4),
                       "nonstationary": bool(kp[1] < 0.05),
                       "acf1": round(float(a[1]), 4), "acf7": round(float(a[7]), 4),
                       "acf30": round(float(a[30]), 4)}
# 대조: 메탄 발생량 자체의 ACF (§2.10 타깃)
ci = CH4.interpolate(limit=7).dropna()
ac = acf(ci.values, nlags=40, fft=True)
OUT["stationarity"]["acf1_CH4_volume"] = round(float(ac[1]), 4)
print(f"  KPSS 통계량 {kp[0]:.3f}, p {kp[1]:.4f} → "
      f"{'비정상(non-stationary)' if kp[1] < 0.05 else '정상'}  (논문 5.74, p<0.01 비정상)")
print(f"  MY의 ACF: lag1 {a[1]:.3f}, lag7 {a[7]:.3f}, lag30 {a[30]:.3f}")
print(f"  대조 — 메탄 발생량(㎥/d)의 ACF lag1 {ac[1]:.3f}  ← §2.10이 예측하던 양")
print(f"  → 타깃을 수율로 바꾸자 자기상관이 {ac[1]:.3f} → {a[1]:.3f} 로 떨어진다.")

# RM-ANOVA 대용: 월·요일 그룹 간 평균 차이 (Kruskal–Wallis, 정규성 미가정)
print("\n=== 시간 그룹별 변동 (논문 RM-ANOVA 대응, 비모수 Kruskal–Wallis) ===")
anova = {}
for gname, g in [("월", mi.index.month), ("요일", mi.index.dayofweek)]:
    groups = [mi.values[g == k] for k in np.unique(g)]
    h, p = stats.kruskal(*groups)
    anova[gname] = {"H": round(float(h), 2), "p": float(f"{p:.3g}"), "sig": bool(p < 0.05)}
    print(f"  MY ~ {gname}: H={h:.2f}, p={p:.3g} → {'유의' if p < 0.05 else '무의미'}")
for nm in ["Qin_manure", "temp"]:
    v = X[nm].reindex(mi.index).dropna()
    h, p = stats.kruskal(*[v.values[v.index.month == k] for k in range(1, 13)])
    anova[f"{nm}~월"] = {"H": round(float(h), 2), "p": float(f"{p:.3g}"), "sig": bool(p < 0.05)}
    print(f"  {nm} ~ 월: H={h:.2f}, p={p:.3g}")
OUT["anova"] = anova
USE_MONTH = anova["월"]["sig"]
USE_DOW = anova["요일"]["sig"]
print(f"  → 시간 임베딩에 월 {'포함' if USE_MONTH else '제외'}, 요일 {'포함' if USE_DOW else '제외'}")

# TLCC (Kendall τ) — 논문 Fig.3b
print("\n=== 시차 교차상관 TLCC (Kendall τ, 오프셋 −SRT~+SRT) ===")
tl = {}
for nm in FEATS:
    best, rows = None, []
    for off in range(-int(SRT), int(SRT) + 1):
        d = pd.concat([X[nm].shift(off).rename("x"), MY.rename("y")], axis=1).dropna()
        if len(d) < 100:
            continue
        t, p = stats.kendalltau(d.x.values, d.y.values)
        rows.append({"offset": off, "tau": round(float(t), 4), "p": float(f"{p:.3g}")})
        if best is None or abs(t) > abs(best["tau"]):
            best = rows[-1]
    tl[nm] = {"scan": rows, "best": best}
    print(f"  {nm:16s} 최대 |τ| {best['tau']:+.4f} @ 오프셋 {best['offset']:+3d}일  (p {best['p']})")
OUT["tlcc"] = tl


# ================================================================ 모델
class RevIN(nn.Module):
    """인스턴스 정규화·역정규화 (논문 §2.4.3)."""

    def __init__(self, n):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n)); self.b = nn.Parameter(torch.zeros(n))

    def norm(self, x):
        self.mu = x.mean(1, keepdim=True)
        self.sd = x.std(1, keepdim=True) + 1e-5
        return self.g * (x - self.mu) / self.sd + self.b

    def denorm(self, y, ch=-1):
        return self.sd[:, :, ch] * (y - self.b[ch]) / self.g[ch] + self.mu[:, :, ch]


class Proposed(nn.Module):
    """패칭(패치 내 변수 혼합) + RevIN + 시간 임베딩 + Transformer, 다단계 직접예측."""

    def __init__(self, V, L, H, d=32, nh=4, ne=2, pl=4, st=2,
                 patch=True, revin=True, temporal=True, n_time=0):
        super().__init__()
        self.V, self.L, self.H = V, L, H
        self.use_patch, self.use_revin, self.use_temp = patch, revin, temporal
        self.rev = RevIN(V) if revin else None
        if patch:
            self.np_ = (L - pl) // st + 1
            self.pl, self.st = pl, st
            self.val = nn.Linear(pl * V, d)          # 패치 내 변수 혼합
        else:
            self.np_ = L
            self.val = nn.Linear(V, d)
        pe = torch.zeros(self.np_, d)
        pos = torch.arange(self.np_).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.temb = nn.Linear(n_time, d) if (temporal and n_time) else None
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nh, d * 4, .1, batch_first=True, norm_first=True), ne)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(self.np_ * d, H))
        # RevIN 은 창 내부 통계로 되돌리므로, 입력에 적용한 전역 표준화까지 되돌려야
        # 예측값이 원 MY 스케일로 나온다.
        self.register_buffer("gsc", torch.tensor([0.0, 1.0]))

    def set_scale(self, mu, sd):
        self.gsc = torch.tensor([float(mu), float(sd)])

    def forward(self, x, tf=None):                   # x:(B,L,V)  tf:(B,n_time)
        z = self.rev.norm(x) if self.use_revin else x
        if self.use_patch:
            p = torch.stack([z[:, i * self.st: i * self.st + self.pl].reshape(z.size(0), -1)
                             for i in range(self.np_)], 1)
        else:
            p = z
        e = self.val(p) + self.pe
        if self.temb is not None and tf is not None:
            e = e + self.temb(tf).unsqueeze(1)
        y = self.head(self.enc(e))
        if self.use_revin:
            y = self.rev.denorm(y, -1) * self.gsc[1] + self.gsc[0]
        return y


class Seq(nn.Module):
    def __init__(self, cell, V, H, d=64, n=2):
        super().__init__()
        self.r = cell(V, d, n, batch_first=True, dropout=.1)
        self.head = nn.Linear(d, H)

    def forward(self, x, tf=None):
        return self.head(self.r(x)[0][:, -1])


class Vanilla(nn.Module):
    def __init__(self, V, L, H, d=32, nh=4, ne=2):
        super().__init__()
        self.emb = nn.Linear(V, d)
        pe = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nh, d * 4, .1, batch_first=True, norm_first=True), ne)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(L * d, H))

    def forward(self, x, tf=None):
        return self.head(self.enc(self.emb(x) + self.pe))


# ================================================================ 표본 구성
def build(L, H, feats):
    """(N,L,V) 입력 / (N,H) 타깃. MY가 실제 관측된 스텝만 손실·채점에 쓴다."""
    A = pd.concat([X[feats], MY.rename("MY")], axis=1)
    A["MY"] = A["MY"].ffill(limit=3)                       # 입력용만 보간
    Av = A.values.astype("float32")
    yv = MY.values.astype("float32")
    tfe = []
    if USE_MONTH:
        tfe += [np.sin(2 * np.pi * F.month / 12), np.cos(2 * np.pi * F.month / 12)]
    if USE_DOW:
        tfe += [np.sin(2 * np.pi * F.dayofweek / 7), np.cos(2 * np.pi * F.dayofweek / 7)]
    TF = np.stack(tfe, 1).astype("float32") if tfe else np.zeros((len(F), 0), "float32")
    xs, ys, ms, ts, ds = [], [], [], [], []
    for t in range(L - 1, len(F) - H):
        win = Av[t - L + 1: t + 1]
        tgt = yv[t + 1: t + 1 + H]
        if not np.isfinite(win).all() or not np.isfinite(tgt).any():
            continue
        xs.append(win); ys.append(np.nan_to_num(tgt))
        ms.append(np.isfinite(tgt)); ts.append(TF[t]); ds.append(F[t + 1])
    return (np.array(xs, "float32"), np.array(ys, "float32"), np.array(ms),
            np.array(ts, "float32"), pd.DatetimeIndex(ds))


def split(ds):
    a = np.asarray(ds)
    f = lambda lo, hi: (a >= np.datetime64(lo)) & (a <= np.datetime64(hi))
    return f(*TR), f(*VA), f(*TE)


def metr(t, p, m):
    t, p = t[m], p[m]
    return {"MSE": round(float(((t - p) ** 2).mean()), 6),
            "MAE": round(float(np.abs(t - p).mean()), 5),
            "R2": round(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4),
            "MAPE": round(float(np.abs((t - p) / t).mean() * 100), 2), "n": int(m.sum())}


def run(kind, L, H, feats, seed, cfg=None, epochs=150, pat=20):
    Xa, ya, ma, ta, ds = build(L, H, feats)
    tr, va, te = split(ds)
    mu = Xa[tr].reshape(-1, Xa.shape[2]).mean(0); sd = Xa[tr].reshape(-1, Xa.shape[2]).std(0) + 1e-8
    Xn = (Xa - mu) / sd
    torch.manual_seed(seed); np.random.seed(seed)
    V, nt = Xa.shape[2], ta.shape[1]
    cfg = cfg or {}
    m = ({"proposed": lambda: Proposed(V, L, H, n_time=nt, **cfg),
          "gru": lambda: Seq(nn.GRU, V, H), "lstm": lambda: Seq(nn.LSTM, V, H),
          "vanilla": lambda: Vanilla(V, L, H)}[kind])()
    if kind == "proposed":
        m.set_scale(mu[-1], sd[-1])
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
    T = lambda z, k: torch.tensor(z[k])
    xt, yt, mt, tt = T(Xn, tr), T(ya, tr), T(ma, tr), T(ta, tr)
    xv, yv_, mv, tv = T(Xn, va), T(ya, va), T(ma, va), T(ta, va)
    best, bs, bad = 1e18, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            pr = m(xt[b], tt[b])
            loss = (((pr - yt[b]) ** 2) * mt[b]).sum() / mt[b].sum().clamp(min=1)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pr = m(xv, tv)
            v = float((((pr - yv_) ** 2) * mv).sum() / mv.sum().clamp(min=1))
        if v < best - 1e-8:
            best, bs, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= pat:
                break
    m.load_state_dict(bs); m.eval()
    out = {}
    with torch.no_grad():
        for nm, k in [("valid", va), ("test", te)]:
            p = m(T(Xn, k), T(ta, k)).numpy()
            out[nm] = metr(ya[k], p, ma[k])
            out[nm + "_by_step"] = [metr(ya[k][:, s:s + 1], p[:, s:s + 1], ma[k][:, s:s + 1])["MSE"]
                                    for s in range(H)]
    out["params"] = int(sum(q.numel() for q in m.parameters()))
    return out


def persist(L, H, feats):
    """마지막 관측 MY를 H스텝 전체에 유지."""
    Xa, ya, ma, ta, ds = build(L, H, feats)
    tr, va, te = split(ds)
    p = np.repeat(Xa[:, -1, -1][:, None], H, 1)
    return {nm: metr(ya[k], p[k], ma[k]) for nm, k in [("valid", va), ("test", te)]}


# ================================================================ 1) 문맥길이 선택
print("\n=== 문맥길이 선택 (체류시간의 배수, 사고학습 기준·H=7) ===")
ctx_res = {}
for nm, L in CTX.items():
    r = [run("proposed", L, 7, FEATS, s) for s in SEEDS]
    mse = float(np.mean([x["valid"]["MSE"] for x in r]))
    ctx_res[nm] = {"L": L, "valid_MSE": round(mse, 6),
                   "test_MSE": round(float(np.mean([x["test"]["MSE"] for x in r])), 6)}
    print(f"  {nm:8s} (L={L:2d}일)  사고학습 MSE {mse:.6f}")
BEST_L = CTX[min(ctx_res, key=lambda k: ctx_res[k]["valid_MSE"])]
OUT["context"] = {"scan": ctx_res, "selected_L": BEST_L}
print(f"  → 선택 L = {BEST_L}일")

# ================================================================ 2) 모델 비교
print(f"\n=== 모델 비교 (L={BEST_L}, 다단계 직접예측) ===")
res = {}
for tag, feats in [("논문충실(Q_in 포함)", FEATS),
                   ("분모배제(Q_in·SRT_d 제외)", [f for f in FEATS if f not in DENOM_FEATS])]:
    res[tag] = {}
    print(f"\n  [{tag}]  변수 {len(feats)}개")
    for H in HORIZONS:
        row = {"persistence": persist(BEST_L, H, feats)}
        for kind, label in [("gru", "RNN(GRU)"), ("lstm", "LSTM"),
                            ("vanilla", "Transformer"), ("proposed", "제안모델")]:
            rr = [run(kind, BEST_L, H, feats, s) for s in SEEDS]
            row[label] = {sp: {k: round(float(np.mean([x[sp][k] for x in rr])), 6)
                               for k in ["MSE", "MAE", "R2", "MAPE"]} for sp in ["valid", "test"]}
            row[label]["test"]["n"] = rr[0]["test"]["n"]
            row[label]["test"]["R2_sd"] = round(float(np.std([x["test"]["R2"] for x in rr])), 4)
            row[label]["params"] = rr[0]["params"]
        pm = row["persistence"]["test"]["MSE"]
        row["skill"] = {k: round(1 - v["test"]["MSE"] / pm, 4)
                        for k, v in row.items() if isinstance(v, dict) and "test" in v}
        res[tag][f"H{H}"] = row
        s_ = " ".join(f"{k.split('(')[0]} {v:+.3f}" for k, v in row["skill"].items()
                      if k != "persistence")
        print(f"    H={H:2d}일  persistence R² {row['persistence']['test']['R2']:+.3f} | "
              f"제안 R² {row['제안모델']['test']['R2']:+.3f} | skill {s_}")
OUT["models"] = res

# ================================================================ 3) 절제 연구
print(f"\n=== 절제 연구 (L={BEST_L}, H=7, 논문충실 변수) ===")
abl = {}
for nm, cfg in [("제안모델", {}), ("w/o Patching", {"patch": False}),
                ("w/o Norm", {"revin": False}), ("w/o Temporal", {"temporal": False})]:
    rr = [run("proposed", BEST_L, 7, FEATS, s, cfg) for s in SEEDS]
    abl[nm] = {"MSE": round(float(np.mean([x["test"]["MSE"] for x in rr])), 6),
               "MAE": round(float(np.mean([x["test"]["MAE"] for x in rr])), 5),
               "R2": round(float(np.mean([x["test"]["R2"] for x in rr])), 4)}
    print(f"  {nm:16s} MSE {abl[nm]['MSE']:.6f}  MAE {abl[nm]['MAE']:.5f}  R² {abl[nm]['R2']:+.4f}")
base = abl["제안모델"]["MSE"]
for nm in abl:
    abl[nm]["MSE_increase_pct"] = round(100 * (abl[nm]["MSE"] - base) / base, 2)
OUT["ablation"] = abl

# ================================================================ 4) 열화 점수
prop = {h: res["논문충실(Q_in 포함)"][h]["제안모델"]["test"]["MSE"] for h in res["논문충실(Q_in 포함)"]}
deg = {}
for lab in ["RNN(GRU)", "LSTM", "Transformer"]:
    v = [100 * (res["논문충실(Q_in 포함)"][h][lab]["test"]["MSE"] - prop[h]) / prop[h] for h in prop]
    deg[lab] = round(float(np.mean(v)), 2)
OUT["degradation"] = deg
print("\n=== 열화 점수 Δ (제안모델 대비 MSE 증가율 평균, 논문 식) ===")
for k, v in deg.items():
    print(f"  {k:14s} {v:+7.2f}%")

Path("outputs").mkdir(exist_ok=True)
json.dump(OUT, open("outputs/mtsf_results.json", "w", encoding="utf-8"), ensure_ascii=False)
print("\n저장: outputs/mtsf_results.json")
