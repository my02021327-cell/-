# -*- coding: utf-8 -*-
"""Han et al. (CEJ 513:163064, 2025) iTransformer 방법론의 영천 적용 및 이식성 검정.

■ 원 논문
   Y. Han et al., "Time series prediction of anaerobic digestion yield and carbon
   emissions from food waste based on iTransformer model", Chem. Eng. J. 513 (2025) 163064.
   음식물류폐기물 AD 시설 6개월(2020.1~6) 일별 자료, 15개 변수.
   iTransformer(= Inverted Transformer + Temporal Embedding + Dynamic Attention
   + Multi-Scale Feature Extraction)로 CH₄ 생산량과 CO₂ 배출량을 예측.
   보고 성능: R² 0.9949, MSE 3946.96, 정확도 98.55% (GRU/ARIMA/LSTM/Transformer 대비 우위)

■ 영천 적용 시 지켜야 할 네 가지 제약 (사용자 지정)
   1. CO₂ 배출량은 측정하지 않는다 → 타깃은 CH₄ 단일. emissions 변수 제외.
   2. 영천 데이터에 없는 피처는 채택하지 않는다 → 아래 FEATURE_MAP 의 '채택' 열 참조.
   3. 영천에서 부적합하다면 그 이유를 다른 문헌 근거로 제시한다 → §5 진단.
   4. 영천은 가수분해조가 없고 산생성조 + 혐기소화조 2조 구성이다.
      논문의 h* 블록은 '가수분해·산성화 통합조' 측정값이므로 영천의 acid_* 와
      1:1 대응이 아니다. 대응시키되 그 차이를 명시한다.

■ 논문이 하지 않은 것을 여기서는 한다
   원 논문에는 **persistence(전일값 유지) 기준선이 없다.** AD 가스계열은 자기상관이
   극히 높아(영천 ACF₁ = 0.906) 아무것도 학습하지 않아도 R² 0.88~0.90 이 나온다.
   따라서 R² 0.99 라는 수치만으로는 모델이 무엇을 학습했는지 알 수 없다.
   여기서는 (a) persistence 를 항상 함께 싣고, (b) skill score = 1 − MSE/MSE_persist 로
   순수 이득을 표기하며, (c) 타깃 자기이력을 입력에서 뺀 설정(B)을 따로 돌린다.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# ================================================================ 변수 대응표
# (논문 변수, 의미, 영천 컬럼 or None, 채택여부, 비고)
FEATURE_MAP = [
    ("hpH", "가수분해·산성화조 pH", "acid_pH", True, "영천은 가수분해조가 없어 산생성조 pH로 대응"),
    ("hVFA", "가수분해·산성화조 VFA", None, False, "영천 산생성조 VFA 미측정"),
    ("hTS", "가수분해·산성화조 TS", "acid_TS_pct", True, "산생성조 TS"),
    ("hVS", "가수분해·산성화조 VS", "acid_VS_pct", True, "산생성조 VS"),
    ("hAM", "가수분해·산성화조 암모니아", "acid_NH3N_mgL", False, "커버리지 7.8%, 2022~23년 사실상 전무"),
    ("hCOD", "가수분해·산성화조 COD", "acid_CODcr_mgL", True, "산생성조 CODcr"),
    ("adpH", "소화조 pH", "dig_pH", True, "A·B 평균"),
    ("adVFA", "소화조 VFA", "VFA_A_mgL", True, "A계열 (B는 2023년 오염)"),
    ("adALK", "소화조 알칼리도", "ALK_A_mgL", True, "A계열 (B는 2023년 VFA값으로 오염)"),
    ("adTS", "소화조 TS", "dig_TS", True, "A·B 평균"),
    ("adVS", "소화조 VS", "dig_VS", True, "A·B 평균"),
    ("adAM", "소화조 암모니아", "NH3N_A_mgL", False, "커버리지 11.0%, 2022~23년 사실상 전무"),
    ("adCOD", "소화조 COD", "dig_CODcr", True, "A·B 평균"),
    ("tf", "총 투입량", "feed_AB_tpd", True, "소화조 투입량"),
    ("production", "메탄 생산량", "CH4_m3d", True, "★타깃"),
    ("emissions", "CO₂ 배출량", None, False, "제약 1 — 영천은 CO₂ 배출량 미측정"),
]
EXOG = [m[2] for m in FEATURE_MAP if m[3] and m[2] not in (None, "CH4_m3d")]
TARGET = "CH4_m3d"

TR = ("2018-01-01", "2021-12-31")
VA = ("2022-01-01", "2022-12-31")
TE_ = ("2023-01-01", "2023-09-17")

# ================================================================ 데이터
M = pd.read_csv("data/영천BGP_MASTER_2018-2023.csv", parse_dates=["date"]).set_index("date")
M = M[M.index <= "2023-09-17"]
M["dig_pH"] = M[["dig_pH_A", "dig_pH_B"]].mean(axis=1)
M["dig_TS"] = M[["dig_TS_A_pct", "dig_TS_B_pct"]].mean(axis=1)
M["dig_VS"] = M[["dig_VS_A_pct", "dig_VS_B_pct"]].mean(axis=1)
M["dig_CODcr"] = M[["dig_CODcr_A_mgL", "dig_CODcr_B_mgL"]].mean(axis=1)

# 실험실 계측은 주 4회 수준의 계단형이므로 결측일은 직전값 유지(최대 7일).
# 타깃은 절대 채우지 않는다 — 채운 값을 채점하면 순환이다.
Xdf = M[EXOG].ffill(limit=7)
ffill_frac = float(M[EXOG].isna().mean().mean())
yv = M[TARGET]

OUT = {"feature_map": [{"paper": a, "meaning": b, "yeongcheon": c, "adopted": d, "note": e}
                       for a, b, c, d, e in FEATURE_MAP],
       "n_exog": len(EXOG), "ffill_frac": round(ffill_frac, 3),
       "target_obs": int(yv.notna().sum()), "n_days": int(len(M))}


def windows(L, H, use_target_hist):
    """(N,L,V) 입력, (N,) 타깃, 날짜. 타깃 관측일만 표본으로 삼는다."""
    cols = EXOG + ([TARGET] if use_target_hist else [])
    A = pd.concat([Xdf, yv.ffill(limit=7).rename(TARGET)], axis=1)[cols].values
    yt = yv.values
    idx = M.index
    xs, ys, ds = [], [], []
    for t in range(L - 1, len(M) - H):
        j = t + H
        if not np.isfinite(yt[j]):
            continue
        w = A[t - L + 1: t + 1]
        if not np.isfinite(w).all():
            continue
        xs.append(w); ys.append(yt[j]); ds.append(idx[j])
    return np.array(xs, "float32"), np.array(ys, "float32"), pd.DatetimeIndex(ds)


def split(ds):
    a = np.asarray(ds)
    f = lambda lo, hi: (a >= np.datetime64(lo)) & (a <= np.datetime64(hi))
    return f(*TR), f(*VA), f(*TE_)


def metrics(t, p):
    return {"R2": round(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4),
            "RMSE": round(float(np.sqrt(((t - p) ** 2).mean())), 1),
            "MSE": round(float(((t - p) ** 2).mean()), 1),
            "MAPE": round(float(np.abs((t - p) / t).mean() * 100), 2),
            "ACC": round(float(100 * (1 - np.abs((t - p) / t).mean())), 2), "n": int(len(t))}


# ================================================================ 모델
class TemporalEmbedding(nn.Module):
    """TE(t) = sin(t / 10000^(2i/d)) — 논문 식 (2). 룩백 창의 시점 정보를 주입."""

    def __init__(self, L):
        super().__init__()
        i = torch.arange(L).float()
        pe = torch.sin(i.unsqueeze(1) / (10000 ** (2 * torch.arange(1).float() / max(L, 2))))
        self.register_buffer("pe", pe.squeeze(1))

    def forward(self, x):                       # x: (B,L,V)
        return x + self.pe.view(1, -1, 1) * 0.0 + self.pe.view(1, -1, 1)


class DynamicAttention(nn.Module):
    """softmax((QKᵀ + T)/√d)·V — 논문 식 (3). T는 시점 관련 학습 가중행렬."""

    def __init__(self, d, h, n_tok):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.T = nn.Parameter(torch.zeros(h, n_tok, n_tok))

    def forward(self, x):                       # x: (B,N,d)
        B, N, d = x.shape
        sh = lambda z: z.view(B, N, self.h, self.dk).transpose(1, 2)
        a = (sh(self.q(x)) @ sh(self.k(x)).transpose(-2, -1)) / self.dk ** .5 + self.T[:, :N, :N]
        z = (a.softmax(-1) @ sh(self.v(x))).transpose(1, 2).reshape(B, N, d)
        return self.o(z)


class MSFE(nn.Module):
    """Concat(Conv_k1, Conv_k2, …) — 논문 식 (4). 시간축 다중 스케일 특징 추출."""

    def __init__(self, d, ks=(3, 5, 7)):
        super().__init__()
        self.c = nn.ModuleList([nn.Conv1d(d, d // len(ks), k, padding=k // 2) for k in ks])
        self.p = nn.Linear((d // len(ks)) * len(ks), d)

    def forward(self, x):                       # x: (B,N,d)
        z = x.transpose(1, 2)
        return self.p(torch.cat([c(z) for c in self.c], 1).transpose(1, 2))


class EncLayer(nn.Module):
    def __init__(self, d, h, n_tok, ff):
        super().__init__()
        self.at = DynamicAttention(d, h, n_tok)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        x = self.n1(x + self.at(x))
        return self.n2(x + self.ff(x))


class ITransformer(nn.Module):
    """변수를 토큰으로 뒤집는 iTransformer. 논문 Fig.2(b) 구조."""

    def __init__(self, V, L, d=512, h=8, n_enc=2, n_dec=1, ff=None, drop=.1):
        super().__init__()
        ff = ff or d * 2
        self.te = TemporalEmbedding(L)
        self.emb = nn.Linear(L, d)                       # 변수별 전체 룩백 → 토큰
        self.drop = nn.Dropout(drop)
        self.enc = nn.ModuleList([EncLayer(d, h, V, ff) for _ in range(n_enc)])
        self.msfe = MSFE(d)
        self.dec = nn.ModuleList([EncLayer(d, h, V, ff) for _ in range(n_dec)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(V * d, 1))

    def forward(self, x):                                # (B,L,V)
        z = self.emb(self.te(x).transpose(1, 2))         # (B,V,d)
        z = self.drop(z)
        for l in self.enc:
            z = l(z)
        z = z + self.msfe(z)
        for l in self.dec:
            z = l(z)
        return self.head(z).squeeze(-1)


class VanillaTransformer(nn.Module):
    """시점을 토큰으로 쓰는 전통 Transformer 인코더 (논문 비교군)."""

    def __init__(self, V, L, d=128, h=8, n=2):
        super().__init__()
        self.emb = nn.Linear(V, d)
        pe = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, h, d * 2, .1, batch_first=True, norm_first=True), n)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.enc(self.emb(x) + self.pe)[:, -1]).squeeze(-1)


class RNNBase(nn.Module):
    def __init__(self, cell, V, d=128, n=2):
        super().__init__()
        self.r = cell(V, d, n, batch_first=True, dropout=.1)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.r(x)[0][:, -1]).squeeze(-1)


def build(name, V, L, cfg):
    if name == "iTransformer":
        return ITransformer(V, L, **cfg)
    if name == "Transformer":
        return VanillaTransformer(V, L)
    if name == "LSTM":
        return RNNBase(nn.LSTM, V)
    if name == "GRU":
        return RNNBase(nn.GRU, V)
    raise ValueError(name)


def train_eval(name, Xa, ya, ds, cfg, seed, epochs=150, patience=20, lr=1e-3, bs=32):
    tr, va, te = split(ds)
    mu, sd = Xa[tr].reshape(-1, Xa.shape[2]).mean(0), Xa[tr].reshape(-1, Xa.shape[2]).std(0) + 1e-8
    Xn = (Xa - mu) / sd
    ym, ys_ = ya[tr].mean(), ya[tr].std()
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(name, Xa.shape[2], Xa.shape[1], cfg)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.tensor(Xn[tr]); yt = torch.tensor((ya[tr] - ym) / ys_)
    xv = torch.tensor(Xn[va]); yvv = torch.tensor((ya[va] - ym) / ys_)
    best, bstate, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.mse_loss(m(xt[b]), yt[b]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(m(xv), yvv).item()
        if v < best - 1e-5:
            best, bstate, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(bstate); m.eval()
    out = {}
    with torch.no_grad():
        for nm, msk in [("valid", va), ("test", te)]:
            p = m(torch.tensor(Xn[msk])).numpy() * ys_ + ym
            out[nm] = metrics(ya[msk], p)
    out["params"] = int(sum(p.numel() for p in m.parameters()))
    out["epochs"] = ep + 1
    return out


def persistence(ds, H):
    """마지막 관측 CH₄ 를 그대로 H일 뒤 예측값으로 쓴다."""
    s = yv.ffill(limit=7).shift(H)
    d = pd.concat([yv.rename("y"), s.rename("p")], axis=1).loc[ds].dropna()
    return {nm: metrics(d.loc[a:b, "y"].values, d.loc[a:b, "p"].values)
            for nm, (a, b) in [("valid", VA), ("test", TE_)]}


def arima(ds, H):
    """ARIMA(2,1,2) — 관측일만 등간격으로 간주하고 롤링 원스텝(H스텝) 예측."""
    from statsmodels.tsa.arima.model import ARIMA
    s = yv.dropna()
    hist = list(s.loc[:TR[1]].values)
    pred, dts = [], []
    for dt, val in s.loc[VA[0]:].items():
        try:
            f = ARIMA(hist, order=(2, 1, 2)).fit().forecast(H)[-1]
        except Exception:
            f = hist[-1]
        pred.append(f); dts.append(dt); hist.append(val)
    r = pd.DataFrame({"y": s.loc[VA[0]:].values, "p": pred}, index=dts)
    return {nm: metrics(r.loc[a:b, "y"].values, r.loc[a:b, "p"].values)
            for nm, (a, b) in [("valid", VA), ("test", TE_)]}


# ================================================================ 실행
print("=== 변수 대응 (제약 2·4 적용) ===")
for a, b, c, d, e in FEATURE_MAP:
    print(f"  {a:11s} {b:22s} {'○' if d else '✗'} {str(c or '—'):18s} {e}")
print(f"  → 논문 15개 중 채택 {len(EXOG)}개(예측변수) + 타깃 1개, 제외 4개")
print(f"  결측 보간(직전값 유지) 비율 {ffill_frac*100:.1f}%, 타깃 관측 {int(yv.notna().sum())}일")

CFG_PAPER = dict(d=512, h=8, n_enc=2, n_dec=1)     # 논문 설정 (hidden 512, head 8)
CFG_SMALL = dict(d=64, h=8, n_enc=2, n_dec=1)      # 영천 표본 수에 맞춘 축소 설정
SEEDS = [0, 1, 2]
RES = {}

for setting, use_hist in [("A_논문동일_자기이력포함", True), ("B_외생변수만_자기이력제외", False)]:
    for H in [1, 7]:
        key = f"{setting}|H{H}"
        Xa, ya, ds = windows(L := 30, H, use_hist)
        tr, va, te = split(ds)
        print(f"\n=== {key}  창길이 {L}일, 변수 {Xa.shape[2]}개, "
              f"표본 학습 {tr.sum()} / 사고학습 {va.sum()} / 평가 {te.sum()} ===")
        r = {"n_var": int(Xa.shape[2]), "n": [int(tr.sum()), int(va.sum()), int(te.sum())]}
        r["persistence"] = persistence(ds, H)
        print(f"  {'persistence':16s} 사고학습 R² {r['persistence']['valid']['R2']:+.4f}  "
              f"평가 R² {r['persistence']['test']['R2']:+.4f}  MAPE {r['persistence']['test']['MAPE']:.2f}%")
        r["ARIMA"] = arima(ds, H)
        print(f"  {'ARIMA(2,1,2)':16s} 사고학습 R² {r['ARIMA']['valid']['R2']:+.4f}  "
              f"평가 R² {r['ARIMA']['test']['R2']:+.4f}  MAPE {r['ARIMA']['test']['MAPE']:.2f}%")
        for name in ["GRU", "LSTM", "Transformer", "iTransformer"]:
            cfgs = [("논문설정", CFG_PAPER), ("축소설정", CFG_SMALL)] if name == "iTransformer" \
                else [("", {})]
            for tag, cfg in cfgs:
                t0 = time.time()
                runs = [train_eval(name, Xa, ya, ds, cfg, s) for s in SEEDS]
                lbl = f"{name}{('·' + tag) if tag else ''}"
                agg = {}
                for sp in ["valid", "test"]:
                    agg[sp] = {k: round(float(np.mean([x[sp][k] for x in runs])), 4)
                               for k in ["R2", "RMSE", "MSE", "MAPE", "ACC"]}
                    agg[sp]["R2_sd"] = round(float(np.std([x[sp]["R2"] for x in runs])), 4)
                    agg[sp]["n"] = runs[0][sp]["n"]
                agg["params"] = runs[0]["params"]
                agg["epochs"] = round(float(np.mean([x["epochs"] for x in runs])), 1)
                r[lbl] = agg
                print(f"  {lbl:16s} 사고학습 R² {agg['valid']['R2']:+.4f}  "
                      f"평가 R² {agg['test']['R2']:+.4f} (±{agg['test']['R2_sd']:.3f})  "
                      f"MAPE {agg['test']['MAPE']:.2f}%  "
                      f"파라미터 {agg['params']:,}  {time.time()-t0:.0f}s")
        # skill score (평가구간, persistence 대비)
        pm = r["persistence"]["test"]["MSE"]
        r["skill"] = {k: round(1 - (v["test"]["MSE"] if "test" in v else 1e9) / pm, 4)
                      for k, v in r.items() if isinstance(v, dict) and "test" in v}
        print("  skill score (1 − MSE/MSE_persistence, 평가구간):")
        for k, v in sorted(r["skill"].items(), key=lambda x: -x[1]):
            print(f"     {k:22s} {v:+.4f}{'   ← persistence 미달' if v < 0 else ''}")
        RES[key] = r

OUT["results"] = RES
OUT["persistence_context"] = {
    "acf1": round(float(yv.dropna().autocorr(1)), 4),
    "cv_pct": round(float(yv.std() / yv.mean() * 100), 1)}
Path("outputs").mkdir(exist_ok=True)
with open("outputs/itransformer_results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n저장: outputs/itransformer_results.json")
