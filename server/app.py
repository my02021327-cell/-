"""
BioGuard SCADA 백엔드 — 업로드된 HMI 프론트엔드를 실제 모델에 연결한다

실행
    python -m src.bgp.train_bands      # 1) 구간별 모델 평가·채택
    python -m src.bgp.serve_models     # 2) 채택 모델을 전체 자료로 재적합·저장
    python -m server.app               # 3) 서버 기동 (기본 http://127.0.0.1:8000)

설계 원칙
  · 서버는 모델을 새로 학습하지 않는다. `outputs/models/` 산출물만 읽는다.
  · `asof` 질의로 원점을 옮길 수 있다. 과거 임의 시점에서 화면을 재현해
    "그날 이 화면이 무엇을 말했을 것인가"를 검증할 수 있게 하기 위한 것이다.
  · 예측구간은 가정한 분포가 아니라 **폴드밖 잔차의 경험 분위수**로 만든다.
  · 구간별 모델이 naive 를 못 이기면(RMSSE ≥ 100 %) 그 사실을 응답에 실어 보내고
    제어 권고를 생성하지 않는다. 화면에서 근거 없는 지시가 나가지 않게 한다.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.bgp import config as C
from src.bgp.advisor import diagnose, recommend

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
BANDS_DIR = os.path.join(C.OUT_DIR, "bands")

app = FastAPI(title="BioGuard SCADA API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# ══════════════════════════════════════════════════════════════════════════════
# 상태 적재
# ══════════════════════════════════════════════════════════════════════════════
class Store:
    def __init__(self):
        self.ready = False
        self.reason = "not loaded"
        self.df = self.X = None
        self.models: dict = {}
        self.manifest: dict = {}
        self.summary: dict = {}
        self.resid: dict = {}

    def load(self):
        mpath = os.path.join(C.MODEL_DIR, "manifest.json")
        ppath = os.path.join(C.MODEL_DIR, "panel.joblib")
        if not (os.path.exists(mpath) and os.path.exists(ppath)):
            self.reason = ("모델 산출물이 없다. `python -m src.bgp.train_bands` 와 "
                           "`python -m src.bgp.serve_models` 를 먼저 실행할 것.")
            return self
        self.manifest = json.load(open(mpath, encoding="utf-8"))
        panel = joblib.load(ppath)
        self.df, self.X = panel["df"], panel["X"]
        for name, meta in self.manifest.get("bands", {}).items():
            p = os.path.join(C.ROOT, meta["path"])
            if os.path.exists(p):
                self.models[name] = joblib.load(p)
        sp = os.path.join(BANDS_DIR, "summary.json")
        if os.path.exists(sp):
            self.summary = json.load(open(sp, encoding="utf-8"))
        self._load_residuals()
        self.ready = bool(self.models)
        self.reason = "ok" if self.ready else "적재된 구간 모델이 없다"
        return self

    def _load_residuals(self):
        """폴드밖 잔차 → 지평별 경험 분위수. 예측구간의 근거."""
        for name in C.HORIZON_BANDS:
            b = name[0]
            f = os.path.join(BANDS_DIR, f"oof_{b}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f, encoding="utf-8-sig")
            pick = self.manifest.get("bands", {}).get(b, {}).get("model")
            col = f"pred_{pick}" if pick and f"pred_{pick}" in d.columns else None
            if col is None:
                cands = [c for c in d.columns if c.startswith("pred_")
                         and c not in ("pred_Naive",)]
                if not cands:
                    continue
                col = cands[0]
            d["resid"] = d["y_true"] - d[col]
            q = {}
            for h, g in d.groupby("h"):
                if len(g) >= 20:
                    q[int(h)] = {"p2_5": float(g["resid"].quantile(0.025)),
                                 "p50": float(g["resid"].quantile(0.5)),
                                 "p97_5": float(g["resid"].quantile(0.975))}
            self.resid[b] = q


STORE = Store().load()


# ══════════════════════════════════════════════════════════════════════════════
# 보조
# ══════════════════════════════════════════════════════════════════════════════
def _require_ready():
    if not STORE.ready:
        raise HTTPException(503, STORE.reason)


def _origin_index(asof: str | None) -> int:
    d = STORE.df
    if asof is None:
        return int(len(d) - 1)
    ts = pd.Timestamp(asof)
    idx = d.index[d["date"] <= ts]
    if len(idx) == 0:
        raise HTTPException(400, f"asof={asof} 는 자료 시작 이전이다")
    return int(idx[-1])


def _band_meta(band: str) -> dict:
    if band not in STORE.models:
        raise HTTPException(404, f"구간 {band} 모델이 없다. 사용 가능: {list(STORE.models)}")
    cv = STORE.summary.get("bands", {}).get(band, {})
    serving = STORE.manifest.get("bands", {}).get(band, {}).get("cv", {})
    return {"cv": cv, "serving": serving}


def _trustworthy(band: str) -> bool:
    s = STORE.manifest.get("bands", {}).get(band, {}).get("cv", {})
    return bool(s.get("beats_naive", True))


def _predict_band(band: str, t0: int) -> list[dict]:
    m = STORE.models[band]
    lo, hi = m["lo"], m["hi"]
    hs = np.arange(lo, hi + 1)
    row = STORE.X.iloc[[t0]][m["features"]]
    D = pd.concat([row] * len(hs), ignore_index=True)
    D["__h__"] = hs.astype(np.float32)
    p = np.asarray(m["model"].predict(D), float).ravel()

    q = STORE.resid.get(band, {})
    base_date = STORE.df["date"].iloc[t0]
    out = []
    for h, v in zip(hs, p):
        r = q.get(int(h)) or (list(q.values())[-1] if q else None)
        out.append({
            "h": int(h),
            "date": str((base_date + pd.Timedelta(days=int(h))).date()),
            "ch4_m3d": round(float(v), 1),
            "lo95": round(float(v + r["p2_5"]), 1) if r else None,
            "hi95": round(float(v + r["p97_5"]), 1) if r else None,
        })
    return out


def _state(t0: int) -> pd.Series:
    return STORE.df.iloc[t0]


def _baseline(t0: int, days: int = C.RELATIVE_ALARM["baseline_days"]) -> pd.Series:
    lo = max(0, t0 - days + 1)
    return STORE.df.iloc[lo : t0 + 1].median(numeric_only=True)


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {
        "ready": STORE.ready, "reason": STORE.reason,
        "bands": list(STORE.models),
        "data_range": [str(STORE.df["date"].min().date()), str(STORE.df["date"].max().date())]
        if STORE.df is not None else None,
    }


@app.get("/api/bands")
def bands():
    """구간별 검증 성적 — 화면이 '이 예측을 믿어도 되는가'를 표시할 수 있게 한다."""
    _require_ready()
    out = {}
    for b in STORE.models:
        cv = STORE.summary.get("bands", {}).get(b, {})
        sv = STORE.manifest.get("bands", {}).get(b, {}).get("cv", {})
        out[b] = {
            "range_days": [STORE.models[b]["lo"], STORE.models[b]["hi"]],
            "model": STORE.models[b]["model_name"],
            "pooled_R2": sv.get("pooled_R2"),
            "RMSSE_pct": sv.get("RMSSE_pct"),
            "meets_r2_goal": sv.get("meets_r2_goal"),
            "beats_naive": sv.get("beats_naive"),
            "naive_R2": cv.get("naive", {}).get("pooled_R2"),
            "n_test": cv.get("n_test"),
            "n_folds": cv.get("n_folds"),
        }
    return out


@app.get("/api/overview")
def overview(asof: str | None = Query(None)):
    """상단 신호등 + 4개 게이지."""
    _require_ready()
    t0 = _origin_index(asof)
    st, bl = _state(t0), _baseline(t0)
    diag = diagnose(st, bl)
    worst = max(diag, key=lambda d: {"정상": 0, "주의": 1, "위험": 2}[d["level"]]) if diag else None
    lamp = worst["level"] if worst else "정상"

    band = "h1_3" if "h1_3" in STORE.models else list(STORE.models)[0]
    fc = _predict_band(band, t0)
    tomorrow = fc[0] if fc else None

    def tag(level):
        return {"정상": "NORMAL", "주의": "CAUTION", "위험": "ACTION"}[level]

    def lv(col):
        for d in diag:
            if d["variable"] == col:
                return d["level"]
        return "정상"

    return {
        "asof": str(st["date"].date()),
        "lamp": lamp,
        "alarm": {
            "code": f"ALARM {worst['variable']}" if worst and lamp != "정상" else "NO ACTIVE ALARM",
            "title": (f"{worst['variable']} 기준선 대비 {worst['deviation_pct']:+.1f}% "
                      f"[{tag(lamp)}]") if worst else "전 계통 정상",
            "detail": " | ".join(
                f"{d['variable']}: {d['value']} ({d['deviation_pct']:+.1f}% vs 90d)"
                for d in diag[:3]),
        },
        "gauges": [
            {"label": f"메탄생산 예측 (T+{tomorrow['h']})" if tomorrow else "메탄생산 예측",
             "value": tomorrow["ch4_m3d"] if tomorrow else None, "unit": "m³",
             "status": "NORMAL" if _trustworthy(band) else "NO-MODEL",
             "note": f"{band} · R²={STORE.manifest['bands'][band]['cv'].get('pooled_R2')}"},
            {"label": "소화조 온도", "value": None if not np.isfinite(st.get("dig_T_A_C", np.nan))
             else round(float(st["dig_T_A_C"]), 1), "unit": "℃",
             "status": tag(lv("dig_T_A_C"))},
            {"label": "VFA / Alk 비율", "value": None if not np.isfinite(st.get("VFA_ALK_A", np.nan))
             else round(float(st["VFA_ALK_A"]), 3), "unit": "RATIO",
             "status": tag(lv("VFA_ALK_A"))},
            {"label": "메탄 가스 순도",
             "value": round(float(st.get("CH4_pct_filled", np.nan)), 1)
             if np.isfinite(st.get("CH4_pct_filled", np.nan)) else None,
             "unit": "%", "status": tag(lv(C.CONC))},
        ],
        "diagnostics": diag,
    }


@app.get("/api/trend")
def trend(asof: str | None = None, back: int = 14, band: str = "h1_3"):
    """실측 이력 + 구간 예측(95% 경험 예측구간)."""
    _require_ready()
    t0 = _origin_index(asof)
    lo = max(0, t0 - back + 1)
    hist = STORE.df.iloc[lo : t0 + 1]
    return {
        "asof": str(STORE.df["date"].iloc[t0].date()),
        "band": band,
        "history": [
            {"date": str(d.date()),
             "ch4_m3d": None if not np.isfinite(v) else round(float(v), 1),
             "observed": bool(np.isfinite(v)),
             "ch4_filled": round(float(f), 1) if np.isfinite(f) else None}
            for d, v, f in zip(hist["date"], hist[C.TARGET], hist["CH4_m3d_filled"])],
        "forecast": _predict_band(band, t0),
        "trust": {"beats_naive": _trustworthy(band),
                  "pooled_R2": STORE.manifest.get("bands", {}).get(band, {})
                  .get("cv", {}).get("pooled_R2")},
    }


@app.get("/api/telemetry")
def telemetry(asof: str | None = None):
    """계측 매트릭스 — 값 + 기준선 대비 편차 + 신호등."""
    _require_ready()
    t0 = _origin_index(asof)
    st, bl = _state(t0), _baseline(t0)
    labels = {
        "dig_T_A_C": ("소화조 온도", "℃"), "VFA_ALK_A": ("VFA / Alk 비율", ""),
        C.CONC: ("메탄 가스 순도", "%"), "dig_pH_A": ("소화조 내부 pH", ""),
        C.FLOW: ("총 바이오가스 발생량", "m³/d"), "feed_AB_tpd": ("소화조 투입량", "t/d"),
        "OLR_calc": ("유기물 부하율 OLR", "kgVS/㎥·d"), "FAN_calc": ("유리암모니아 FAN", "mg/L"),
        "VS_destruction_pct": ("VS 분해율", "%"), "Y_COD": ("COD 기준 메탄수율", "㎥/kgCOD"),
    }
    diag = {d["variable"]: d for d in diagnose(st, bl)}
    rows = []
    for col, (label, unit) in labels.items():
        v = st.get(col, np.nan)
        if col == C.CONC and not np.isfinite(v):
            v = st.get("CH4_pct_filled", np.nan)
        d = diag.get(col, {})
        rows.append({
            "variable": col, "label": label, "unit": unit,
            "value": None if not np.isfinite(v) else round(float(v), 3),
            "baseline_90d": d.get("baseline_90d"),
            "deviation_pct": d.get("deviation_pct"),
            "level": d.get("level", "정상"),
            "guideline_note": d.get("guideline_note", ""),
        })
    return {"asof": str(st["date"].date()), "rows": rows}


@app.get("/api/forecast")
def forecast(asof: str | None = None, band: str = "h1_3"):
    _require_ready()
    _band_meta(band)
    t0 = _origin_index(asof)
    return {"asof": str(STORE.df["date"].iloc[t0].date()), "band": band,
            "points": _predict_band(band, t0), "trust": _trustworthy(band)}


@app.get("/api/advisory")
def advisory(asof: str | None = None, band: str = "h7_14"):
    """
    운전 제어 권고 (요구사항 5).
    모델에 반사실 질의를 던져 「무엇을 얼마나 바꾸면 메탄이 얼마나 변하는가」를 낸다.
    """
    _require_ready()
    if band not in STORE.models:
        raise HTTPException(404, f"구간 {band} 모델 없음")
    t0 = _origin_index(asof)
    m = STORE.models[band]
    row = STORE.X.iloc[[t0]][m["features"]].copy()
    row["__h__"] = np.float32((m["lo"] + m["hi"]) / 2)
    st = _state(t0)
    bl = _baseline(t0)
    base_ch4 = float(bl.get("CH4_m3d_filled", STORE.df["CH4_m3d_filled"].median()))
    rec = recommend(m["model"], row, st, list(row.columns), band, base_ch4,
                    model_is_trustworthy=_trustworthy(band))
    rec["asof"] = str(st["date"].date())
    rec["diagnostics"] = diagnose(st, bl)
    return rec


class SimulateReq(BaseModel):
    asof: str | None = None
    band: str = "h7_14"
    changes: dict[str, float] = {}          # 원천변수명 → 배수(온도는 가산)


@app.post("/api/simulate")
def simulate(req: SimulateReq):
    """what-if — 조작단을 바꿨을 때의 예측을 즉시 되돌려 준다."""
    _require_ready()
    if req.band not in STORE.models:
        raise HTTPException(404, f"구간 {req.band} 모델 없음")
    t0 = _origin_index(req.asof)
    m = STORE.models[req.band]
    hs = np.arange(m["lo"], m["hi"] + 1)
    base = STORE.X.iloc[[t0]][m["features"]]

    def frame(mods: dict):
        r = base.copy()
        for src, mult in mods.items():
            cols = [c for c in m["features"]
                    if c.split("__", 1)[-1].split("_lag")[0].split("_ma")[0]
                    .split("_tr")[0] == src]
            for c in cols:
                r[c] = r[c] + mult if src == "dig_T_A_C" else r[c] * mult
        D = pd.concat([r] * len(hs), ignore_index=True)
        D["__h__"] = hs.astype(np.float32)
        return D

    p0 = np.asarray(m["model"].predict(frame({})), float)
    p1 = np.asarray(m["model"].predict(frame(req.changes)), float)
    return {
        "asof": str(STORE.df["date"].iloc[t0].date()), "band": req.band,
        "changes": req.changes,
        "baseline_mean_ch4": round(float(p0.mean()), 1),
        "scenario_mean_ch4": round(float(p1.mean()), 1),
        "delta_ch4": round(float(p1.mean() - p0.mean()), 1),
        "delta_pct": round(100 * float(p1.mean() - p0.mean()) / float(p0.mean()), 2),
        "points": [{"h": int(h), "baseline": round(float(a), 1), "scenario": round(float(b), 1)}
                   for h, a, b in zip(hs, p0, p1)],
    }


@app.get("/api/alarms")
def alarms(asof: str | None = None, days: int = 180):
    """경보 이력 — 시설 자체 90일 기준선 대비 상대편차로 산출한다."""
    _require_ready()
    t0 = _origin_index(asof)
    lo = max(0, t0 - days + 1)
    out = []
    cau, dan = C.RELATIVE_ALARM["caution_pct"], C.RELATIVE_ALARM["danger_pct"]
    for i in range(lo, t0 + 1):
        st, bl = STORE.df.iloc[i], _baseline(i)
        for d in diagnose(st, bl):
            if d["level"] != "정상":
                out.append({"date": str(st["date"].date()), **d})
    out.sort(key=lambda r: (r["date"], -abs(r["deviation_pct"])), reverse=True)
    return {"n": len(out), "thresholds": {"caution_pct": cau, "danger_pct": dan},
            "items": out[:300]}


# ── 정적 프론트엔드 ──────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
