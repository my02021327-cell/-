"""
서빙용 최종 모델 학습·저장

`train_bands.py` 의 rolling-origin 결과(`outputs/bands/serving.json`)가 구간별로
어떤 모델을 채택할지 정한다. 여기서는 그 선택을 받아 **전체 자료로 재적합**하고
피처 목록과 함께 저장한다. 서버는 이 산출물만 읽는다.

■ 기준선과 스태킹도 그대로 서빙한다
  짧은 지평에서는 FlowAnchor(당일 실측 유량 × 최근 관측 농도)가 학습 모델 전부를
  이겼고, 중장기에서는 수축 스태킹이 이겼다. 둘 다 sklearn 인터페이스로 감싸
  (`AnchorModel`, `ShrunkStack`) 다른 모델과 같은 경로로 서빙한다.
  「채택된 것은 A 인데 저장된 것은 B」가 되는 상황을 만들지 않는다.

■ 스태킹 가중은 다시 고르지 않는다
  교차검증에서 폴드 학습셋 안의 서로 다른 절반으로 정한 (w, s) 를 그대로 가져온다.
  서빙 단계에서 재선택하면 시험 자료를 보게 된다.

실행 : python -m src.bgp.serve_models
산출 : outputs/models/band_<name>.joblib · outputs/models/manifest.json
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.bgp import config as C
from src.bgp.features import horizon_pairs
from src.bgp.horizon import select_features
from src.bgp.models import (
    AnchorModel,
    ColumnModel,
    FlowConcModel,
    ShrunkStack,
    model_zoo,
)
from src.bgp.pipeline import prepare

BANDS_DIR = os.path.join(C.OUT_DIR, "bands")
ANCHOR_COL = "tmp__y_last"          # = CH4_m3d_causal (원점의 인과 재구성 메탄)


def _load(name: str) -> dict:
    p = os.path.join(BANDS_DIR, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


# 스태킹 멤버 이름 → 서빙 가능한 추정기
BASELINE_COLS = {"Naive": "tmp__y_last_obs", "SeasonalNaive": "tmp__y_ma30",
                 "FlowAnchor": ANCHOR_COL}


def _fit_member(name: str, zoo: dict, D: pd.DataFrame, tgt: np.ndarray,
                df: pd.DataFrame, o: np.ndarray, h: np.ndarray):
    """멤버 하나를 전체 자료로 적합한다. 서빙할 수 없는 멤버는 None."""
    if name in BASELINE_COLS:
        col = BASELINE_COLS[name]
        return ColumnModel(col).fit(D, tgt) if col in D.columns else None
    if name == "FlowConc":
        tstar = o + h
        flow = df[C.FLOW].ffill().to_numpy(float)[tstar]
        conc = df["CH4_pct_causal"].ffill().to_numpy(float)[tstar]
        ok = np.isfinite(flow) & np.isfinite(conc)
        if ok.sum() < 200:
            return None
        return FlowConcModel(C.DQ["ranges"][C.CONC]).fit(D[ok], flow[ok], conc[ok])
    if name in zoo:
        m = zoo[name][0]
        m.fit(D, tgt)
        return m
    return None


def build(n_features: int = 120, max_train: int = 30000) -> dict:
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    df, X, groups, info = prepare()
    serving = _load("serving.json")
    summary = _load("summary.json")
    zoo = model_zoo(fast=False)
    manifest = {"info": info, "bands": {}}

    y_lab = df["CH4_m3d_filled"]
    observed = df[C.TARGET].notna()

    for name, lo, hi in C.HORIZON_BANDS:
        o, h, tgt, obs = horizon_pairs(X, y_lab, observed, lo, hi, min_origin=60)
        if len(o) < 500:
            continue
        if len(o) > max_train:
            keep = np.arange(len(o))[-max_train:]
            o, h, tgt = o[keep], h[keep], tgt[keep]

        Xd = X.iloc[o].reset_index(drop=True)
        feat = select_features(Xd, tgt, k=n_features)
        # 앵커 열과 기준선 열은 스태킹 서빙에 반드시 필요하다
        must = [c for c in (ANCHOR_COL, "tmp__y_last_obs", "tmp__y_ma30")
                if c in X.columns]
        feat = must + [c for c in feat if c not in must]
        feat = feat[:n_features]
        D = Xd[feat].copy()
        D["__h__"] = h.astype(np.float32)

        pick = serving.get(name, {}).get("model", "HistGBM")
        weights = (summary.get("bands", {}).get(name, {})
                   .get("stack_weights_mean", {}))

        if pick == "FlowAnchor":
            est = AnchorModel(ANCHOR_COL).fit(D, tgt)
            members_used = {}
        elif pick == "Stack":
            # 교차검증에서 가중을 받은 멤버는 **전부** 서빙해야 한다. 일부만 살리면
            # 검증된 모델과 서빙되는 모델이 달라진다.
            members_used, fitted = {}, {}
            for k, v in weights.items():
                if k.startswith("_") or v <= 1e-4:
                    continue
                m = _fit_member(k, zoo, D, tgt, df, o, h)
                if m is None:
                    continue
                fitted[k] = m
                members_used[k] = float(v)
            est = ShrunkStack(fitted, members_used, ANCHOR_COL).fit(D, tgt)
        else:
            cands = zoo.get(pick) or zoo["HistGBM"]
            est = cands[0]
            est.fit(D, tgt)
            members_used = {}

        path = os.path.join(C.MODEL_DIR, f"band_{name}.joblib")
        joblib.dump({"model": est, "features": feat, "lo": lo, "hi": hi,
                     "model_name": pick, "stack_members": members_used,
                     "anchor_col": ANCHOR_COL}, path)
        manifest["bands"][name] = {
            "model": pick, "n_features": len(feat), "n_train": int(len(tgt)),
            "path": os.path.relpath(path, C.ROOT),
            "stack_members": members_used,
            # 앵커 전용 모델은 조작단에 반응하지 않는다 — 화면이 그 사실을 알아야 한다
            "responds_to_actuators": pick != "FlowAnchor",
            "cv": serving.get(name, {}),
        }
        extra = (f" members={list(members_used)}" if members_used else "")
        print(f"[{name:7s}] {pick:14s} n={len(tgt):6d} feat={len(feat)}{extra}", flush=True)

    snap_path = os.path.join(C.MODEL_DIR, "panel.joblib")
    joblib.dump({"df": df, "X": X, "groups": groups}, snap_path, compress=3)
    manifest["panel"] = os.path.relpath(snap_path, C.ROOT)

    with open(os.path.join(C.MODEL_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("저장 →", C.MODEL_DIR)
    return manifest


if __name__ == "__main__":
    build()
