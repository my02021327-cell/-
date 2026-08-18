"""
서빙용 최종 모델 학습·저장

`train_bands.py` 의 rolling-origin 결과(`outputs/bands/serving.json`)가 구간별로
어떤 모델을 채택할지 정한다. 여기서는 그 선택을 받아 **전체 자료로 재적합**하고
피처 목록·정규화 정보와 함께 저장한다. 서버는 이 산출물만 읽는다.

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
from src.bgp.models import model_zoo
from src.bgp.pipeline import prepare

BANDS_DIR = os.path.join(C.OUT_DIR, "bands")


def _load_serving() -> dict:
    p = os.path.join(BANDS_DIR, "serving.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {}


def build(n_features: int = 120, max_train: int = 30000) -> dict:
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    df, X, groups, info = prepare()
    serving = _load_serving()
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
        D = Xd[feat].copy()
        D["__h__"] = h.astype(np.float32)

        pick = serving.get(name, {}).get("model", "HistGBM")
        cands = zoo.get(pick) or zoo["HistGBM"]
        est = cands[0]
        est.fit(D, tgt)

        path = os.path.join(C.MODEL_DIR, f"band_{name}.joblib")
        joblib.dump({"model": est, "features": feat, "lo": lo, "hi": hi,
                     "model_name": pick}, path)
        manifest["bands"][name] = {
            "model": pick, "n_features": len(feat), "n_train": int(len(tgt)),
            "path": os.path.relpath(path, C.ROOT),
            "cv": serving.get(name, {}),
        }
        print(f"[{name:7s}] {pick:15s} n={len(tgt):6d} feat={len(feat)} → {os.path.basename(path)}",
              flush=True)

    # 서버가 쓸 최신 상태 스냅샷 (마지막 원점의 피처 행 + 원자료 상태)
    snap_path = os.path.join(C.MODEL_DIR, "panel.joblib")
    joblib.dump({"df": df, "X": X, "groups": groups}, snap_path, compress=3)
    manifest["panel"] = os.path.relpath(snap_path, C.ROOT)

    with open(os.path.join(C.MODEL_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("저장 →", C.MODEL_DIR)
    return manifest


if __name__ == "__main__":
    build()
