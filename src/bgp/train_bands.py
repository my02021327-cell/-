"""
지평 구간별 메탄 예측 모델 학습·평가 (요구사항 1~6 의 모델링 본체)

실행
  python -m src.bgp.train_bands                     # 전 구간 · 전 각도
  python -m src.bgp.train_bands --fast              # 후보 축소(빠른 확인)
  python -m src.bgp.train_bands --with-gru          # 시계열 심층학습(GRU) 포함
  python -m src.bgp.train_bands --ablation          # 각도별 단독 성능까지

산출
  outputs/bands/summary.json        구간·모델별 pooled R²·RMSSE·대응검정
  outputs/bands/oof_<band>.csv      폴드밖 예측 (대시보드·잔차진단용)
  outputs/bands/ablation.json       각도별 단독 성능
  outputs/bands/serving.json        서빙용 채택 모델 지정
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.bgp import config as C
from src.bgp.horizon import run_band
from src.bgp.pipeline import prepare

OUT = os.path.join(C.OUT_DIR, "bands")


def _strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "_oof"}


def save_oof(res: dict, path: str):
    o = res["_oof"]
    d = {"origin_idx": o["t"], "h": o["h"], "fold": o["fold"], "y_true": o["y"]}
    for k, v in o["pred"].items():
        d[f"pred_{k}"] = v
    pd.DataFrame(d).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--with-gru", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--bands", default="", help="쉼표구분 band 이름. 비우면 전부")
    ap.add_argument("--n-features", type=int, default=120)
    ap.add_argument("--max-train", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    df, X, groups, info = prepare()

    bands = C.HORIZON_BANDS
    if args.bands:
        want = set(args.bands.split(","))
        bands = [b for b in bands if b[0] in want]

    summary = {"info": info, "bands": {}, "target_r2_goal": 0.85}
    for band in bands:
        t = time.time()
        res = run_band(df, X, df["CH4_m3d_filled"], df[C.TARGET], band,
                       fast=args.fast, with_gru=args.with_gru,
                       n_features=args.n_features, max_train=args.max_train,
                       verbose=False)
        if "error" in res:
            print(f"[{band[0]}] {res['error']}", flush=True)
            continue
        save_oof(res, os.path.join(OUT, f"oof_{band[0]}.csv"))
        summary["bands"][band[0]] = _strip(res)
        best = next(iter(res["models"].items()))
        print(f"[{band[0]:7s}] {time.time()-t:6.0f}s  folds={res['n_folds']:2d} "
              f"n={res['n_test']:5d}  naive R²={res['naive']['pooled_R2']:7.3f} | "
              f"best={best[0]:14s} R²={best[1]['pooled_R2']:7.4f} "
              f"RMSSE={best[1]['RMSSE_pct']:6.1f}%", flush=True)

    # ── 각도별 단독 성능 (요구사항 2) ────────────────────────────────────────
    if args.ablation:
        abl = {}
        for angle in ("substrate", "chemistry", "vsbalance", "temporal"):
            _, Xa, _, _ = prepare(angles=(angle,), verbose=False)
            abl[angle] = {}
            for band in bands:
                r = run_band(df, Xa, df["CH4_m3d_filled"], df[C.TARGET], band,
                             fast=True, n_features=min(args.n_features, Xa.shape[1]),
                             verbose=False)
                if "error" in r:
                    continue
                b = next(iter(r["models"].items()))
                abl[angle][band[0]] = {"best_model": b[0], "pooled_R2": b[1]["pooled_R2"],
                                       "RMSSE_pct": b[1]["RMSSE_pct"]}
                print(f"  [ablation {angle:10s} {band[0]:7s}] R²={b[1]['pooled_R2']:7.4f} "
                      f"({b[0]})", flush=True)
        with open(os.path.join(OUT, "ablation.json"), "w", encoding="utf-8") as f:
            json.dump(abl, f, ensure_ascii=False, indent=2)
        summary["ablation"] = abl

    # ── 서빙용 채택 모델 : 구간마다 pooled R² 최고 & RMSSE<100 인 것 ──────────
    serving = {}
    for bname, res in summary["bands"].items():
        ok = {k: v for k, v in res["models"].items() if v["RMSSE_pct"] < 100}
        pick = max(ok or res["models"], key=lambda k: (ok or res["models"])[k]["pooled_R2"])
        serving[bname] = {
            "model": pick, "pooled_R2": res["models"][pick]["pooled_R2"],
            "RMSSE_pct": res["models"][pick]["RMSSE_pct"],
            "meets_r2_goal": bool(res["models"][pick]["pooled_R2"] >= 0.85),
            "beats_naive": bool(res["models"][pick]["RMSSE_pct"] < 100),
        }
    summary["serving"] = serving
    with open(os.path.join(OUT, "serving.json"), "w", encoding="utf-8") as f:
        json.dump(serving, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'구간':8s}{'채택 모델':>16s}{'pooled R²':>12s}{'RMSSE%':>9s}{'R²≥0.85':>9s}")
    for b, s in serving.items():
        print(f"{b:8s}{s['model']:>16s}{s['pooled_R2']:>12.4f}{s['RMSSE_pct']:>9.1f}"
              f"{'✓' if s['meets_r2_goal'] else '✗':>9s}")
    print(f"\n총 {time.time()-t0:.0f}s · 저장 → {OUT}")


if __name__ == "__main__":
    main()
