"""
파이프라인·API 연기 시험 — 배포 전에 이것부터 통과시킨다

`python -m src.bgp.train_bands` 와 `python -m src.bgp.serve_models` 를 돌린 뒤 실행한다.
서버를 실제로 띄우지 않고 FastAPI TestClient 로 전 엔드포인트를 두드린다.

실행 : python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    print("[1/4] 자료 · 품질")
    from src.bgp.data import build_frame, quality_report
    from src.bgp import config as C

    df = build_frame()
    q = quality_report(df)
    check("자료 2,086일 적재", q["n_days"] == 2086, str(q["n_days"]))
    check("유량 결측 0 %", q["flow_missing_pct"] == 0.0, f"{q['flow_missing_pct']}%")
    check("타깃 결측 = 농도 결측", q["target_missing_pct"] == q["conc_missing_pct"],
          f"{q['target_missing_pct']}% / {q['conc_missing_pct']}%")
    check("2018년 열역학 위반 편중 재현",
          q["thermo_violation_by_year"].get(2018, 0) > 60,
          f"{q['thermo_violation_by_year'].get(2018)}%")
    check("2023 ALK_B 폐기", df.loc[df['date'] >= '2023-01-01', 'ALK_B_mgL'].notna().sum() == 0)

    print("[2/4] 보간")
    from src.bgp.impute import HybridImputer
    from src.bgp.data import ANGLES

    pred = [c for k, v in ANGLES.items() if k != "temporal"
            for c in v if c in df.columns] + [C.FLOW, "dow", "month"]
    imp = HybridImputer().fit(df[df["date"] < "2022-01-01"], pred, n_blocks=40)
    check("보간 후보를 가림 실험으로 선택", imp.method_ in imp.CANDIDATES, imp.method_)
    check("소프트센서가 최선이 아님을 확인(논문 그대로 쓰면 안 되는 근거)",
          imp.scores_.get("softsensor", 9) > imp.scores_[imp.method_],
          json.dumps(imp.scores_, ensure_ascii=False))
    causal = HybridImputer(causal_only=True).fit(df[df["date"] < "2022-01-01"], pred, n_blocks=40)
    check("실시간 경로는 인과 후보만", causal.method_ != "linear", causal.method_)

    print("[3/4] 피처 인과성")
    from src.bgp.pipeline import prepare
    import numpy as np

    d2, X, groups, info = prepare(verbose=False)
    check("네 각도 모두 생성", set(groups) == {"substrate", "chemistry", "vsbalance", "temporal"},
          str({k: len(v) for k, v in groups.items()}))
    # 인과성 : 미래를 참조하면 마지막 행 이후 값을 바꿔도 과거 피처가 흔들린다
    from src.bgp.features import build_features

    # prepare() 와 **같은 타깃 열**로 다시 만들어야 비교가 성립한다.
    # (다른 열로 만들면 시계열 각도 전체가 달라져 누출과 무관하게 불일치가 난다)
    X0, _ = build_features(d2, target_col="CH4_m3d_causal")
    d3 = d2.copy()
    d3.loc[d3.index[-30:], C.FLOW] = d3[C.FLOW].iloc[-30:] * 3.0
    X3, _ = build_features(d3, target_col="CH4_m3d_causal")
    a = np.nan_to_num(X0.iloc[:-30].to_numpy(float))
    b = np.nan_to_num(X3.iloc[:-30].to_numpy(float))
    same = np.allclose(a, b)
    if not same:
        bad = np.where(~np.isclose(a, b).all(axis=0))[0]
        detail = ", ".join(X0.columns[i] for i in bad[:5])
    else:
        detail = f"{X0.shape[1]}개 열 · 마지막 30일 유량 3배 교란"
    check("미래 값을 바꿔도 과거 피처가 불변(누출 없음)", same, detail)

    print("[4/4] API")
    try:
        from fastapi.testclient import TestClient
        from server.app import app

        c = TestClient(app)
        h = c.get("/api/health").json()
        if not h.get("ready"):
            print(f"  SKIP  모델 산출물 없음 — {h.get('reason')}")
        else:
            for path in ("/api/bands", "/api/overview", "/api/telemetry",
                         "/api/trend?back=14", "/api/alarms?days=90"):
                r = c.get(path)
                check(f"GET {path}", r.status_code == 200, f"HTTP {r.status_code}")
            b = list(c.get("/api/bands").json())[0]
            r = c.get(f"/api/advisory?band={b}")
            check(f"GET /api/advisory?band={b}", r.status_code == 200)
            a = r.json()
            check("권고에 예상 ΔCH₄ 가 붙어 있다",
                  (not a["actions"]) or all("expected_delta_ch4_m3d" in x for x in a["actions"]))
            check("신뢰 못하는 구간은 권고를 내지 않는다",
                  a["model_trustworthy"] or not a["actions"])
            r = c.post("/api/simulate", json={"band": b, "changes": {"feed_AB_tpd": 1.1}})
            check("POST /api/simulate", r.status_code == 200,
                  f"Δ={r.json().get('delta_ch4')}" if r.status_code == 200 else "")
            check("정적 프론트엔드 서빙", c.get("/").status_code == 200)
    except Exception as e:
        check("API 기동", False, f"{type(e).__name__}: {e}")

    print()
    if FAIL:
        print(f"실패 {len(FAIL)}건: " + ", ".join(FAIL))
        sys.exit(1)
    print("전 항목 통과")


if __name__ == "__main__":
    main()
