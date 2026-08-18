"""
운전 제어 권고 — 예측된 메탄 변화에 대해 '무엇을 얼마나 바꿀지'를 근거와 함께 제시

요구사항 5. 이 모듈은 규칙표를 읽어 문구를 고르지 않는다. **채택된 예측 모델 자체에
반사실(counterfactual) 질의**를 던져 「그 변수를 이만큼 바꾸면 메탄이 이만큼 변한다」를
직접 계산하고, 거기에 물리·지침 제약을 씌워 실행 가능한 값으로 자른다.

■ 제어 가능 변수와 제약
  운전원이 실제로 손댈 수 있는 것만 후보에 넣는다. VFA·알칼리도·pH·NH₃-N 은 결과이지
  조작단이 아니므로 **권고 대상이 아니라 안전 게이트**로만 쓴다.

  | 조작단 | 열 | 범위 근거 |
  |---|---|---|
  | 소화조 투입량 | feed_AB_tpd | 최근 90일 5~95 백분위 안 |
  | 기질 배합 (음폐수/가축분뇨/음식물) | mix_* | 반입 실적 분포 안, 합=1 |
  | 소화조 온도 | dig_T_A_C | 설정치 ±1 ℃ |
  | 탈수 처리량(→ 유출) | dewater_tpd | 최근 90일 분포 안 |

■ 안전 게이트 (위반 시 '부하 증대' 계열 권고를 금지한다)
  · VFA/ALK ≥ 0.4     산 축적. 부하를 더 넣으면 산패로 간다.
  · FAN ≥ 300 mg/L    암모니아 저해 구간. 질소 부하를 늘리면 안 된다.
  · pH < 6.8          완충 붕괴 진행.
  · OLR ≥ 4.0         지침 상한.

■ 정직성 규약
  · 권고마다 **예상 ΔCH₄ 와 그 근거(반사실 곡선의 기울기)** 를 함께 낸다.
  · 모델이 그 구간에서 naive 를 못 이기면(RMSSE ≥ 100 %) 권고를 내지 않고
    "이 지평에서는 모델 근거가 없음"을 그대로 표시한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bgp import config as C

# 조작단 정의 : (열, 표시명, 단위, 스텝 탐색 배수, 부하증대 방향인가)
ACTUATORS = [
    ("feed_AB_tpd", "소화조 투입량", "t/d", (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15), True),
    ("mix_foodww", "음폐수 배합비", "비율", (0.8, 0.9, 1.0, 1.1, 1.2), True),
    ("mix_manure", "가축분뇨 배합비", "비율", (0.8, 0.9, 1.0, 1.1, 1.2), True),
    ("mix_food", "음식물류 배합비", "비율", (0.8, 0.9, 1.0, 1.1, 1.2), True),
    ("dig_T_A_C", "소화조 온도", "℃", (-1.0, -0.5, 0.0, 0.5, 1.0), False),
    ("dewater_tpd", "탈수 처리량", "t/d", (0.8, 0.9, 1.0, 1.1, 1.2), False),
]

SAFETY_GATES = [
    ("VFA_ALK_A", 0.40, "ge", "VFA/알칼리도 {v:.2f} ≥ 0.40 — 산 축적 구간. 유기물 부하 증대 금지."),
    ("FAN_calc", 300.0, "ge", "유리암모니아 {v:.0f} mg/L ≥ 300 — 저해 구간. 질소 부하 증대 금지."),
    ("dig_pH_A", 6.8, "le", "소화조 pH {v:.2f} < 6.8 — 완충 붕괴 진행. 부하 증대 금지."),
    ("OLR_calc", 4.0, "ge", "OLR {v:.2f} ≥ 4.0 kgVS/㎥·d — 지침 상한. 부하 증대 금지."),
]


# 게이트를 발화시킬 수 있는 측정의 최대 나이(일). 이보다 오래된 값은 여전히 보수적으로
# 게이트를 걸되 '오래된 값'임을 명시한다 — 이 시설은 NH₃-N 이 89 % 결측이고 FAN 이
# 1년 넘게 갱신되지 않는 구간이 있다. 그 값으로 조용히 판정하면 근거를 속이는 것이다.
GATE_MAX_AGE_DAYS = 30


def check_safety(state: pd.Series, ages: dict | None = None) -> list[dict]:
    """
    안전 게이트 판정. 걸린 게이트는 '부하 증대' 방향 권고를 봉쇄한다.

    `ages` 는 변수별 측정 나이(일)다. 오래된 값으로도 게이트는 걸지만(보수적 선택),
    응답에 나이와 stale 표시를 실어 운전원이 근거의 신선도를 볼 수 있게 한다.
    """
    ages = ages or {}
    hits = []
    for col, thr, op, msg in SAFETY_GATES:
        v = state.get(col, np.nan)
        if not np.isfinite(v):
            continue
        if (op == "ge" and v >= thr) or (op == "le" and v <= thr):
            age = ages.get(col)
            stale = age is not None and age > GATE_MAX_AGE_DAYS
            m = msg.format(v=float(v))
            if age is not None:
                m += f" (측정 {age}일 전{' — 오래된 값' if stale else ''})"
            hits.append({"variable": col, "value": round(float(v), 3),
                         "threshold": thr, "age_days": age, "stale": bool(stale),
                         "message": m})
    return hits


def counterfactual_curve(model, x_row: pd.DataFrame, feature_cols: list[str],
                         base_value: float, multipliers, additive: bool = False):
    """
    한 조작단을 흔들어 예측 메탄이 어떻게 움직이는지 곡선으로 만든다.

    설계행렬에는 같은 원천 변수의 파생열(이동평균·시차·추세)이 여러 개 들어 있으므로,
    **원천 이름이 들어간 열을 모두 함께** 흔들어야 한다. 하나만 바꾸면 모델이 모순된
    입력을 받아 기울기가 왜곡된다.
    """
    rows, values = [], []
    for m in multipliers:
        v = base_value + m if additive else base_value * m
        r = x_row.copy()
        for c in feature_cols:
            if additive:
                r[c] = r[c] + m
            else:
                r[c] = r[c] * m
        rows.append(r)
        values.append(v)
    Xc = pd.concat(rows, ignore_index=True)
    pred = np.asarray(model.predict(Xc), float).ravel()
    return np.asarray(values, float), pred


def recommend(model, x_row: pd.DataFrame, state: pd.Series, design_cols: list[str],
              band: str, baseline_ch4: float, model_is_trustworthy: bool = True,
              max_actions: int = 3, ages: dict | None = None) -> dict:
    """
    반환 : 예측·안전판정·권고 목록.

    권고는 「예상 ΔCH₄ 가 큰 순서」로 정렬하되, 안전 게이트에 걸린 방향은 제외한다.
    """
    pred_now = float(np.asarray(model.predict(x_row), float).ravel()[0])
    gates = check_safety(state, ages)
    load_blocked = len(gates) > 0

    out = {
        "band": band,
        "predicted_ch4_m3d": round(pred_now, 1),
        "baseline_ch4_m3d": round(float(baseline_ch4), 1),
        "delta_vs_baseline_pct": round(100 * (pred_now - baseline_ch4) / baseline_ch4, 1)
        if baseline_ch4 else None,
        "safety_gates": gates,
        "load_increase_blocked": load_blocked,
        "actions": [],
        "model_trustworthy": bool(model_is_trustworthy),
    }
    if not model_is_trustworthy:
        out["note"] = ("이 지평에서 채택 모델이 naive 기준선을 이기지 못한다(RMSSE ≥ 100 %). "
                       "제어 권고의 근거가 없으므로 권고를 생성하지 않는다.")
        return out

    cands = []
    for col, label, unit, steps, is_load in ACTUATORS:
        cols = [c for c in design_cols if c.split("__", 1)[-1].split("_lag")[0]
                .split("_ma")[0].split("_tr")[0] == col]
        if not cols:
            continue
        base = float(state.get(col, np.nan))
        if not np.isfinite(base):
            continue
        additive = unit == "℃"
        vals, preds = counterfactual_curve(model, x_row, cols, base, steps, additive)
        i0 = int(np.argmin(np.abs(np.asarray(steps, float) - (0.0 if additive else 1.0))))
        d = preds - preds[i0]
        for j, (v, dv) in enumerate(zip(vals, d)):
            if j == i0 or abs(dv) < 1e-6:
                continue
            direction_is_load_up = (v > base) if is_load else False
            if load_blocked and direction_is_load_up:
                continue
            cands.append({
                "variable": col, "label": label, "unit": unit,
                "current": round(base, 3), "target": round(float(v), 3),
                "change_pct": round(100 * (v - base) / base, 1) if base else None,
                "expected_delta_ch4_m3d": round(float(dv), 1),
                "expected_delta_pct": round(100 * float(dv) / pred_now, 2) if pred_now else None,
                "sensitivity_m3d_per_unit": round(float(dv / (v - base)), 1) if v != base else None,
            })

    cands.sort(key=lambda a: -abs(a["expected_delta_ch4_m3d"]))
    seen, picked = set(), []
    for a in cands:                      # 변수당 최선의 조작 하나씩만
        if a["variable"] in seen:
            continue
        seen.add(a["variable"])
        picked.append(a)
        if len(picked) >= max_actions:
            break
    out["actions"] = picked
    return out


def diagnose(state: pd.Series, baseline: pd.Series) -> list[dict]:
    """
    건강상태 진단 — 문헌 절대임계가 아니라 **시설 자체 90일 기준선 대비 상대편차**로 낸다.

    문헌 임계를 그대로 경보로 쓰면 이 시설에서는 pH 98.4 %, OLR 86.2 % 가 발화해
    경보가 죽는다. 이 시설은 pH 7.95 · OLR 1.14 로 6년간 안정 운전한 순치된 계이고,
    일반 기준 밖에 있는 것이 곧 이상은 아니다. 문헌 밴드는 등급이 아니라
    '설계·문헌 대비 어디에 있는지' 보여주는 참고로만 병기한다.
    """
    rows = []
    cau, dan = C.RELATIVE_ALARM["caution_pct"], C.RELATIVE_ALARM["danger_pct"]
    for col in ("VFA_ALK_A", "dig_pH_A", "dig_T_A_C", "OLR_calc", "FAN_calc",
                C.FLOW, C.CONC, "feed_AB_tpd"):
        v, b = state.get(col, np.nan), baseline.get(col, np.nan)
        if not (np.isfinite(v) and np.isfinite(b)) or b == 0:
            continue
        dev = 100.0 * (v - b) / abs(b)

        # 나쁜 방향으로의 이탈만 등급을 올린다. 절대값만 보면 개선이 경보가 된다.
        direction = C.ALARM_DIRECTION.get(col, "both")
        if direction == "up":
            mag = max(dev, 0.0)
        elif direction == "down":
            mag = max(-dev, 0.0)
        else:
            mag = abs(dev)
        level = "정상" if mag < cau else ("주의" if mag < dan else "위험")

        ref = ""
        gb = C.GUIDELINE.get(col if col != "OLR_calc" else "OLR_kgVS_m3d")
        if gb:
            lo, hi = gb["good"]
            ref = "지침 정상역" if lo <= v <= hi else f"지침 정상역({lo}~{hi}) 밖"
        rows.append({"variable": col, "value": round(float(v), 3),
                     "baseline_90d": round(float(b), 3),
                     "deviation_pct": round(float(dev), 1),
                     "direction": direction,
                     "level": level, "guideline_note": ref})

    # 열역학 검사 — Y_COD 는 0 < Y ≤ 0.35 밖이면 값이 아니라 계측 오류의 신호다.
    y = state.get("Y_COD", np.nan)
    if np.isfinite(y) and (y <= 0 or y > C.CH4_PER_KG_COD):
        rows.append({"variable": "Y_COD", "value": round(float(y), 3),
                     "baseline_90d": None, "deviation_pct": None, "direction": "both",
                     "level": "위험",
                     "guideline_note": f"열역학 상한 {C.CH4_PER_KG_COD} 초과 — "
                                       "COD·유량·가스 중 하나가 잘못 계측된 날이다"})
    return rows
