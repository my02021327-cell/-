"""리포트/의사결정 로그 보조 유틸."""
import os, json, datetime
import numpy as np

DECISION_LOG = "reports/decision_log.md"


def log_decision(stage, msg):
    os.makedirs("reports", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_needed = not os.path.exists(DECISION_LOG)
    with open(DECISION_LOG, "a") as f:
        if header_needed:
            f.write("# 데이터 스누핑 / 의사결정 로그 (decision_log)\n\n"
                    "결과를 본 뒤 설정을 바꾼 횟수가 많을수록 최종 성능은 낙관 편향된다. "
                    "본 로그는 최종 보고서에 첨부된다.\n\n"
                    "| 시각 | 단계 | 내용 |\n|---|---|---|\n")
        f.write(f"| {ts} | {stage} | {msg} |\n")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
