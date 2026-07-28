"""이화학 외생변수 실패가 계측 단절 탓인지 폴드별로 분해한다.

전체 폴드를 합쳐 보면 이화학 구성이 R2 -26 까지 무너진다. 폴드 3 이
2022 Q4 계측 단절(알칼리도 반감, 온도 -4.7σ)을 가로지르므로, 단절 이전
폴드에서도 같은 붕괴가 일어나는지 확인해야 원인을 특정할 수 있다.
실행: `python -m src.chem_perfold`
"""
import warnings

warnings.filterwarnings("ignore")

import src.arimax_chem as ac
from src.final_ensemble import prepare


def main():
    endog, X = ac.frames()
    d = prepare()
    d = d[(d.index.year >= 2018) & (d.index.year <= 2023)]
    for k, (cut, end) in enumerate(ac.FOLDS, 1):
        print(f"폴드{k}: 학습 ~{d.index[cut-1].date()} / "
              f"평가 {d.index[cut].date()}~{d.index[min(end, len(d))-1].date()}")
    print()
    for k, f in enumerate(ac.FOLDS, 1):
        print(f"── 폴드 {k} ──")
        for name, (p, c) in ac.VARIANTS.items():
            r = ac.run_variant(endog, X, p, c, folds=[f])
            print(f"   {name:18s} " +
                  "  ".join(f"h{h}={v[0]:7.3f}" for h, v in sorted(r.items())))
        print()


if __name__ == "__main__":
    main()
