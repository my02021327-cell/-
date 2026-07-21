"""특징 중요도 — 안정성 기준 (PROMPT 2 §5).

단일 fit 중요도는 n=40 에서 무의미. 50회(5 seed × 10 fold) 순열 중요도(테스트셋)와
선택 빈도(stability selection)를 집계해 70% 이상에서 상위권에 든 특징만 보고한다.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut

from .cv import make_estimator
from .pipeline import get_output_feature_names


def stability_selection(cfg, factory, data, seeds=(0, 1, 2, 3, 4), top_k=10,
                        feature_blocks=None, n_repeats=10):
    """반환: taxa 수준 순열 중요도 선택빈도표 + (선형모델시) 계수 부호 일관성표."""
    X, y, groups = data["X"], np.asarray(data["y"]), data["groups"].values
    taxa_cols = list(X.columns)
    topk_count = {c: 0 for c in taxa_cols}
    total_fits = 0
    # 선형 계수 부호 집계(전처리 특징명 기준)
    sign_pos, sign_neg, appear = {}, {}, {}

    for seed in seeds:
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            est = make_estimator(cfg["model_key"], cfg["transform"], cfg["filter_params"],
                                 dict(cfg["model_kwargs"]), factory, feature_blocks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X.iloc[tr], y[tr])
                # (1) taxa 수준 순열 중요도 — 테스트 fold 에서
                r = permutation_importance(est, X.iloc[te], y[te], n_repeats=n_repeats,
                                           random_state=seed, scoring="r2")
            imp = pd.Series(r.importances_mean, index=taxa_cols)
            for c in imp.nlargest(top_k).index:
                topk_count[c] += 1
            total_fits += 1
            # (2) 선형 계수 부호 일관성
            model = est.named_steps["model"]
            coef = getattr(model, "coef_", None)
            if coef is not None:
                coef = np.asarray(coef).ravel()
                try:
                    names = list(est.named_steps["pre"].get_feature_names_out())
                except Exception:
                    names = [f"f{i}" for i in range(len(coef))]
                if len(names) == len(coef):
                    for nm, cf in zip(names, coef):
                        appear[nm] = appear.get(nm, 0) + 1
                        if cf > 0:
                            sign_pos[nm] = sign_pos.get(nm, 0) + 1
                        elif cf < 0:
                            sign_neg[nm] = sign_neg.get(nm, 0) + 1

    freq = pd.DataFrame({
        "taxon": taxa_cols,
        "selection_freq": [topk_count[c] / total_fits for c in taxa_cols],
    }).sort_values("selection_freq", ascending=False).reset_index(drop=True)

    sign_rows = []
    for nm, ap in appear.items():
        pos = sign_pos.get(nm, 0); neg = sign_neg.get(nm, 0)
        consistency = max(pos, neg) / ap if ap else 0
        sign_rows.append(dict(feature=nm, appear=ap,
                              frac_appear=ap / total_fits,
                              sign=("+" if pos >= neg else "-"),
                              sign_consistency=consistency,
                              nonzero_freq=(pos + neg) / total_fits))
    sign_df = (pd.DataFrame(sign_rows).sort_values("nonzero_freq", ascending=False)
               .reset_index(drop=True)) if sign_rows else pd.DataFrame()
    return dict(freq=freq, total_fits=total_fits, sign=sign_df, top_k=top_k)
