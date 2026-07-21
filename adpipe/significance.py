"""통계적 유의성 검정 (PROMPT 2 §4). 성능표만으로는 부족하다."""
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score
from sklearn.metrics import r2_score
from scipy import stats

from .cv import make_estimator


def fixed_estimator(cfg, factory, feature_blocks=None):
    return make_estimator(cfg["model_key"], cfg["transform"], cfg["filter_params"],
                          cfg["model_kwargs"], factory, feature_blocks)


def permutation_test(cfg, factory, data, n_perms=500, feature_blocks=None, seed=0):
    """순열 검정: 타깃을 site 그룹 구조 보존한 채 셔플, LOGO 로 귀무분포 생성.

    p>0.05 이면 그 모델은 신호를 못 잡은 것.
    """
    X, y, groups = data["X"], np.asarray(data["y"]), data["groups"].values
    est = fixed_estimator(cfg, factory, feature_blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, perm_scores, pvalue = permutation_test_score(
            est, X, y, groups=groups, cv=LeaveOneGroupOut(),
            scoring="r2", n_permutations=n_perms, n_jobs=1, random_state=seed)
    return dict(score=float(score), pvalue=float(pvalue),
                perm_mean=float(np.mean(perm_scores)),
                perm_scores=np.asarray(perm_scores))


def bootstrap_r2_ci(y, oof, groups, n_boot=2000, alpha=0.05, seed=0):
    """site 단위 부트스트랩으로 R² 신뢰구간(percentile). 반복측정 구조 반영."""
    y = np.asarray(y)
    m = ~np.isnan(oof)
    sites = np.unique(groups[m])
    by_site = {s: np.where((groups == s) & m)[0] for s in sites}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(sites, size=len(sites), replace=True)
        idx = np.concatenate([by_site[s] for s in pick])
        yy, pp = y[idx], oof[idx]
        if np.var(yy) == 0:
            continue
        boots.append(r2_score(yy, pp))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = r2_score(y[m], oof[m])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                boot_mean=float(boots.mean()))


def model_comparison(per_site_mae, winner, alpha=0.05):
    """모델별 site 단위 MAE 에 대해 winner 와의 Wilcoxon signed-rank + BH 보정.

    ⚠ 10 fold(site)로 다수 모델 비교는 검정력이 매우 낮다 — 결론 전 반드시 명시.
    """
    sites = sorted(set.intersection(*[set(d.keys()) for d in per_site_mae.values()]))
    w = np.array([per_site_mae[winner][s] for s in sites])
    rows, pvals, others = [], [], []
    for name, d in per_site_mae.items():
        if name == winner:
            continue
        v = np.array([d[s] for s in sites])
        try:
            stat, p = stats.wilcoxon(w, v)
        except ValueError:
            p = 1.0
        pvals.append(p); others.append(name)
        rows.append(dict(model=name, mae_mean=float(v.mean()),
                         winner_mae=float(w.mean()), pvalue=float(p)))
    # BH 보정
    if pvals:
        order = np.argsort(pvals)
        m = len(pvals)
        bh = np.empty(m)
        prev = 1.0
        for rank, idx in enumerate(order[::-1]):
            k = m - rank
            prev = min(prev, pvals[idx] * m / k)
            bh[idx] = prev
        for i, r in enumerate(rows):
            r["pvalue_bh"] = float(bh[i])
    return dict(sites=sites, n_folds=len(sites), rows=rows)


def learning_curve_sites(cfg, factory, data, sizes=range(3, 10), n_draws=8,
                         feature_blocks=None, seed=0):
    """site 수를 늘리며 held-out site 성능 추이. 포화 안하면 표본부족이 병목."""
    X, y, groups = data["X"], np.asarray(data["y"]), data["groups"].values
    all_sites = np.unique(groups)
    rng = np.random.default_rng(seed)
    curve = []
    for k in sizes:
        if k >= len(all_sites):
            break
        scores = []
        for _ in range(n_draws):
            perm = rng.permutation(all_sites)
            train_sites, test_sites = perm[:k], perm[k:]
            tr = np.isin(groups, train_sites); te = np.isin(groups, test_sites)
            est = fixed_estimator(cfg, factory, feature_blocks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X.iloc[tr], y[tr])
                pred = np.asarray(est.predict(X.iloc[te])).ravel()
            if np.var(y[te]) > 0:
                scores.append(r2_score(y[te], pred))
        if scores:
            curve.append((k, float(np.mean(scores)), float(np.std(scores))))
    return curve
