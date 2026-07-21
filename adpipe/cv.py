"""중첩 교차검증 프로토콜 (PROMPT 2 §1). 확정 후 절대 변경 금지.

외부: LeaveOneGroupOut(site) = 10 fold
내부: GroupKFold(5) on training sites → (변환 × 필터 × 하이퍼파라미터) 선택
반복: seed 5개(내부 분할/모델 시드) → 분산 추정
"""
import warnings
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, KFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .models import MODEL_REGISTRY

INNER_MIN_PREV = [0.1, 0.25]     # 필터 강도도 내부에서 튜닝(고정 금지)


def make_estimator(model_key, transform, filter_params, model_kwargs, factory,
                   feature_blocks=None):
    spec = MODEL_REGISTRY[model_key]
    pre = factory.build(transform, filter_params=filter_params, feature_blocks=feature_blocks)
    est = spec["build"](**model_kwargs)
    return SkPipeline([("pre", pre), ("model", est)])


def _predict(est, X):
    p = est.predict(X)
    return np.asarray(p).ravel()


def _pooled_oof_r2(est_factory, X, y, groups, splitter):
    """splitter 로 OOF 예측을 모아 pooled R² 반환."""
    oof = np.full(len(y), np.nan)
    Xv = X
    for tr, te in splitter.split(Xv, y, groups):
        est = est_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xv.iloc[tr], np.asarray(y)[tr])
            oof[te] = _predict(est, Xv.iloc[te])
    mask = ~np.isnan(oof)
    if mask.sum() < 3 or np.var(np.asarray(y)[mask]) == 0:
        return -np.inf, oof
    return r2_score(np.asarray(y)[mask], oof[mask]), oof


def _eval_grid(model_key, factory, X, y, groups, folds, seed, feature_blocks):
    """주어진 folds 로 (transform, min_prev, model_kwargs) 그리드를 평가하고 최적 반환.

    전처리(변환)는 (tf, min_prev, fold) 당 1회만 계산해 캐시하고, 모델 하이퍼파라미터
    후보는 캐시된 변환 행렬 위에서 estimator 만 재적합한다(속도).
    """
    spec = MODEL_REGISTRY[model_key]
    param_keys = _param_keys(spec["build"])
    Xtr = X
    ytr = np.asarray(y)
    best = (-np.inf, None)
    for tf in spec["transforms"]:
        for mp in INNER_MIN_PREV:
            fp = {"min_prevalence": mp}
            # (1) 전처리 캐시: fold 별 (Ztr, ytr_idx, Zval, val_idx)
            cached = []
            ok = True
            for tri, vai in folds:
                pre = factory.build(tf, filter_params=fp, feature_blocks=feature_blocks)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Ztr = pre.fit_transform(Xtr.iloc[tri], ytr[tri])
                        Zval = pre.transform(Xtr.iloc[vai])
                except Exception:
                    ok = False
                    break
                cached.append((Ztr, ytr[tri], Zval, vai))
            if not ok or not cached:
                continue
            # (2) 모델 하이퍼파라미터 후보를 캐시 위에서 평가
            for mk in spec["grid"]:
                mk = dict(mk)
                if "random_state" in param_keys:
                    mk.setdefault("random_state", seed)
                oof = np.full(len(ytr), np.nan)
                good = True
                for Ztr, ytri, Zval, vai in cached:
                    est = spec["build"](**mk)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            est.fit(Ztr, ytri)
                            oof[vai] = _predict(est, Zval)
                    except Exception:
                        good = False
                        break
                if not good:
                    continue
                m = ~np.isnan(oof)
                if m.sum() < 3 or np.var(ytr[m]) == 0:
                    continue
                score = r2_score(ytr[m], oof[m])
                if score > best[0]:
                    best = (score, dict(transform=tf, filter_params=fp, model_kwargs=mk))
    default = dict(transform=spec["transforms"][0],
                   filter_params={"min_prevalence": 0.25},
                   model_kwargs=dict(spec["grid"][0]))
    return best[0], (best[1] or default)


def inner_select(model_key, factory, Xtr, ytr, gtr, seed, feature_blocks):
    """내부 GroupKFold(5)로 (transform, min_prev, model_kwargs) 최적 조합 선택."""
    n_inner = min(5, len(np.unique(gtr)))
    if n_inner < 2:
        n_inner = 2
    folds = list(GroupKFold(n_splits=n_inner).split(Xtr, ytr, gtr))
    _, cfg = _eval_grid(model_key, factory, Xtr, ytr, gtr, folds, seed, feature_blocks)
    return cfg


def optimistic_best(model_key, factory, data, feature_blocks=None, seed=0):
    """⚠ 낙관(선택편향) 추정: 전체 데이터에 LOGO 를 돌려 config 를 test 를 보고 고른 값.

    탐색용 그리드 표에만 사용하며 '승자의 저주로 낙관 편향'을 명시한다.
    """
    X, y, groups = data["X"], np.asarray(data["y"]), data["groups"].values
    folds = list(LeaveOneGroupOut().split(X, y, groups))
    score, cfg = _eval_grid(model_key, factory, X, y, groups, folds, seed, feature_blocks)
    return score, cfg


def _param_keys(build):
    """estimator 파라미터 키 집합(random_state 수용 여부 판별용)."""
    try:
        return set(build().get_params().keys())
    except Exception:
        return set()


def nested_cv_oof(model_key, factory, data, feature_blocks=None, seeds=(0, 1, 2, 3, 4),
                  verbose=False):
    """중첩 CV: seed 별 외부 LOGO OOF 예측 + 선택된 config 로그.

    반환: dict(oof={seed: array}, chosen=[(seed,test_site,config)...], y, groups)
    """
    X, y, groups = data["X"], np.asarray(data["y"]), data["groups"].values
    logo = LeaveOneGroupOut()
    oof_by_seed, chosen = {}, []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for tr, te in logo.split(X, y, groups):
            Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
            cfg = inner_select(model_key, factory, Xtr, ytr, gtr, seed, feature_blocks)
            mk = dict(cfg["model_kwargs"])
            est = make_estimator(model_key, cfg["transform"], cfg["filter_params"], mk,
                                 factory, feature_blocks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(Xtr, ytr)
                oof[te] = _predict(est, X.iloc[te])
            chosen.append(dict(seed=int(seed), test_site=str(groups[te][0]),
                               transform=cfg["transform"],
                               min_prevalence=cfg["filter_params"]["min_prevalence"],
                               model_kwargs=dict(cfg["model_kwargs"])))
        oof_by_seed[seed] = oof
        if verbose:
            print(f"  [{model_key}] seed {seed}: pooled R²="
                  f"{r2_score(y, oof):.3f}")
    return dict(oof=oof_by_seed, chosen=chosen, y=y, groups=groups)


# ------------------------------- metrics -------------------------------------
def modal_config(model_key, chosen):
    """중첩 CV 가 fold 별로 고른 config 중 최빈 조합을 '고정 config'로 요약.

    ⚠ 다운스트림(순열검정/중요도/학습곡선)은 단일 파이프라인이 필요하므로 최빈 config 를
    쓴다. 최종 '성능' 수치는 여전히 중첩 CV 외부 루프 값이며, 이 고정 config 는 해석·검정용.
    """
    from collections import Counter
    tf = Counter(c["transform"] for c in chosen).most_common(1)[0][0]
    mp = Counter(c["min_prevalence"] for c in chosen).most_common(1)[0][0]
    mk_key = Counter(repr(sorted(c["model_kwargs"].items())) for c in chosen).most_common(1)[0][0]
    mk = dict(eval(mk_key))
    return dict(model_key=model_key, transform=tf,
                filter_params={"min_prevalence": mp}, model_kwargs=mk)


def metrics_from_oof(y, oof):
    m = ~np.isnan(oof)
    yy, pp = np.asarray(y)[m], oof[m]
    return dict(r2=r2_score(yy, pp),
                rmse=float(np.sqrt(mean_squared_error(yy, pp))),
                mae=float(mean_absolute_error(yy, pp)))


def seed_r2_summary(res):
    """seed 별 pooled R² 의 평균/표준편차/구간."""
    y = res["y"]
    r2s = [r2_score(y, res["oof"][s]) for s in res["oof"]]
    r2s = np.array(r2s)
    return dict(mean=float(r2s.mean()), sd=float(r2s.std(ddof=1) if len(r2s) > 1 else 0.0),
                values=r2s.tolist())


def per_site_scores(y, oof, groups, metric="mae"):
    """site 별 점수(부트스트랩/Wilcoxon 용). MAE(기본) 또는 절대오차 평균."""
    out = {}
    for g in np.unique(groups):
        m = (groups == g) & (~np.isnan(oof))
        if m.sum() == 0:
            continue
        err = np.abs(np.asarray(y)[m] - oof[m])
        out[g] = float(err.mean())
    return out


# --------------------- 누수 진단용: 무작위 K-fold(라벨=누수) -----------------
def random_kfold_r2(est_factory, X, y, seed=0, k=5):
    """무작위 K-fold OOF R² (site 무시). ⚠ 누수된 낙관 추정치."""
    y = np.asarray(y)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        est = est_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X.iloc[tr], y[tr])
            oof[te] = _predict(est, X.iloc[te])
    return r2_score(y, oof)
