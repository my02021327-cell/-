"""누수 없음(no-leakage) 단위 테스트 (PROMPT 1 §STEP 7, 필수 통과).

증명 전략: fit(train) 이후 transform(test)의 각 테스트 행 출력은 '학습 fold 파라미터'와
'그 행 자신'에만 의존해야 한다. 따라서 테스트셋의 *다른* 행을 임의로 교란해도 특정 테스트
행의 변환 결과는 변하지 않아야 한다(테스트셋 전역 통계를 참조하지 않음).
"""
import os, sys
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import warnings
import numpy as np
import pytest

from adpipe.config import get_modeling_data
from adpipe.transforms import TRANSFORM_REGISTRY
from adpipe.pipeline import build_pipeline, PipelineFactory

_D = get_modeling_data("genus")
_CTX = dict(taxonomy=_D["taxonomy"], domain_taxa=_D["domain_taxa"], meta=_D["meta"])


def _split():
    groups = _D["groups"]
    test_site = list(dict.fromkeys(groups.tolist()))[0]
    tr = groups != test_site
    te = groups == test_site
    return _D["X"][tr.values], _D["X"][te.values], _D["y"][tr.values]


@pytest.mark.parametrize("key", list(TRANSFORM_REGISTRY.keys()))
def test_transform_is_rowwise_given_fit(key):
    """다른 테스트 행 교란이 대상 행 변환을 바꾸지 않음 → 테스트 전역통계 미참조."""
    Xtr, Xte, ytr = _split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline(key, filter_params={"min_prevalence": 0.25}, ctx=_CTX)
        pipe.fit(Xtr, ytr)
        Z0 = pipe.transform(Xte)

        # 대상 행(0) 은 두고, 나머지 테스트 행을 크게 교란
        Xte_p = Xte.copy()
        rng = np.random.default_rng(0)
        if len(Xte_p) > 1:
            noise = rng.random((len(Xte_p) - 1, Xte_p.shape[1])) * 100
            Xte_p.iloc[1:, :] = noise
        Z1 = pipe.transform(Xte_p)

    assert Z0.shape == Z1.shape
    # 대상 행(0) 의 변환이 불변이어야 한다
    np.testing.assert_allclose(Z0[0], Z1[0], rtol=1e-6, atol=1e-8,
                               err_msg=f"[{key}] 테스트셋 다른 행 교란이 대상 행 변환을 바꿈(누수)")


def test_fit_params_independent_of_test():
    """test 를 fit 에 절대 넣지 않음: train-only fit 후 test 교란해도 학습 파라미터 불변."""
    Xtr, Xte, ytr = _split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline("clr", filter_params={"min_prevalence": 0.25}, ctx=_CTX)
        pipe.fit(Xtr, ytr)
        scaler = pipe.named_steps["scaler"]
        mean_before = np.array(scaler.mean_, copy=True)
        # 테스트 변환을 여러 번(교란 포함) 수행
        _ = pipe.transform(Xte)
        Xte_p = Xte.copy(); Xte_p.iloc[:, :] = 999.0
        _ = pipe.transform(Xte_p)
        mean_after = np.array(scaler.mean_)
    np.testing.assert_allclose(mean_before, mean_after, rtol=0, atol=0,
                               err_msg="transform 이 학습된 scaler 파라미터를 변경함")


def test_filter_kept_from_train_only():
    """필터 통과 taxa 는 학습 fold 에서만 결정된다."""
    Xtr, Xte, ytr = _split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline("clr", filter_params={"min_prevalence": 0.25}, ctx=_CTX)
        pipe.fit(Xtr, ytr)
        filt = pipe.named_steps["features"].transformer_list[0][1].named_steps["filter"]
    # 통과 목록이 train 유병률로 재현되는지
    from adpipe.filter import PrevalenceAbundanceFilter
    ref = PrevalenceAbundanceFilter(min_prevalence=0.25).fit(Xtr)
    assert set(filt.kept_) == set(ref.kept_)


def test_factory_pickle_roundtrip(tmp_path):
    import pickle
    f = PipelineFactory(_D["taxonomy"], _D["domain_taxa"], _D["meta"])
    b = pickle.dumps(f)
    f2 = pickle.loads(b)
    Xtr, Xte, ytr = _split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = f2.build("clr", filter_params={"min_prevalence": 0.25})
        pipe.fit(Xtr, ytr)
        Z = pipe.transform(Xte)
    assert Z.shape[0] == len(Xte)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
