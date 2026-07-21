"""STEP 7 — 파이프라인 조립 (PROMPT 1 §STEP 7).

build_pipeline(transform_key, filter_params, feature_blocks, ctx) -> sklearn.Pipeline
구조: FeatureUnion(블록들) → VarianceThreshold → Scaler
  · taxa 블록 = Filter → ZeroReplace(로그비 변환일 때만) → Transform
  · 전 과정이 단일 Pipeline 으로 직렬화되어 cross_val_score 에 그대로 투입 가능.
  · 모든 fit 은 넘겨받은 X(=학습 fold)만 본다 → 누수 없음.
"""
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold

from .filter import PrevalenceAbundanceFilter
from .zero_replace import MultiplicativeReplacement
from .transforms import TRANSFORM_REGISTRY, make_transform
from .derived import AlphaDiversity, BetaPCoA, DomainRatios, DesignFeatures

DEFAULT_BLOCKS = ["taxa", "alpha", "beta", "domain", "design"]


def _taxa_block(transform_key, filter_params, ctx):
    spec = TRANSFORM_REGISTRY[transform_key]
    fp = dict(filter_params or {})
    delta_factor = fp.pop("delta_factor", 0.65)
    steps = [("filter", PrevalenceAbundanceFilter(**fp))]
    if spec["needs_zero_replace"]:
        steps.append(("zero_replace", MultiplicativeReplacement(delta_factor=delta_factor)))
    steps.append(("transform", make_transform(transform_key, taxonomy=ctx.get("taxonomy"))))
    return Pipeline(steps)


def build_pipeline(transform_key, filter_params=None, feature_blocks=None, ctx=None,
                   scale=None):
    """전처리 파이프라인 생성.

    transform_key  : TRANSFORM_REGISTRY 키
    filter_params  : PrevalenceAbundanceFilter 파라미터(예: min_prevalence)
    feature_blocks : ['taxa','alpha','beta','domain','design'] 부분집합
    ctx            : dict(taxonomy=, domain_taxa=, meta=) 참조 테이블(학습 대상 아님)
    scale          : None(auto)/True/False
    """
    ctx = ctx or {}
    feature_blocks = feature_blocks or DEFAULT_BLOCKS
    fp = dict(filter_params or {})
    # filter 전용 파라미터만 남기고 delta_factor 는 분리
    fp.pop("delta_factor", None)

    union_parts = []
    for b in feature_blocks:
        if b == "taxa":
            union_parts.append(("taxa", _taxa_block(transform_key, fp, ctx)))
        elif b == "alpha":
            union_parts.append(("alpha", AlphaDiversity()))
        elif b == "beta":
            union_parts.append(("beta", BetaPCoA(n_axes=3)))
        elif b == "domain":
            union_parts.append(("domain", DomainRatios(
                taxonomy=ctx.get("taxonomy"), domain_taxa=ctx.get("domain_taxa"))))
        elif b == "design":
            union_parts.append(("design", DesignFeatures(meta=ctx.get("meta"))))
        else:
            raise ValueError(f"알 수 없는 블록: {b}")

    union = FeatureUnion(union_parts)

    # 스케일 결정
    reg_scaler = TRANSFORM_REGISTRY[transform_key]["scaler"]
    if scale is None:
        use_scale = not (feature_blocks == ["taxa"] and reg_scaler == "none")
    else:
        use_scale = bool(scale)
    scaler = StandardScaler(with_mean=True) if use_scale else \
        FunctionTransformer(feature_names_out="one-to-one")

    pipe = Pipeline([
        ("features", union),
        ("var", VarianceThreshold(0.0)),
        ("scaler", scaler),
    ])
    return pipe


class PipelineFactory:
    """직렬화 가능한 파이프라인 팩토리 (artifacts/pipeline_factory.pkl).

    참조 테이블(taxonomy/metadata/domain_taxa)만 담고 있으며 학습 파라미터는 없다.
    PROMPT 2 는 이 팩토리를 언피클해 `build(transform_key, ...)` 로 파이프라인을 만든다.
    """
    def __init__(self, taxonomy, domain_taxa, meta):
        self.ctx = dict(taxonomy=taxonomy, domain_taxa=domain_taxa, meta=meta)
        self.transform_keys = list(TRANSFORM_REGISTRY.keys())
        self.default_blocks = list(DEFAULT_BLOCKS)

    def build(self, transform_key, filter_params=None, feature_blocks=None, scale=None):
        return build_pipeline(transform_key, filter_params, feature_blocks,
                              ctx=self.ctx, scale=scale)


def get_output_feature_names(fitted_pipeline):
    """적합된 전처리 파이프라인의 최종 출력 특징 이름."""
    try:
        return list(fitted_pipeline.get_feature_names_out())
    except Exception:
        # 폴백: features union 이름만
        return list(fitted_pipeline.named_steps["features"].get_feature_names_out())
