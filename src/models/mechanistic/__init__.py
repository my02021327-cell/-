"""HRT-free process-informed mechanistic model primitives for Phase 05."""

from .kernels import (
    K_CANDIDATES,
    KERNEL_TAIL_TOL,
    NORMALIZATION_ATOL,
    FirstOrderKernel,
    build_first_order_kernel,
    exact_daily_weights,
    first_order_daily_weights,
    kernel_audit_frame,
    minimal_inclusive_max_lag,
)
from .kinetic_state import (
    MIN_KERNEL_COVERAGE,
    build_kinetic_state,
    coverage_normalized_kinetic_state,
    validate_daily_load,
)
from .nonnegative_fit import (
    NonnegativeLinearFit,
    PhysicalPredictions,
    fit_nonnegative_least_squares,
    fit_nonnegative_linear,
    predict_nonnegative_linear,
)
from .single_pool import (
    SinglePoolModel,
    fit_single_pool,
    predict_single_pool,
)
from .two_pool import (
    MIN_RATE_RATIO,
    TwoPoolModel,
    fit_two_pool,
    predict_two_pool,
    two_pool_candidate_pairs,
    validate_two_pool_rates,
)


__all__ = [
    "K_CANDIDATES",
    "KERNEL_TAIL_TOL",
    "MIN_KERNEL_COVERAGE",
    "MIN_RATE_RATIO",
    "NORMALIZATION_ATOL",
    "FirstOrderKernel",
    "NonnegativeLinearFit",
    "PhysicalPredictions",
    "SinglePoolModel",
    "TwoPoolModel",
    "build_first_order_kernel",
    "build_kinetic_state",
    "coverage_normalized_kinetic_state",
    "exact_daily_weights",
    "first_order_daily_weights",
    "fit_nonnegative_least_squares",
    "fit_nonnegative_linear",
    "fit_single_pool",
    "fit_two_pool",
    "kernel_audit_frame",
    "minimal_inclusive_max_lag",
    "predict_nonnegative_linear",
    "predict_single_pool",
    "predict_two_pool",
    "two_pool_candidate_pairs",
    "validate_daily_load",
    "validate_two_pool_rates",
]
