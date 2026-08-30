"""Development-only Phase 06 figures rendered with Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import ImageDraw

from .mechanistic_reporting import (
    EDGE,
    GRID,
    HEIGHT,
    INK,
    MUTED,
    PALETTE,
    PANEL,
    WIDTH,
    _canvas,
    _font,
    _line_panel,
    _panel,
    _target_label,
)
from .ml_statistical.contracts import FEATURE_SETS, TRACK_STRICT


MODEL_COLORS = {
    "S00_RIDGE": PALETTE[0],
    "S01_ELASTIC_NET": PALETTE[1],
    "S10_RANDOM_FOREST": PALETTE[2],
    "S20_XGBOOST": PALETTE[3],
    "S21_LIGHTGBM": PALETTE[4],
}


def _development_only(frame: pd.DataFrame, name: str) -> None:
    for column in frame.columns:
        lowered = str(column).lower()
        if any(token in lowered for token in ("date", "timestamp", "start", "end")):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if bool(parsed.dt.year.dropna().ge(2022).any()):
                raise AssertionError(f"{name}.{column} contains post-2021 dates")
    if "period" in frame:
        text = frame["period"].dropna().astype(str).str.upper()
        if bool(text.str.contains("2022|2023|VALIDATION|FINAL", regex=True).any()):
            raise AssertionError(f"{name} contains sealed period metadata")


def _locked_metrics(
    metrics: pd.DataFrame,
    locks: pd.DataFrame,
    *,
    support_type: str,
) -> pd.DataFrame:
    rows = metrics.loc[
        metrics["period"].eq("DEVELOPMENT")
        & metrics["support_type"].eq(support_type)
    ].copy()
    strict = locks.loc[locks["availability_track"].eq(TRACK_STRICT), [
        "target_name", "horizon", "model_id", "feature_set", "availability_track"
    ]]
    return rows.merge(
        strict,
        on=["target_name", "horizon", "model_id", "feature_set", "availability_track"],
        how="inner",
        validate="one_to_one",
    )


def _metric_lines(
    metrics: pd.DataFrame,
    locks: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    support_type: str,
    title: str,
    subtitle: str,
    ylabel: str,
    zero_line: bool,
) -> None:
    image, draw = _canvas(title, subtitle)
    rows = _locked_metrics(metrics, locks, support_type=support_type)
    targets = list(rows["target_name"].dropna().unique())[:2]
    for panel_index, target in enumerate(targets):
        series = []
        subset = rows.loc[rows["target_name"].eq(target)]
        for model_id in MODEL_COLORS:
            model = subset.loc[subset["model_id"].eq(model_id)]
            mapping = {
                int(row.horizon): float(getattr(row, metric))
                for row in model.itertuples(index=False)
                if pd.notna(getattr(row, metric))
            }
            if mapping:
                series.append((model_id.replace("S", "S", 1), mapping, MODEL_COLORS[model_id]))
        left = 55 + panel_index * 775
        _line_panel(
            draw,
            (left, 145, left + 735, 910),
            _target_label(target),
            ylabel,
            series,
            zero_line=zero_line,
            floor_zero=not zero_line,
        )
        draw.text((left + 430, 174), "h7 / h14 are primary", fill=MUTED, font=_font(11))
    image.save(path)


def _feature_set_increment(comparison: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Incremental feature-set value",
        "Development common-support RMSE improvement (before minus after); positive is better.",
    )
    targets = list(comparison["target_name"].dropna().unique())[:2]
    for panel_index, target in enumerate(targets):
        left = 55 + panel_index * 775
        box = (left, 145, left + 735, 910)
        _panel(draw, box, _target_label(target))
        subset = comparison.loc[
            comparison["target_name"].eq(target)
            & comparison["availability_track"].eq(TRACK_STRICT)
            & comparison["horizon"].isin([7, 14])
        ].copy()
        subset["improvement"] = -pd.to_numeric(subset["delta_RMSE"], errors="coerce")
        grouped = subset.groupby("to_set")["improvement"].median().reindex(FEATURE_SETS[1:])
        values = grouped.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        maximum = max(abs(finite).max(), 1.0) if finite.size else 1.0
        plot_left, plot_right = left + 90, left + 700
        plot_top, plot_bottom = 245, 790
        zero_y = (plot_top + plot_bottom) / 2
        draw.line((plot_left, zero_y, plot_right, zero_y), fill=INK, width=2)
        slot = (plot_right - plot_left) / len(grouped)
        for index, (name, value) in enumerate(grouped.items()):
            x0 = plot_left + index * slot + 18
            x1 = plot_left + (index + 1) * slot - 18
            if pd.notna(value):
                height = float(value) / maximum * (plot_bottom - plot_top) * 0.42
                y = zero_y - height
                color = PALETTE[2] if value >= 0 else PALETTE[3]
                draw.rounded_rectangle(
                    (x0, min(zero_y, y), x1, max(zero_y, y)), radius=5, fill=color
                )
                draw.text((x0, y - 23 if value >= 0 else y + 5), f"{value:+.1f}", fill=INK, font=_font(13))
            draw.text((x0, plot_bottom + 16), name, fill=MUTED, font=_font(13))
        draw.text((left + 85, 205), "Median across model families and h7/h14", fill=MUTED, font=_font(14))
    image.save(path)


def _fold_stability(
    fold_metrics: pd.DataFrame,
    locks: pd.DataFrame,
    path: Path,
) -> None:
    image, draw = _canvas(
        "Development OOF fold stability",
        "Primary horizons; each line is the lowest-development-RMSE locked STRICT candidate.",
    )
    own = fold_metrics.loc[fold_metrics["support_type"].eq("OWN_AVAILABLE")]
    for panel_index, target in enumerate(own["target_name"].dropna().unique()[:2]):
        left = 55 + panel_index * 775
        box = (left, 145, left + 735, 910)
        _panel(draw, box, _target_label(target))
        series: list[tuple[str, list[tuple[int, float]], str]] = []
        for h_index, horizon in enumerate((7, 14)):
            target_locks = locks.loc[
                locks["target_name"].eq(target)
                & locks["horizon"].eq(horizon)
                & locks["availability_track"].eq(TRACK_STRICT)
            ].sort_values("RMSE", kind="stable")
            if target_locks.empty:
                continue
            chosen = target_locks.iloc[0]
            rows = own.loc[
                own["target_name"].eq(target)
                & own["horizon"].eq(horizon)
                & own["model_id"].eq(chosen["model_id"])
                & own["feature_set"].eq(chosen["feature_set"])
                & own["availability_track"].eq(TRACK_STRICT)
            ].sort_values("outer_fold")
            points = [
                (int(str(row.outer_fold).split("_")[-1]), float(row.RMSE))
                for row in rows.itertuples(index=False)
                if pd.notna(row.RMSE)
            ]
            series.append((f"h{horizon} {chosen['model_id']}", points, PALETTE[h_index]))
        values = [value for _, points, _ in series for _, value in points]
        if not values:
            continue
        plot_left, plot_right = left + 90, left + 700
        plot_top, plot_bottom = 245, 790
        y_min, y_max = min(values), max(values)
        pad = max((y_max - y_min) * 0.1, 1.0)
        y_min, y_max = y_min - pad, y_max + pad
        xp = lambda value: plot_left + (value - 1) * (plot_right - plot_left) / 12
        yp = lambda value: plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)
        for tick in range(1, 14, 2):
            x = xp(tick)
            draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
            draw.text((x - 8, plot_bottom + 12), f"F{tick:02d}", fill=MUTED, font=_font(11))
        for tick in range(6):
            value = y_min + tick * (y_max - y_min) / 5
            y = yp(value)
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
            draw.text((left + 8, y - 8), f"{value:,.0f}", fill=MUTED, font=_font(12))
        for index, (label, points, color) in enumerate(series):
            pixel = [(xp(x), yp(y)) for x, y in points]
            if len(pixel) > 1:
                draw.line(pixel, fill=color, width=4)
            for x, y in pixel:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.line((left + 100 + index * 290, 850, left + 128 + index * 290, 850), fill=color, width=4)
            draw.text((left + 138 + index * 290, 842), label, fill=INK, font=_font(12))
    image.save(path)


def _top_feature_bars(
    feature_stability: pd.DataFrame,
    path: Path,
    *,
    value_column: str,
    title: str,
    subtitle: str,
) -> None:
    image, draw = _canvas(title, subtitle)
    strict = feature_stability.loc[
        feature_stability["availability_track"].eq(TRACK_STRICT)
        & feature_stability["horizon"].isin([7, 14])
    ].copy()
    targets = list(strict["target"].dropna().unique())[:2]
    for panel_index, target in enumerate(targets):
        left = 55 + panel_index * 775
        box = (left, 145, left + 735, 910)
        _panel(draw, box, _target_label(target))
        grouped = (
            strict.loc[strict["target"].eq(target)]
            .groupby("feature")[value_column]
            .median()
            .sort_values(ascending=False)
            .head(12)
            .sort_values()
        )
        maximum = max(float(grouped.max()) if len(grouped) else 0.0, 1.0e-12)
        plot_left, plot_right = left + 230, left + 700
        top, bottom = 215, 855
        slot = (bottom - top) / max(len(grouped), 1)
        for index, (feature, value) in enumerate(grouped.items()):
            y0 = top + index * slot + 5
            y1 = top + (index + 1) * slot - 5
            x1 = plot_left + float(value) / maximum * (plot_right - plot_left)
            draw.rounded_rectangle((plot_left, y0, x1, y1), radius=4, fill=PALETTE[index % 5])
            draw.text((left + 18, y0), str(feature)[:28], fill=INK, font=_font(11))
            draw.text((x1 + 5, y0), f"{float(value):.3g}", fill=MUTED, font=_font(10))
    image.save(path)


def _r2_scatter(metrics: pd.DataFrame, locks: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "R2 level versus R2 delta",
        "Locked STRICT development OOF candidates; upper-right is desirable but Skill remains separately audited.",
    )
    rows = _locked_metrics(metrics, locks, support_type="OWN_AVAILABLE")
    for panel_index, target in enumerate(rows["target_name"].dropna().unique()[:2]):
        left = 55 + panel_index * 775
        box = (left, 145, left + 735, 910)
        _panel(draw, box, _target_label(target))
        subset = rows.loc[rows["target_name"].eq(target)]
        x = pd.to_numeric(subset["R2_level"], errors="coerce")
        y = pd.to_numeric(subset["R2_delta"], errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y)
        subset, x, y = subset.loc[valid], x.loc[valid], y.loc[valid]
        if subset.empty:
            continue
        x_min, x_max = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
        y_min, y_max = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
        x_pad, y_pad = max((x_max - x_min) * 0.1, 0.05), max((y_max - y_min) * 0.1, 0.05)
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad
        plot_left, plot_right = left + 95, left + 700
        plot_top, plot_bottom = 230, 820
        xp = lambda value: plot_left + (value - x_min) * (plot_right - plot_left) / (x_max - x_min)
        yp = lambda value: plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)
        if x_min <= 0 <= x_max:
            draw.line((xp(0), plot_top, xp(0), plot_bottom), fill=GRID, width=2)
        if y_min <= 0 <= y_max:
            draw.line((plot_left, yp(0), plot_right, yp(0)), fill=GRID, width=2)
        for row in subset.itertuples(index=False):
            color = MODEL_COLORS.get(row.model_id, PALETTE[7])
            px, py = xp(float(row.R2_level)), yp(float(row.R2_delta))
            radius = 7 if int(row.horizon) in (7, 14) else 4
            draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color, outline="white")
        draw.text((plot_right - 90, 845), "R2_level", fill=MUTED, font=_font(13))
        draw.text((left + 20, 205), "R2_delta", fill=MUTED, font=_font(13))
    image.save(path)


def render_required_figures(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_comparison: pd.DataFrame,
    feature_stability: pd.DataFrame,
    locks: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    for name, frame in (
        ("metrics", metrics),
        ("fold_metrics", fold_metrics),
        ("feature_comparison", feature_comparison),
        ("feature_stability", feature_stability),
    ):
        _development_only(frame, name)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _metric_lines(
        metrics, locks, directory / "06_ml_rmse_by_horizon.png",
        metric="RMSE", support_type="OWN_AVAILABLE",
        title="Development OOF candidate comparison: RMSE",
        subtitle="Locked STRICT configurations only; no 2022/2023 performance is shown.",
        ylabel="RMSE", zero_line=False,
    )
    _metric_lines(
        metrics, locks, directory / "06_ml_skill_vs_persistence.png",
        metric="Skill_vs_persistence", support_type="PERSISTENCE_PAIRED",
        title="Development OOF Skill versus strict persistence",
        subtitle="Exact immutable Phase 04 support; positive Skill is better.",
        ylabel="Skill", zero_line=True,
    )
    _feature_set_increment(feature_comparison, directory / "06_feature_set_increment.png")
    _fold_stability(fold_metrics, locks, directory / "06_model_fold_stability.png")
    _top_feature_bars(
        feature_stability,
        directory / "06_feature_selection_frequency.png",
        value_column="selection_frequency",
        title="Feature selection frequency",
        subtitle="STRICT primary horizons; fold-local selection frequency is predictive, not causal evidence.",
    )
    _top_feature_bars(
        feature_stability,
        directory / "06_feature_importance_stability.png",
        value_column="importance_median",
        title="Feature importance stability",
        subtitle="Median fold importance; tree built-in importance is diagnostic only and never used to refit the same fold.",
    )
    _r2_scatter(metrics, locks, directory / "06_r2_level_vs_delta.png")


__all__ = ["render_required_figures"]
