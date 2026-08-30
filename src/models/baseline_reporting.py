"""Dependency-light Phase 04 charts rendered with Pillow.

The runtime intentionally does not require matplotlib.  These figures expose only
development/validation metrics and never accept sealed-test rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .baseline_contract import HORIZONS


WIDTH = 1600
HEIGHT = 980
BG = "#F7F8FA"
INK = "#172033"
GRID = "#D9DEE8"
MUTED = "#5D6678"
PALETTE = [
    "#0B6E99",
    "#D97706",
    "#2F855A",
    "#C24156",
    "#6B46C1",
    "#008C8C",
    "#A05A2C",
    "#4A5568",
    "#D53F8C",
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((64, 42), title, fill=INK, font=_font(34, bold=True))
    draw.text((64, 91), subtitle, fill=MUTED, font=_font(20))
    return image, draw


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if pd.notna(value) and np.isfinite(value)]


def _line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    series: list[tuple[str, list[float], str]],
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top = left + 95, top + 92
    plot_right, plot_bottom = right - 35, bottom - 78
    draw.rounded_rectangle(box, radius=16, fill="white", outline="#E1E5EC", width=2)
    draw.text((left + 24, top + 18), title, fill=INK, font=_font(23, bold=True))
    all_values = _finite(value for _, values, _ in series for value in values)
    if not all_values:
        draw.text((plot_left, plot_top), "No eligible values", fill=MUTED, font=_font(18))
        return
    y_min = min(0.0, min(all_values))
    y_max = max(all_values)
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    margin = 0.08 * (y_max - y_min)
    y_min -= margin
    y_max += margin

    def xp(index: int) -> float:
        return plot_left + index * (plot_right - plot_left) / (len(HORIZONS) - 1)

    def yp(value: float) -> float:
        return plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)

    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = yp(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((left + 10, y - 10), f"{value:,.1f}", fill=MUTED, font=_font(15))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    for index, horizon in enumerate(HORIZONS):
        x = xp(index)
        draw.text((x - 10, plot_bottom + 14), str(horizon), fill=MUTED, font=_font(16))
    draw.text((plot_right - 75, plot_bottom + 45), "Horizon (d)", fill=MUTED, font=_font(15))
    draw.text((left + 10, top + 53), ylabel, fill=MUTED, font=_font(15))

    legend_x = left + 26
    legend_y = bottom - 40
    for label, values, color in series:
        points = [(xp(i), yp(float(value))) for i, value in enumerate(values) if pd.notna(value)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2)
        draw.line((legend_x, legend_y + 8, legend_x + 28, legend_y + 8), fill=color, width=4)
        draw.text((legend_x + 38, legend_y - 2), label, fill=INK, font=_font(15))
        legend_x += 220


def _metric_vector(
    metrics: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    period: str,
    support: str,
    metric: str,
) -> list[float]:
    subset = metrics[
        metrics["target_name"].eq(target)
        & metrics["baseline_id"].eq(baseline)
        & metrics["period"].eq(period)
        & metrics["support_type"].eq(support)
    ].set_index("horizon")
    return [subset.at[horizon, metric] if horizon in subset.index else np.nan for horizon in HORIZONS]


def persistence_horizon_figure(metrics: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Strict persistence degrades with forecast horizon",
        "RMSE on split-contained target dates; the 2023 final test is sealed and excluded.",
    )
    for panel_index, target in enumerate(("CH4_m3d_observed", "biogas_AB_m3d")):
        left = 55 + panel_index * 775
        series = []
        for period, color in (("DEVELOPMENT", PALETTE[0]), ("2022_VALIDATION", PALETTE[1])):
            series.append(
                (
                    period.replace("2022_", ""),
                    _metric_vector(
                        metrics,
                        target=target,
                        baseline="B10_PERSISTENCE_STRICT",
                        period=period,
                        support="OWN_AVAILABLE",
                        metric="RMSE",
                    ),
                    color,
                )
            )
        _line_panel(
            draw,
            (left, 145, left + 735, 905),
            "CH4" if target.startswith("CH4") else "Total biogas",
            "RMSE (reported volume basis)",
            series,
        )
    image.save(path)


def baseline_rmse_figure(metrics: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "2022 validation baseline RMSE by horizon",
        "Exact strict-persistence paired support; every cell uses identical origin-target pairs.",
    )
    baseline_ids = list(
        metrics.loc[
            metrics["period"].eq("2022_VALIDATION")
            & metrics["support_type"].eq("PERSISTENCE_PAIRED"),
            "baseline_id",
        ].drop_duplicates()
    )
    short_names = {
        "B00_EXPANDING_MEAN": "B00  EXP MEAN",
        "B01_EXPANDING_MEDIAN": "B01  EXP MEDIAN",
        "B10_PERSISTENCE_STRICT": "B10  STRICT PERSIST",
        "B11_LAST_OBSERVED": "B11  LAST OBSERVED",
        "B20_MOVING_AVG_7D": "B20  MOVING AVG 7D",
        "B21_MOVING_AVG_14D": "B21  MOVING AVG 14D",
        "B22_MOVING_AVG_30D": "B22  MOVING AVG 30D",
        "B30_WEEKLY_SEASONAL_NAIVE": "B30  WEEKLY NAIVE",
        "B31_ANNUAL_SEASONAL_NAIVE": "B31  ANNUAL NAIVE",
    }
    for panel_index, target in enumerate(("CH4_m3d_observed", "biogas_AB_m3d")):
        x0 = 55 + panel_index * 775
        draw.rounded_rectangle((x0, 145, x0 + 735, 915), radius=16, fill="white", outline="#E1E5EC", width=2)
        draw.text((x0 + 24, 165), "CH4" if panel_index == 0 else "Total biogas", fill=INK, font=_font(23, bold=True))
        draw.text((x0 + 24, 199), "RMSE on persistence-paired support", fill=MUTED, font=_font(15))
        cell_w, cell_h = 100, 62
        grid_x, grid_y = x0 + 205, 250
        for j, horizon in enumerate(HORIZONS):
            draw.text((grid_x + j * cell_w + 30, grid_y - 30), f"h{horizon}", fill=MUTED, font=_font(15, bold=True))
        target_rows = metrics[
            metrics["target_name"].eq(target)
            & metrics["period"].eq("2022_VALIDATION")
            & metrics["support_type"].eq("PERSISTENCE_PAIRED")
        ].set_index(["baseline_id", "horizon"])
        all_values = _finite(target_rows["RMSE"])
        value_min, value_max = min(all_values), max(all_values)
        span = value_max - value_min if value_max > value_min else 1.0
        for i, baseline in enumerate(baseline_ids):
            y = grid_y + i * cell_h
            draw.text((x0 + 22, y + 18), short_names.get(baseline, baseline[:20]), fill=INK, font=_font(14))
            for j, horizon in enumerate(HORIZONS):
                value = target_rows.loc[(baseline, horizon), "RMSE"]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                ratio = 0.5 if pd.isna(value) else float((value - value_min) / span)
                color = (
                    int(92 + 151 * ratio),
                    int(211 - 84 * ratio),
                    int(153 - 91 * ratio),
                )
                x = grid_x + j * cell_w
                draw.rounded_rectangle((x, y, x + cell_w - 8, y + cell_h - 8), radius=6, fill=color)
                label = "NA" if pd.isna(value) else f"{float(value):,.0f}"
                bbox = draw.textbbox((0, 0), label, font=_font(14, bold=True))
                text_w = bbox[2] - bbox[0]
                draw.text((x + (cell_w - 8 - text_w) / 2, y + 16), label, fill=INK, font=_font(14, bold=True))
    image.save(path)


def r2_level_delta_figure(metrics: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Level R2 does not imply change-prediction skill",
        "Strict persistence, 2022 validation. Each label is forecast horizon in days.",
    )
    subset = metrics[
        metrics["baseline_id"].eq("B10_PERSISTENCE_STRICT")
        & metrics["period"].eq("2022_VALIDATION")
        & metrics["support_type"].eq("OWN_AVAILABLE")
    ].copy()
    box = (110, 160, 1490, 900)
    draw.rounded_rectangle(box, radius=16, fill="white", outline="#E1E5EC", width=2)
    left, top, right, bottom = 225, 230, 1430, 790
    x_values = _finite(subset["R2_level"])
    y_values = _finite(subset["R2_delta"])
    x_min, x_max = min(-0.5, min(x_values, default=-0.5)), max(1.0, max(x_values, default=1.0))
    y_min, y_max = min(-0.1, min(y_values, default=-0.1)), max(0.1, max(y_values, default=0.1))

    def xp(value: float) -> float:
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def yp(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    for value in np.linspace(x_min, x_max, 7):
        x = xp(float(value))
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x - 18, bottom + 12), f"{value:.1f}", fill=MUTED, font=_font(15))
    for value in np.linspace(y_min, y_max, 6):
        y = yp(float(value))
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((135, y - 8), f"{value:.3f}", fill=MUTED, font=_font(15))
    if x_min <= 0 <= x_max:
        draw.line((xp(0), top, xp(0), bottom), fill="#9AA3B2", width=2)
    if y_min <= 0 <= y_max:
        draw.line((left, yp(0), right, yp(0)), fill="#9AA3B2", width=2)
    draw.line((left, top, left, bottom), fill=INK, width=2)
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.text((right - 85, bottom + 48), "R2 level", fill=MUTED, font=_font(18))
    draw.text((125, top - 35), "R2 delta", fill=MUTED, font=_font(18))

    for target_index, target in enumerate(("CH4_m3d_observed", "biogas_AB_m3d")):
        color = PALETTE[target_index]
        rows = subset[subset["target_name"].eq(target)]
        for row in rows.itertuples(index=False):
            if pd.isna(row.R2_level) or pd.isna(row.R2_delta):
                continue
            x, y = xp(float(row.R2_level)), yp(float(row.R2_delta))
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color, outline="white", width=3)
            draw.text((x + 14, y - 12), f"h{row.horizon}", fill=color, font=_font(17, bold=True))
        legend_y = 835 + target_index * 28
        draw.ellipse((235, legend_y, 253, legend_y + 18), fill=color)
        label = "CH4" if target_index == 0 else "Total biogas"
        draw.text((265, legend_y - 3), label, fill=INK, font=_font(17))
    image.save(path)


def coverage_figure(metrics: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "2022 validation scoring coverage",
        "OWN_AVAILABLE support. CH4 coverage varies with observation availability; 2023 is excluded.",
    )
    subset = metrics[
        metrics["period"].eq("2022_VALIDATION")
        & metrics["support_type"].eq("OWN_AVAILABLE")
    ].copy()
    baselines = list(subset["baseline_id"].drop_duplicates())
    for panel_index, target in enumerate(("CH4_m3d_observed", "biogas_AB_m3d")):
        x0 = 65 + panel_index * 770
        draw.rounded_rectangle((x0, 145, x0 + 730, 910), radius=16, fill="white", outline="#E1E5EC", width=2)
        draw.text((x0 + 24, 165), "CH4" if panel_index == 0 else "Total biogas", fill=INK, font=_font(23, bold=True))
        cell_w, cell_h = 104, 62
        grid_x, grid_y = x0 + 190, 230
        for j, horizon in enumerate(HORIZONS):
            draw.text((grid_x + j * cell_w + 33, grid_y - 30), f"h{horizon}", fill=MUTED, font=_font(15, bold=True))
        target_rows = subset[subset["target_name"].eq(target)].set_index(["baseline_id", "horizon"])
        for i, baseline in enumerate(baselines):
            y = grid_y + i * cell_h
            draw.text((x0 + 18, y + 18), baseline.split("_")[0], fill=MUTED, font=_font(15, bold=True))
            draw.text((x0 + 62, y + 18), "_".join(baseline.split("_")[1:])[:15], fill=INK, font=_font(13))
            for j, horizon in enumerate(HORIZONS):
                value = np.nan
                if (baseline, horizon) in target_rows.index:
                    value = target_rows.loc[(baseline, horizon), "coverage_rate"]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0]
                coverage = 0.0 if pd.isna(value) else float(np.clip(value, 0, 1))
                red = int(244 - 150 * coverage)
                green = int(227 - 60 * (1 - coverage))
                blue = int(214 - 75 * coverage)
                color = (red, green, blue)
                x = grid_x + j * cell_w
                draw.rounded_rectangle((x, y, x + cell_w - 8, y + cell_h - 8), radius=6, fill=color)
                label = "NA" if pd.isna(value) else f"{100*coverage:.1f}%"
                draw.text((x + 17, y + 16), label, fill=INK, font=_font(14, bold=True))
    image.save(path)


def render_required_figures(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Render all required figures after asserting sealed-period absence."""

    if metrics["period"].astype(str).str.contains("2023|FINAL_TEST", case=False).any():
        raise ValueError("sealed 2023 metric row supplied to figure renderer")
    output_dir.mkdir(parents=True, exist_ok=True)
    persistence_horizon_figure(metrics, output_dir / "04_persistence_horizon_performance.png")
    baseline_rmse_figure(metrics, output_dir / "04_baseline_rmse_by_horizon.png")
    r2_level_delta_figure(metrics, output_dir / "04_r2_level_vs_delta.png")
    coverage_figure(metrics, output_dir / "04_baseline_coverage.png")
