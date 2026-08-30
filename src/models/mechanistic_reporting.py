"""Dependency-light Phase 05 figures rendered with Pillow only.

The renderer is intentionally defensive: Phase 05 figures are development-OOF
artifacts, so any visible 2022/2023 period or date metadata is rejected before a
pixel is written.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 980
HORIZONS = (1, 3, 7, 14, 30)

BG = "#F7F8FA"
PANEL = "#FFFFFF"
INK = "#172033"
MUTED = "#5D6678"
GRID = "#D9DEE8"
EDGE = "#E1E5EC"
ZERO = "#8993A4"
PALETTE = (
    "#0B6E99",
    "#D97706",
    "#2F855A",
    "#C24156",
    "#6B46C1",
    "#008C8C",
    "#A05A2C",
    "#4A5568",
)


@lru_cache(maxsize=None)
def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((64, 40), title, fill=INK, font=_font(34, bold=True))
    draw.text((64, 88), subtitle, fill=MUTED, font=_font(19))
    return image, draw


def _required(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    assert not missing, f"{name} is missing required columns: {missing}"


def _assert_development_only(name: str, frame: pd.DataFrame) -> None:
    """Reject validation/final-test metadata without inspecting target values."""

    if frame.empty:
        return

    date_tokens = ("date", "start", "end", "timestamp")
    for column in frame.columns:
        lower = str(column).lower()
        if any(token in lower for token in date_tokens):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            years = parsed.dt.year.dropna()
            assert not years.ge(2022).any(), (
                f"{name}.{column} contains a 2022/2023 row; Phase 05 figures "
                "must be development-only"
            )

    metadata_columns = {
        "period",
        "evaluation_period",
        "split",
        "dataset_split",
        "year",
        "outer_fold",
        "fold",
    }
    forbidden = re.compile(r"2022|2023|VALIDATION|FINAL[_ -]?TEST", re.IGNORECASE)
    for column in frame.columns:
        if str(column).lower() not in metadata_columns:
            continue
        text = frame[column].dropna().astype(str)
        assert not text.str.contains(forbidden, regex=True).any(), (
            f"{name}.{column} contains sealed validation/test metadata"
        )


def _finite(values: Iterable[object]) -> list[float]:
    output: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        number = float(value)
        if np.isfinite(number):
            output.append(number)
    return output


def _target_label(value: object) -> str:
    text = str(value)
    return "CH4" if "CH4" in text.upper() else "Total biogas"


def _model_color(model: object) -> str:
    text = str(model)
    fixed = {
        "M1F": PALETTE[0],
        "M1TS": PALETTE[1],
        "M1VS": PALETTE[2],
        "M1TS2": PALETTE[4],
        "M2": PALETTE[3],
    }
    if text in fixed:
        return fixed[text]
    return PALETTE[sum(ord(char) for char in text) % len(PALETTE)]


def _natural_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", str(value))
    )


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str = "",
) -> None:
    draw.rounded_rectangle(box, radius=16, fill=PANEL, outline=EDGE, width=2)
    draw.text((box[0] + 22, box[1] + 16), title, fill=INK, font=_font(22, bold=True))
    if subtitle:
        draw.text((box[0] + 22, box[1] + 48), subtitle, fill=MUTED, font=_font(14))


def _line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    series: list[tuple[str, dict[int, float], str]],
    *,
    zero_line: bool = False,
    floor_zero: bool = False,
) -> None:
    _panel(draw, box, title)
    left, top, right, bottom = box
    plot_left, plot_top = left + 94, top + 86
    plot_right, plot_bottom = right - 32, bottom - 96
    values = _finite(value for _, mapping, _ in series for value in mapping.values())
    if not values:
        draw.text((plot_left, plot_top), "No eligible development values", fill=MUTED, font=_font(18))
        return

    y_min, y_max = min(values), max(values)
    if floor_zero:
        y_min = min(0.0, y_min)
    if zero_line:
        y_min, y_max = min(0.0, y_min), max(0.0, y_max)
    if np.isclose(y_min, y_max):
        pad = max(1.0, abs(y_max) * 0.1)
        y_min, y_max = y_min - pad, y_max + pad
    else:
        pad = 0.09 * (y_max - y_min)
        y_min, y_max = y_min - pad, y_max + pad

    def xp(horizon: int) -> float:
        index = HORIZONS.index(int(horizon))
        return plot_left + index * (plot_right - plot_left) / (len(HORIZONS) - 1)

    def yp(value: float) -> float:
        return plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)

    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = yp(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((left + 8, y - 9), f"{value:,.2f}", fill=MUTED, font=_font(13))
    if zero_line and y_min <= 0 <= y_max:
        draw.line((plot_left, yp(0), plot_right, yp(0)), fill=ZERO, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    draw.text((left + 8, top + 53), ylabel, fill=MUTED, font=_font(13))
    for horizon in HORIZONS:
        x = xp(horizon)
        draw.text((x - 11, plot_bottom + 13), str(horizon), fill=MUTED, font=_font(15))
    draw.text((plot_right - 75, plot_bottom + 42), "Horizon (d)", fill=MUTED, font=_font(14))

    for label, mapping, color in series:
        points = [
            (xp(horizon), yp(float(mapping[horizon])))
            for horizon in HORIZONS
            if horizon in mapping and pd.notna(mapping[horizon])
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2)

    legend_x, legend_y = left + 22, bottom - 48
    for index, (label, _, color) in enumerate(series[:6]):
        x = legend_x + (index % 3) * 220
        y = legend_y + (index // 3) * 25
        draw.line((x, y + 8, x + 25, y + 8), fill=color, width=4)
        draw.text((x + 34, y), str(label)[:20], fill=INK, font=_font(13))


def _development_metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics.copy()
    if "period" in rows.columns:
        period = rows["period"].astype(str)
        development = period.str.upper().eq("DEVELOPMENT")
        if development.any():
            rows = rows[development]
    return rows


def _preferred_support(rows: pd.DataFrame, *, skill: bool) -> pd.DataFrame:
    if "support_type" not in rows.columns or rows.empty:
        return rows
    support = rows["support_type"].astype(str).str.upper()
    if skill:
        preferred = support.str.contains("PERSISTENCE")
    else:
        preferred = support.eq("OWN_AVAILABLE") | support.str.contains("OWN_SUPPORT")
    return rows[preferred] if preferred.any() else rows


def _metric_series(
    metrics: pd.DataFrame,
    target: object,
    metric: str,
) -> list[tuple[str, dict[int, float], str]]:
    subset = metrics[metrics["target_name"].astype(str).eq(str(target))]
    output: list[tuple[str, dict[int, float], str]] = []
    for model in sorted(subset["model_id"].dropna().unique(), key=_natural_key)[:6]:
        rows = subset[subset["model_id"].eq(model)]
        grouped = rows.groupby("horizon", dropna=False)[metric].mean()
        mapping = {int(h): float(value) for h, value in grouped.items() if pd.notna(value)}
        output.append((str(model), mapping, _model_color(model)))
    return output


def _kernel_shapes(kernel_audit: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "HRT-free first-order daily kernels",
        "Raw integrated daily-bin weights; sum(weights) + audited tail mass = 1.",
    )
    box = (65, 145, 1535, 910)
    _panel(draw, box, "Kernel shapes", "Vertical axis is log10(weight); no HRT is implied.")
    left, top, right, bottom = 155, 235, 1480, 715
    ordered = kernel_audit.sort_values("k").drop_duplicates("k")
    max_lag = max(1, int(ordered["max_lag"].max()))
    curves: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    logs: list[float] = []
    for index, row in enumerate(ordered.itertuples(index=False)):
        k = float(row.k)
        lags = np.arange(int(row.max_lag) + 1, dtype=float)
        weights = np.exp(-k * lags) - np.exp(-k * (lags + 1.0))
        log_weights = np.log10(np.maximum(weights, np.finfo(float).tiny))
        curves.append((k, lags, log_weights, PALETTE[index % len(PALETTE)]))
        logs.extend(log_weights.tolist())
    y_min, y_max = min(logs), max(logs)

    def xp(value: float) -> float:
        return left + value * (right - left) / max_lag

    def yp(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    for value in np.linspace(0, max_lag, 6):
        x = xp(float(value))
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x - 13, bottom + 12), f"{value:.0f}", fill=MUTED, font=_font(14))
    for value in np.linspace(y_min, y_max, 6):
        y = yp(float(value))
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((88, y - 8), f"{value:.1f}", fill=MUTED, font=_font(14))
    draw.line((left, top, left, bottom), fill=INK, width=2)
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.text((right - 80, bottom + 43), "Lag (d)", fill=MUTED, font=_font(14))
    draw.text((88, top - 28), "log10(w)", fill=MUTED, font=_font(14))
    for k, lags, values, color in curves:
        points = [(xp(float(lag)), yp(float(value))) for lag, value in zip(lags, values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)

    legend_y = 750
    for index, (k, _, _, color) in enumerate(curves):
        x = 105 + index * 195
        draw.line((x, legend_y + 8, x + 28, legend_y + 8), fill=color, width=4)
        draw.text((x + 38, legend_y), f"k={k:.2f} d^-1", fill=INK, font=_font(14))

    table_y = 810
    for index, row in enumerate(ordered.itertuples(index=False)):
        x = 85 + index * 210
        draw.text((x, table_y), f"K={int(row.max_lag)}", fill=INK, font=_font(13, bold=True))
        draw.text((x, table_y + 23), f"Σw={float(row.sum_weights):.6f}", fill=MUTED, font=_font(12))
        draw.text((x, table_y + 43), f"tail={float(row.tail_mass):.6f}", fill=MUTED, font=_font(12))
    image.save(path)


def _k_selection_frequency(parameter_stability: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Fold-local k selection frequency",
        "Development outer folds only; frequency is descriptive and does not identify a true hydrolysis constant.",
    )
    targets = list(parameter_stability["target"].dropna().unique())[:2]
    if not targets:
        targets = ["No target"]
    for panel_index, target in enumerate(targets):
        x0 = 60 + panel_index * 770
        box = (x0, 150, x0 + (730 if len(targets) > 1 else 1480), 910)
        _panel(draw, box, _target_label(target))
        subset = parameter_stability[
            parameter_stability["target"].astype(str).eq(str(target))
        ]
        numeric_k = pd.to_numeric(subset["selected_k"], errors="coerce")
        counts = numeric_k.dropna().value_counts().sort_index()
        if counts.empty:
            draw.text((x0 + 80, 245), "No selected-k rows", fill=MUTED, font=_font(18))
            continue
        left, top, right, bottom = x0 + 90, 235, box[2] - 45, 805
        max_count = max(1, int(counts.max()))
        n = len(counts)
        slot = (right - left) / n
        for tick in range(6):
            value = tick * max_count / 5
            y = bottom - value * (bottom - top) / max_count
            draw.line((left, y, right, y), fill=GRID, width=1)
            draw.text((x0 + 25, y - 8), f"{value:.0f}", fill=MUTED, font=_font(13))
        total = int(counts.sum())
        for index, (k, count) in enumerate(counts.items()):
            bx0 = left + index * slot + slot * 0.17
            bx1 = left + (index + 1) * slot - slot * 0.17
            by = bottom - float(count) * (bottom - top) / max_count
            color = PALETTE[index % len(PALETTE)]
            draw.rounded_rectangle((bx0, by, bx1, bottom), radius=5, fill=color)
            draw.text((bx0, bottom + 14), f"{float(k):.2f}", fill=MUTED, font=_font(14))
            draw.text((bx0, by - 38), f"{int(count)}", fill=INK, font=_font(15, bold=True))
            draw.text((bx0, by - 19), f"{100*count/total:.1f}%", fill=MUTED, font=_font(12))
        draw.text((right - 55, bottom + 46), "k", fill=MUTED, font=_font(14))
    image.save(path)


def _metrics_figure(
    metrics: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    title: str,
    subtitle: str,
    ylabel: str,
    skill: bool,
) -> None:
    image, draw = _canvas(title, subtitle)
    rows = _preferred_support(_development_metric_rows(metrics), skill=skill)
    targets = list(rows["target_name"].dropna().unique())[:2]
    for panel_index, target in enumerate(targets):
        left = 55 + panel_index * 775
        _line_panel(
            draw,
            (left, 145, left + 735, 910),
            _target_label(target),
            ylabel,
            _metric_series(rows, target, metric),
            zero_line=skill,
            floor_zero=not skill,
        )
    if not targets:
        draw.text((110, 220), "No development metric rows", fill=MUTED, font=_font(20))
    image.save(path)


def _parameter_stability(parameter_stability: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Selected-k stability across outer folds",
        "Cells show fold-local effective k; two-pool rows show k_fast; blank cells were unavailable.",
    )
    targets = list(parameter_stability["target"].dropna().unique())[:2]
    for panel_index, target in enumerate(targets):
        x0 = 45 + panel_index * 775
        box = (x0, 145, x0 + 745, 925)
        _panel(draw, box, _target_label(target))
        subset = parameter_stability[
            parameter_stability["target"].astype(str).eq(str(target))
        ].copy()
        subset["row_label"] = subset["model_id"].astype(str) + " h" + subset["horizon"].astype(str)
        rows = sorted(subset["row_label"].dropna().unique(), key=_natural_key)
        folds = sorted(subset["outer_fold"].dropna().unique(), key=_natural_key)
        if not rows or not folds:
            draw.text((x0 + 80, 235), "No parameter-stability rows", fill=MUTED, font=_font(18))
            continue
        max_rows = 25
        rows = rows[:max_rows]
        grid_left, grid_top = x0 + 205, 215
        grid_right, grid_bottom = box[2] - 24, box[3] - 45
        cell_w = (grid_right - grid_left) / len(folds)
        cell_h = (grid_bottom - grid_top) / len(rows)
        numeric_selected_k = pd.to_numeric(subset["selected_k"], errors="coerce")
        if "k_fast" in subset.columns:
            # Two-pool audit rows store ``slow/fast`` as a pair string; use
            # k_fast for the heatmap color and retain the full pair in CSV.
            numeric_selected_k = numeric_selected_k.fillna(
                pd.to_numeric(subset["k_fast"], errors="coerce")
            )
        kvals = _finite(numeric_selected_k)
        kmin, kmax = min(kvals, default=0.0), max(kvals, default=1.0)
        span = kmax - kmin if kmax > kmin else 1.0
        subset["selected_k_plot"] = numeric_selected_k
        lookup = subset.groupby(["row_label", "outer_fold"])["selected_k_plot"].first()
        for j, fold in enumerate(folds):
            label = str(fold).replace("OUTER_", "F")
            draw.text((grid_left + j * cell_w + 2, grid_top - 27), label[-5:], fill=MUTED, font=_font(10))
        for i, row_label in enumerate(rows):
            y = grid_top + i * cell_h
            draw.text((x0 + 18, y + 2), str(row_label)[:22], fill=INK, font=_font(max(9, min(12, int(cell_h - 3)))))
            for j, fold in enumerate(folds):
                x = grid_left + j * cell_w
                value = lookup.get((row_label, fold), np.nan)
                if pd.isna(value):
                    fill = "#F0F2F5"
                    label = ""
                else:
                    ratio = (float(value) - kmin) / span
                    fill = (
                        int(223 - 118 * ratio),
                        int(235 - 72 * ratio),
                        int(244 - 63 * ratio),
                    )
                    label = f"{float(value):.2f}" if cell_w >= 30 and cell_h >= 18 else ""
                draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=fill, outline="white")
                if label:
                    draw.text((x + 2, y + 2), label, fill=INK, font=_font(9))
    image.save(path)


def _prediction_rmse(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.copy()
    rows["family"] = np.where(
        rows["model_id"].astype(str).str.upper().str.contains("M1TS2|TWO"),
        "Two-pool TS",
        np.where(
            rows["model_id"].astype(str).str.upper().str.contains("M1TS"),
            "Single-pool TS",
            "Other",
        ),
    )
    rows = rows[rows["family"].ne("Other")]
    finite = np.isfinite(pd.to_numeric(rows["y_true"], errors="coerce")) & np.isfinite(
        pd.to_numeric(rows["y_pred"], errors="coerce")
    )
    rows = rows[finite].copy()
    rows["sq_error"] = (rows["y_pred"].astype(float) - rows["y_true"].astype(float)) ** 2
    if rows.empty:
        return pd.DataFrame(columns=["target_name", "horizon", "family", "RMSE"])
    output = rows.groupby(["target_name", "horizon", "family"], as_index=False)["sq_error"].mean()
    output["RMSE"] = np.sqrt(output.pop("sq_error"))
    return output


def _single_two_pool(predictions: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Single-pool versus optional two-pool TS",
        "Development OOF RMSE; lower is better. Complexity is not promoted by fit alone.",
    )
    summary = _prediction_rmse(predictions)
    targets = list(summary["target_name"].dropna().unique())[:2]
    colors = {"Single-pool TS": PALETTE[0], "Two-pool TS": PALETTE[1]}
    for panel_index, target in enumerate(targets):
        subset = summary[summary["target_name"].astype(str).eq(str(target))]
        series = []
        for family in ("Single-pool TS", "Two-pool TS"):
            rows = subset[subset["family"].eq(family)]
            mapping = dict(zip(rows["horizon"].astype(int), rows["RMSE"].astype(float)))
            series.append((family, mapping, colors[family]))
        left = 55 + panel_index * 775
        _line_panel(
            draw,
            (left, 145, left + 735, 910),
            _target_label(target),
            "RMSE",
            series,
            floor_zero=True,
        )
    if not targets:
        draw.text((110, 220), "No paired single/two-pool development predictions", fill=MUTED, font=_font(20))
    image.save(path)


def _selected_residual_rows(residual_selection: pd.DataFrame) -> pd.DataFrame:
    rows = residual_selection.copy()
    selected = rows["selected_tau"]
    if pd.api.types.is_bool_dtype(selected):
        return rows[selected.fillna(False)]
    numeric = pd.to_numeric(selected, errors="coerce")
    unique = set(numeric.dropna().unique().tolist())
    if unique and unique.issubset({0.0, 1.0}):
        return rows[numeric.eq(1.0)]
    tau = pd.to_numeric(rows["tau_res"], errors="coerce")
    equality = numeric.notna() & tau.eq(numeric)
    return rows[equality] if equality.any() else rows[numeric.notna()]


def _residual_inertia(residual_selection: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas(
        "Constrained residual inertia: M1 versus M2",
        "Development inner/outer selections only; M2 target-history dependence remains explicit.",
    )
    selected = _selected_residual_rows(residual_selection)
    targets = list(selected["target"].dropna().unique())[:2]
    for panel_index, target in enumerate(targets):
        subset = selected[selected["target"].astype(str).eq(str(target))]
        grouped = subset.groupby("horizon", as_index=False).agg(
            RMSE_M1=("RMSE_M1", "mean"),
            RMSE_M2=("RMSE_M2", "mean"),
            tau=("tau_res", "median"),
        )
        m1 = dict(zip(grouped["horizon"].astype(int), grouped["RMSE_M1"].astype(float)))
        m2 = dict(zip(grouped["horizon"].astype(int), grouped["RMSE_M2"].astype(float)))
        left = 55 + panel_index * 775
        _line_panel(
            draw,
            (left, 145, left + 735, 910),
            _target_label(target),
            "RMSE",
            [("M1 base", m1, PALETTE[0]), ("M2 residual", m2, PALETTE[3])],
            floor_zero=True,
        )
        tau_text = ", ".join(
            f"h{int(row.horizon)}:{float(row.tau):g}d" for row in grouped.itertuples(index=False)
        )
        draw.text(
            (left + 185, 203),
            f"tau by horizon  {tau_text}"[:66],
            fill=MUTED,
            font=_font(11),
        )
    if not targets:
        draw.text((110, 220), "No selected residual-inertia rows", fill=MUTED, font=_font(20))
    image.save(path)


def render_required_figures(
    kernel_audit: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    residual_selection: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    """Render the seven required Phase 05 PNGs from development-only rows."""

    _required(kernel_audit, ("k", "max_lag", "sum_weights", "tail_mass"), "kernel_audit")
    _required(
        predictions,
        (
            "target_name",
            "horizon",
            "model_id",
            "outer_fold",
            "k",
            "k_fast",
            "k_slow",
            "y_true",
            "y_pred",
        ),
        "predictions",
    )
    _required(
        metrics,
        (
            "target_name",
            "horizon",
            "model_id",
            "period",
            "support_type",
            "RMSE",
            "Skill_vs_persistence",
        ),
        "metrics",
    )
    _required(
        parameter_stability,
        (
            "target",
            "horizon",
            "model_id",
            "outer_fold",
            "selected_k",
            "beta0",
            "beta1",
            "RMSE",
            "Skill",
        ),
        "parameter_stability",
    )
    _required(
        residual_selection,
        (
            "target",
            "horizon",
            "outer_fold",
            "base_model_id",
            "tau_res",
            "RMSE_M1",
            "RMSE_M2",
            "selected_tau",
        ),
        "residual_selection",
    )

    for name, frame in (
        ("kernel_audit", kernel_audit),
        ("predictions", predictions),
        ("metrics", metrics),
        ("parameter_stability", parameter_stability),
        ("residual_selection", residual_selection),
    ):
        _assert_development_only(name, frame)

    output_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _kernel_shapes(kernel_audit, output_dir / "05_kernel_shapes.png")
    _k_selection_frequency(parameter_stability, output_dir / "05_k_selection_frequency.png")
    _metrics_figure(
        metrics,
        output_dir / "05_mechanistic_oof_by_horizon.png",
        metric="RMSE",
        title="Mechanistic development OOF performance",
        subtitle="Horizon-specific RMSE on each model's recorded development support.",
        ylabel="RMSE",
        skill=False,
    )
    _metrics_figure(
        metrics,
        output_dir / "05_skill_vs_persistence.png",
        metric="Skill_vs_persistence",
        title="Mechanistic Skill versus strict persistence",
        subtitle="Exact Phase 04 persistence-paired support; positive Skill is better.",
        ylabel="Skill",
        skill=True,
    )
    _parameter_stability(parameter_stability, output_dir / "05_parameter_stability.png")
    _single_two_pool(predictions, output_dir / "05_single_vs_two_pool.png")
    _residual_inertia(residual_selection, output_dir / "05_M1_vs_M2_residual_inertia.png")
