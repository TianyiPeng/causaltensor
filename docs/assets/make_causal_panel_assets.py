"""Generate the README visuals for the causal panel problem.

Run from the repository root with:

    python docs/assets/make_causal_panel_assets.py

The script writes:
    docs/assets/causal_panel_problem.png
    docs/assets/causal_panel_decomposition.gif
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image


OUT_DIR = Path(__file__).resolve().parent
STATIC_PATH = OUT_DIR / "causal_panel_problem.png"
GIF_PATH = OUT_DIR / "causal_panel_decomposition.gif"

INK = "#1F2937"
MUTED = "#64748B"
GRID = "#CBD5E1"
BLUE = "#2563EB"
RED = "#D1495B"
GREEN = "#2A9D8F"
BG = "#FBFCFE"


def build_example_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic toy panel with staggered treatment adoption."""
    rng = np.random.default_rng(8)
    n_units = 72
    n_periods = 112
    time = np.arange(n_periods)
    scaled_time = np.linspace(0, 1, n_periods)
    scaled_units = np.linspace(0, 1, n_units)[:, None]

    unit_level = -0.55 + 1.45 * scaled_units
    unit_slope = (0.34 - 0.52 * scaled_units) * scaled_time[None, :]
    common_trend = 0.9 * scaled_time + 0.24 * np.sin(2 * np.pi * scaled_time)
    latent_time_factor = np.cos(2.7 * np.pi * scaled_time)
    latent_unit_factor = -0.65 + 1.45 * scaled_units
    smooth_unit_noise = 0.018 * np.convolve(rng.normal(size=n_units), np.ones(9) / 9, mode="same")[:, None]
    smooth_time_noise = 0.012 * np.convolve(rng.normal(size=n_periods), np.ones(13) / 13, mode="same")[None, :]

    baseline = (
        1.35
        + unit_level
        + unit_slope
        + common_trend[None, :]
        + 0.32 * latent_unit_factor * latent_time_factor[None, :]
        + 0.08 * np.sin(3.5 * np.pi * scaled_units) * np.sin(2 * np.pi * scaled_time)[None, :]
        + smooth_unit_noise
        + smooth_time_noise
    )

    treatment = np.zeros_like(baseline)
    treated_units = np.arange(28, n_units)
    adoption_base = np.linspace(40, 88, len(treated_units))
    adoption_jitter = 2 * np.sin(np.linspace(0, 2.5 * np.pi, len(treated_units)))
    adoption_times = np.rint(adoption_base + adoption_jitter).astype(int)
    for unit, start in zip(treated_units, adoption_times):
        treatment[unit, start:] = 1

    effect = np.zeros_like(baseline)
    for unit, start in zip(treated_units, adoption_times):
        horizon = np.maximum(0, time - start)
        unit_progress = (unit - treated_units[0]) / max(len(treated_units) - 1, 1)
        level = 0.34 + 0.52 * unit_progress
        ramp = 1 - np.exp(-horizon / 4.5)
        effect[unit, start:] = level * ramp[start:] + 0.012 * horizon[start:]

    observed = baseline + effect
    return baseline, treatment, effect, observed, time


def style_axes(ax: plt.Axes, xlabel: str = "time", ylabel: str = "units") -> None:
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
        spine.set_linewidth(0.9)


def outline_treated_region(ax: plt.Axes, treatment: np.ndarray, alpha: float = 0.68) -> None:
    x = np.arange(treatment.shape[1])
    y = np.arange(treatment.shape[0])
    ax.contour(x, y, treatment, levels=[0.5], colors=[RED], linewidths=1.1, alpha=alpha)


def draw_matrix(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    cmap: str | ListedColormap,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
    treatment: np.ndarray | None = None,
    interpolation: str = "bicubic",
) -> None:
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap).copy()
    else:
        cmap_obj = cmap

    data = values
    if mask is not None:
        data = np.ma.masked_where(~mask.astype(bool), values)
        cmap_obj.set_bad("#F8FAFC")

    ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation=interpolation,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10.5, color=INK, pad=8, weight="semibold")
    style_axes(ax)
    if treatment is not None:
        outline_treated_region(ax, treatment)


def draw_gap_plot(
    ax: plt.Axes,
    baseline: np.ndarray,
    treatment: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    *,
    stage: int = 4,
    show_xlabel: bool = True,
) -> None:
    treated_units = treatment.any(axis=1)
    treatment_share = treatment[treated_units].mean(axis=0)
    baseline_mean = baseline[treated_units].mean(axis=0)
    observed_mean = observed[treated_units].mean(axis=0)

    for idx, share in enumerate(treatment_share):
        if stage >= 1 and share > 0:
            ax.axvspan(idx - 0.5, idx + 0.5, color=RED, alpha=0.04 + 0.10 * share, lw=0)

    ax.plot(time, baseline_mean, color=BLUE, lw=2.3, label="Counterfactual mean")

    if stage >= 2:
        ax.plot(time, observed_mean, color=RED, lw=2.3, label="Observed treated mean")

    if stage >= 3:
        ax.fill_between(
            time,
            baseline_mean,
            observed_mean,
            where=treatment_share > 0,
            color=RED,
            alpha=0.20,
            interpolate=True,
            label="Estimated treatment gap",
        )

    ax.set_title("Average treated trajectory", fontsize=10.5, color=INK, weight="semibold")
    ax.set_xlabel("time" if show_xlabel else "", color=MUTED, fontsize=9)
    ax.set_ylabel("outcome", color=MUTED, fontsize=9)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def draw_static_asset() -> None:
    baseline, treatment, effect, observed, time = build_example_panel()
    value_min = min(baseline.min(), observed.min())
    value_max = max(baseline.max(), observed.max())

    fig = plt.figure(figsize=(12.2, 6.9), dpi=170, facecolor=BG)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.12], hspace=0.36, wspace=0.24)

    effect_mask = treatment.astype(bool)
    treatment_cmap = ListedColormap(["#E8EEF7", RED])

    draw_matrix(
        fig.add_subplot(grid[0, 0]),
        baseline,
        "Counterfactual baseline $M$",
        "viridis",
        vmin=value_min,
        vmax=value_max,
    )
    draw_matrix(
        fig.add_subplot(grid[0, 1]),
        treatment,
        "Treatment mask $Z$",
        treatment_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    draw_matrix(
        fig.add_subplot(grid[0, 2]),
        effect,
        "Treatment surface $\\tau Z$",
        "magma",
        vmin=0,
        vmax=max(effect.max(), 0.1),
        mask=effect_mask,
    )
    draw_matrix(
        fig.add_subplot(grid[0, 3]),
        observed,
        "Observed panel $O$",
        "viridis",
        vmin=value_min,
        vmax=value_max,
        treatment=treatment,
    )

    ax_gap = fig.add_subplot(grid[1, :])
    draw_gap_plot(ax_gap, baseline, treatment, observed, time, stage=4)

    fig.suptitle(
        "Causal panel data: observed outcomes mix baseline dynamics with treatment effects",
        fontsize=15,
        color=INK,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.918,
        "CausalTensor estimators use untreated and pre-treatment cells to reconstruct the missing counterfactual surface.",
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(STATIC_PATH, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def animation_frame_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(frame).convert("RGBA")


def palettize_frame(image: Image.Image) -> Image.Image:
    try:
        palette_mode = Image.Palette.ADAPTIVE
    except AttributeError:
        palette_mode = Image.ADAPTIVE
    return image.convert("P", palette=palette_mode)


def ease_in_out(progress: float) -> float:
    """Smooth each transition so it does not feel like a hard cut."""
    return progress * progress * (3 - 2 * progress)


def render_animation_frame(stage: int) -> Image.Image:
    baseline, treatment, effect, observed, time = build_example_panel()
    value_min = min(baseline.min(), observed.min())
    value_max = max(baseline.max(), observed.max())
    treatment_cmap = ListedColormap(["#E8EEF7", RED])

    titles = [
        "1. Start with the untreated outcome surface M",
        "2. Treatment assignment Z marks exposed cells",
        "3. The observed panel is O = M + tau Z",
        "4. Treated cells hide the untreated counterfactual",
        "5. Estimate M_hat, then read off the treatment gap",
    ]
    captions = [
        "Rows are units, columns are time periods.",
        "Some units adopt the intervention at different times.",
        "After treatment, outcomes include both baseline dynamics and the policy effect.",
        "The missing object is what each treated cell would have been without treatment.",
        "Panel estimators differ in how they reconstruct this counterfactual surface.",
    ]

    fig = plt.figure(figsize=(8.8, 4.9), dpi=122, facecolor=BG)
    grid = fig.add_gridspec(1, 2, left=0.075, right=0.965, bottom=0.14, top=0.80, wspace=0.28)
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_gap = fig.add_subplot(grid[0, 1])

    if stage == 0:
        draw_matrix(
            ax_matrix,
            baseline,
            "Latent baseline $M$",
            "viridis",
            vmin=value_min,
            vmax=value_max,
        )
    elif stage == 1:
        draw_matrix(
            ax_matrix,
            treatment,
            "Treatment mask $Z$",
            treatment_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
    elif stage == 2:
        draw_matrix(
            ax_matrix,
            observed,
            "Observed outcomes $O$",
            "viridis",
            vmin=value_min,
            vmax=value_max,
            treatment=treatment,
        )
    elif stage == 3:
        draw_matrix(
            ax_matrix,
            observed,
            "Counterfactual cells are hidden",
            "viridis",
            vmin=value_min,
            vmax=value_max,
            treatment=treatment,
        )
        ax_matrix.text(
            0.50,
            0.08,
            "$M$ is unobserved where $Z = 1$",
            transform=ax_matrix.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#111827", "ec": "none", "alpha": 0.78},
        )
    else:
        draw_matrix(
            ax_matrix,
            effect,
            "Estimated effect $\\hat{\\tau}$",
            "magma",
            vmin=0,
            vmax=max(effect.max(), 0.1),
            mask=treatment.astype(bool),
        )

    draw_gap_plot(ax_gap, baseline, treatment, observed, time, stage=stage, show_xlabel=False)

    fig.text(0.5, 0.925, titles[stage], ha="center", va="center", color=INK, fontsize=15, weight="bold")
    fig.text(0.5, 0.865, captions[stage], ha="center", va="center", color=MUTED, fontsize=10)
    fig.text(
        0.5,
        0.060,
        "Goal: separate baseline outcomes from treatment exposure to estimate causal effects.",
        ha="center",
        va="center",
        color=INK,
        fontsize=10,
    )

    image = animation_frame_to_image(fig)
    plt.close(fig)
    return image


def draw_animation_asset() -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []
    stage_frames = [render_animation_frame(stage) for stage in range(5)]

    hold_ms = 3800
    final_hold_ms = 4800
    transition_steps = 10
    transition_ms = 250

    frames.append(palettize_frame(stage_frames[0]))
    durations.append(hold_ms)

    for previous, current in zip(stage_frames, stage_frames[1:]):
        previous_rgba = previous.convert("RGBA")
        current_rgba = current.convert("RGBA")
        for step in range(1, transition_steps + 1):
            alpha = ease_in_out(step / (transition_steps + 1))
            blended = Image.blend(previous_rgba, current_rgba, alpha)
            frames.append(palettize_frame(blended))
            durations.append(transition_ms)
        frames.append(palettize_frame(current_rgba))
        durations.append(hold_ms)

    durations[-1] = final_hold_ms

    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_static_asset()
    draw_animation_asset()
    print(f"Wrote {STATIC_PATH}")
    print(f"Wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
