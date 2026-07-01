"""Generate seaborn figures referenced by planning.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves PNGs, never shows a window —
# on macOS the default "macosx" backend can abort the process if it's launched without
# a normal GUI/WindowServer session (e.g. from an editor's agent or CI), so pin Agg
# before pyplot picks a backend.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

OUT_DIR = Path(__file__).parent.parent / "diagrams"
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)


def scoring_bands():
    """Horizontal band chart for ai_likeness_score thresholds."""
    fig, ax = plt.subplots(figsize=(10, 2.8))

    bands = [
        (0.0, 0.35, "#4C9BE8", "Likely Human-Written\n(score < 0.35, signals agree)"),
        (0.35, 0.65, "#F0B429", "Uncertain — middling evidence\n(0.35 ≤ score ≤ 0.65)"),
        (0.65, 1.0, "#E07A5F", "Likely AI-Generated\n(score > 0.65, signals agree)"),
    ]

    for start, end, color, label in bands:
        ax.barh(0, end - start, left=start, height=0.55, color=color, edgecolor="white", linewidth=1.5)
        mid = (start + end) / 2
        ax.text(mid, 0, label, ha="center", va="center", fontsize=9, color="#1a1a1a", wrap=True)

    for x, name in [(0.35, "0.35"), (0.65, "0.65")]:
        ax.axvline(x, color="#333", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(x, 0.42, name, ha="center", fontsize=8, color="#333")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.8)
    ax.set_xlabel("ai_likeness_score (0 = human-like style, 1 = AI-like style)")
    ax.set_yticks([])
    ax.set_title(
        "Score bands when signals agree and no guard fired\n"
        "(guards and high disagreement override these bands — see decision-space chart)",
        fontsize=11,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scoring_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def decision_space():
    """Heatmap: label region as function of mean score vs signal disagreement."""
    scores = np.linspace(0, 1, 101)
    disagreements = np.linspace(0, 1, 101)
    grid = np.zeros((len(disagreements), len(scores)))

    # 0=human, 1=middling uncertain, 2=AI, 3=conflict uncertain
    for i, d in enumerate(disagreements):
        for j, s in enumerate(scores):
            if d > 0.4:
                grid[i, j] = 3
            elif s < 0.35:
                grid[i, j] = 0
            elif s <= 0.65:
                grid[i, j] = 1
            else:
                grid[i, j] = 2

    cmap = sns.color_palette(["#4C9BE8", "#F0B429", "#E07A5F", "#9B8EC4"], as_cmap=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        grid,
        ax=ax,
        cmap=cmap,
        cbar=False,
        xticklabels=20,
        yticklabels=20,
        linewidths=0,
    )
    ax.set_xlabel("ai_likeness_score (average of both signals)")
    ax.set_ylabel("disagreement = |len_var_ai_score − lexical_ai_score|")
    ax.set_title("Label decision space (low_signal_confidence not shown)\n", fontsize=11)

    patches = [
        mpatches.Patch(color="#4C9BE8", label="Likely Human-Written"),
        mpatches.Patch(color="#F0B429", label="Uncertain — middling"),
        mpatches.Patch(color="#E07A5F", label="Likely AI-Generated"),
        mpatches.Patch(color="#9B8EC4", label="Uncertain — signals conflict (disagreement > 0.4)"),
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decision_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def signal_weights():
    """Stacked bar showing how lexical_ai_score sub-components combine."""
    fig, ax = plt.subplots(figsize=(7, 4))

    components = ["repetition_component\n(1 − MATTR/MATTR_REF)", "stock_phrase_component\n(phrase_rate/PHRASE_REF)"]
    weights = [0.6, 0.4]
    colors = ["#5B8C5A", "#D4A373"]

    bottom = 0
    for comp, w, c in zip(components, weights, colors):
        ax.bar("lexical_ai_score", w, bottom=bottom, color=c, edgecolor="white", width=0.45, label=f"{comp} × {w}")
        ax.text(0, bottom + w / 2, f"{w:.0%}", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        bottom += w

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weight toward lexical_ai_score")
    ax.set_title("Lexical signal: MATTR vs stock-phrase blend\n(MATTR weighted higher because phrase list is a formality detector)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "signal_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    scoring_bands()
    decision_space()
    signal_weights()
    print(f"Wrote diagrams to {OUT_DIR}/")
