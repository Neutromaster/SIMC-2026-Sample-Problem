"""SIMC 2026 — Who are colluding? Task 1: plot the design matrix.

The design matrix X has shape (M, N): rows are students, columns are
questions, entries are +1 (correct) or -1 (incorrect). For Task 1 we
visualise the small subset (M=50, N=25) as a heatmap with per-row and
per-column score summaries so the structure of right/wrong responses is
immediately legible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "input" / "sample.npz"
OUTPUT_DIR = REPO_ROOT / "data" / "output"


def load_design_matrices(path: Path) -> dict[str, np.ndarray]:
    """Load all design matrices from the provided ``.npz`` archive."""
    loaded = np.load(path)
    matrices = {key: loaded[key] for key in loaded.keys()}
    for key, value in matrices.items():
        print(f"key {key} has shape: {value.shape}")
    return matrices


def plot_design_matrix(X: np.ndarray, save_path: Path | None = None) -> plt.Figure:
    """Plot the (M students x N questions) design matrix with marginal summaries.

    The main panel is a binary heatmap (black = incorrect, white = correct).
    The right margin shows each student's total score (sum across questions);
    the bottom margin shows each question's total score (sum across students).
    Both summaries make it easy to spot unusually strong/weak students and
    unusually easy/hard questions at a glance.
    """
    M, N = X.shape

    # Two-tone colormap mirrors the +1/-1 encoding in the problem statement.
    binary_cmap = ListedColormap(["#111111", "#f5f5f5"])

    student_totals = X.sum(axis=1)   # row sums: per-student net score
    question_totals = X.sum(axis=0)  # col sums: per-question net score

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    # Main heatmap + right (student totals) + bottom (question totals) + colorbar.
    gs = GridSpec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 0.18, 0.04],
        height_ratios=[1.0, 0.18],
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_cbar = fig.add_subplot(gs[0, 2])

    # ---- Main heatmap ---------------------------------------------------
    # ``origin='lower'`` puts student index 0 at the bottom, matching the
    # mathematical convention used in the problem statement.
    im = ax_main.imshow(
        X, cmap=binary_cmap, vmin=-1, vmax=1,
        aspect="auto", origin="lower",
        extent=(-0.5, N - 0.5, -0.5, M - 0.5),
    )
    ax_main.set_xlabel("question index $j$", fontsize=11)
    ax_main.set_ylabel("student index $i$", fontsize=11)
    ax_main.set_title(
        f"Design matrix $X_{{ij}}$ — {M} students $\\times$ {N} questions\n"
        "black = incorrect ($-1$), white = correct ($+1$)",
        fontsize=12, pad=10,
    )
    # Light grid between cells makes individual responses easier to read.
    ax_main.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, M, 1), minor=True)
    ax_main.grid(which="minor", color="#888888", linewidth=0.25, alpha=0.4)
    ax_main.tick_params(which="minor", length=0)

    # ---- Right margin: per-student total score --------------------------
    bar_colors_students = [
        "#2b6cb0" if v >= 0 else "#c53030" for v in student_totals
    ]
    ax_right.barh(np.arange(M), student_totals, color=bar_colors_students,
                  edgecolor="black", linewidth=0.3)
    ax_right.axvline(0, color="black", linewidth=0.6)
    ax_right.set_xlabel(f"$\\sum_j X_{{ij}}$\n(net score, max ${N}$)", fontsize=9)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlim(-N, N)
    ax_right.spines[["top", "right"]].set_visible(False)

    # ---- Bottom margin: per-question total score ------------------------
    bar_colors_questions = [
        "#2b6cb0" if v >= 0 else "#c53030" for v in question_totals
    ]
    ax_bottom.bar(np.arange(N), question_totals, color=bar_colors_questions,
                  edgecolor="black", linewidth=0.3)
    ax_bottom.axhline(0, color="black", linewidth=0.6)
    ax_bottom.set_ylabel(f"$\\sum_i X_{{ij}}$\n(max ${M}$)", fontsize=9)
    ax_bottom.tick_params(axis="x", labelbottom=False)
    ax_bottom.set_ylim(-M, M)
    ax_bottom.spines[["top", "right"]].set_visible(False)

    # ---- Discrete colorbar for the heatmap ------------------------------
    cbar = fig.colorbar(im, cax=ax_cbar, ticks=[-1, 1])
    cbar.ax.set_yticklabels(["$-1$\nwrong", "$+1$\nright"])
    cbar.set_label("score $X_{ij}$", fontsize=10)

    # ---- Annotations: data shape + bar legend ---------------------------
    accuracy = (X == 1).mean()
    fig.suptitle(
        "SIMC 2026 — Task 1: design matrix for sample_small "
        f"(overall accuracy = {accuracy:.1%})",
        fontsize=13, fontweight="bold",
    )
    legend_handles = [
        Patch(facecolor="#2b6cb0", edgecolor="black",
              label="net positive (more right than wrong)"),
        Patch(facecolor="#c53030", edgecolor="black",
              label="net negative (more wrong than right)"),
    ]
    fig.legend(handles=legend_handles, loc="lower right",
               bbox_to_anchor=(0.98, 0.01), fontsize=9, frameon=False)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"saved figure to {save_path}")
    return fig


def main() -> None:
    matrices = load_design_matrices(INPUT_PATH)
    # Task 1 asks specifically for the small (50, 25) design matrix.
    X_small = matrices["sample_small"]
    plot_design_matrix(X_small, save_path=OUTPUT_DIR / "task1_design_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()


