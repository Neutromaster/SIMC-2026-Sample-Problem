"""SIMC 2026 — Who are colluding? Task 2: similarity matrix.

Task 2 asks for the pairwise similarity matrix

    S_ij = (X X^T)_ij = sum_{k=1..N} X_ik * X_jk

where X is the (M students x N questions) design matrix with entries in
{-1, +1}. Because each question contributes +1 when two students agree
(both right or both wrong) and -1 when they disagree, S_ij is bounded in
[-N, +N]. The diagonal entries S_ii are always +N (each student is
maximally similar to themselves), and large off-diagonal entries flag
pairs of students whose answer patterns are unusually aligned --- the
first place to look when hunting for collusion.

This script reproduces Figure 2 of the problem statement for the small
(50, 25) sample, with extra annotations:
  - diverging colormap centred at 0 so agreement/disagreement read
    symmetrically;
  - a histogram of off-diagonal similarities so the bulk distribution
    and outliers are visible side-by-side;
  - a textual call-out of the most-similar pair of distinct students.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "input" / "sample.npz"
OUTPUT_DIR = REPO_ROOT / "data" / "output"


def compute_similarity(X: np.ndarray) -> np.ndarray:
    """Return S = X X^T.

    The explicit double sum in the task statement is exactly the matrix
    product X @ X.T, so we delegate to NumPy's BLAS-backed matmul rather
    than writing a Python loop. For X with shape (M, N) the result has
    shape (M, M) and entries in the integer range [-N, +N].
    """
    return X @ X.T


def _most_similar_offdiagonal_pair(S: np.ndarray) -> tuple[int, int, int]:
    """Find (i, j, S_ij) for the largest similarity with i != j.

    We mask the diagonal (always +N, uninformative) and the lower
    triangle (S is symmetric, so each pair appears twice) before taking
    the argmax. Returned indices satisfy i < j.
    """
    M = S.shape[0]
    mask = np.triu(np.ones_like(S, dtype=bool), k=1)
    masked = np.where(mask, S, np.iinfo(np.int64).min)
    flat_idx = int(np.argmax(masked))
    i, j = divmod(flat_idx, M)
    return i, j, int(S[i, j])


def plot_similarity_matrix(
    X: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot S = X X^T as a heatmap with a marginal histogram of pair scores."""
    M, N = X.shape
    S = compute_similarity(X)

    # The heatmap colour scale is centred at 0 (no preferred agreement
    # direction) and extends symmetrically to +/- N, the theoretical max.
    vmax = N
    vmin = -N

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 0.04, 0.55],
        figure=fig,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])

    # ---- Heatmap of the similarity matrix -------------------------------
    # ``origin='lower'`` keeps student 0 at the bottom-left to match the
    # design-matrix figure from Task 1. RdBu_r maps high similarity to
    # red (collusion-suspect) and high dissimilarity to blue.
    im = ax_heat.imshow(
        S, cmap="RdBu_r", vmin=vmin, vmax=vmax,
        aspect="equal", origin="lower",
        extent=(-0.5, M - 0.5, -0.5, M - 0.5),
    )
    ax_heat.set_xlabel("student index $j$", fontsize=11)
    ax_heat.set_ylabel("student index $i$", fontsize=11)
    ax_heat.set_title(
        f"Similarity matrix $S_{{ij}} = (XX^T)_{{ij}}$ — {M} x {M}\n"
        f"each entry in $[-{N}, +{N}]$; diagonal $S_{{ii}} = {N}$ by construction",
        fontsize=12, pad=10,
    )

    # Highlight the most-similar off-diagonal pair with a marker so a
    # reader can find the candidate "colluders" at a glance.
    i_star, j_star, s_star = _most_similar_offdiagonal_pair(S)
    for (x, y) in [(j_star, i_star), (i_star, j_star)]:
        ax_heat.scatter([x], [y], s=160, facecolor="none",
                        edgecolor="black", linewidth=1.8, zorder=5)
    ax_heat.annotate(
        f"most-similar pair\n(i={i_star}, j={j_star}, $S$={s_star})",
        xy=(j_star, i_star), xycoords="data",
        xytext=(12, 12), textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6),
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    # ---- Discrete colorbar ---------------------------------------------
    # Ticks at every even integer keep the bar readable for N=25.
    ticks = np.arange(vmin, vmax + 1, 2)
    cbar = fig.colorbar(im, cax=ax_cbar, ticks=ticks)
    cbar.set_label("similarity score $S_{ij}$", fontsize=10)

    # ---- Marginal histogram of off-diagonal similarities ---------------
    # We exclude the diagonal because S_ii = N is a known constant and
    # would otherwise dominate the histogram. We also use the upper
    # triangle only to avoid double-counting symmetric pairs.
    upper = S[np.triu_indices(M, k=1)]
    bins = np.arange(vmin - 0.5, vmax + 1.5, 1)
    ax_hist.hist(upper, bins=bins, color="#4a5568",
                 edgecolor="white", linewidth=0.3)
    ax_hist.axvline(upper.mean(), color="#2b6cb0", linewidth=1.2,
                    label=f"mean = {upper.mean():.2f}")
    ax_hist.axvline(s_star, color="#c53030", linewidth=1.2, linestyle="--",
                    label=f"max off-diag = {s_star}")
    ax_hist.set_xlabel("off-diagonal similarity $S_{ij}$ ($i<j$)", fontsize=11)
    ax_hist.set_ylabel("number of student pairs", fontsize=11)
    ax_hist.set_title(
        f"distribution of {M*(M-1)//2} unique pair similarities",
        fontsize=11, pad=10,
    )
    ax_hist.legend(fontsize=9, frameon=False, loc="upper right")
    ax_hist.set_xlim(vmin, vmax)
    ax_hist.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "SIMC 2026 — Task 2: pairwise similarity for sample_small",
        fontsize=13, fontweight="bold",
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"saved figure to {save_path}")
    return fig


def main() -> None:
    loaded = np.load(INPUT_PATH)
    X_small = loaded["sample_small"]
    print(f"sample_small shape: {X_small.shape}")
    S = compute_similarity(X_small)
    print(f"similarity matrix shape: {S.shape} (dtype {S.dtype})")
    i, j, s = _most_similar_offdiagonal_pair(S)
    print(f"most-similar off-diagonal pair: students {i} & {j} with S = {s}")
    plot_similarity_matrix(
        X_small, save_path=OUTPUT_DIR / "task2_similarity_matrix.png"
    )
    plt.show()


if __name__ == "__main__":
    main()
