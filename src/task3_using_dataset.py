"""SIMC 2026 — Who are colluding? Task 3: K-means clustering of students.

Task 3 asks us to use K-means on the design matrix X (rows = students,
columns = questions) to group together students who answer similarly.
The visual deliverable is the similarity matrix S = X X^T with rows and
columns permuted so that students in the same cluster sit next to each
other. With a good K, this surfaces square red "blocks" along the
diagonal (within-cluster agreement) against a paler background
(between-cluster disagreement). With a bad K, the blocks are either
absent (K too small) or fragmented into one-student slivers (K too
large).

This module:
  1. Loads sample_small (50 students x 25 questions).
  2. Sweeps K over a range and runs scikit-learn's KMeans for each.
  3. For every K, renders the cluster-sorted similarity matrix and
     saves it to data/output/task3_similarity_K{K}.png. Cluster
     boundaries are drawn as black lines so block structure is obvious.
  4. Scores each K with the silhouette coefficient (cosine metric on
     X --- a natural choice because S_ij is, up to a constant, a scaled
     cosine similarity for fixed-norm +-1 vectors). The "best" K is the
     one that maximises silhouette *among K values that produce no
     singleton clusters*: the problem statement explicitly flags
     one-student groups as the signature of K being too large, so the
     raw silhouette winner is filtered through that constraint before
     being saved as data/output/similarity_best_task3.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "input" / "sample.npz"
OUTPUT_DIR = REPO_ROOT / "data" / "output"

# Fixed seed so the figures are reproducible; n_init=20 picks the best
# of 20 random initialisations, mitigating K-means' sensitivity to the
# starting centroids (the problem statement explicitly warns about this).
RANDOM_STATE = 0
N_INIT = 20

# Range of K values we sweep. The problem statement specifically calls
# out K = 3 (too small), 12 (about right) and 40 (too many); we include
# those plus a denser sweep so the silhouette curve is meaningful.
K_VALUES = [2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40]


def fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Fit KMeans with K=k and return integer cluster labels of shape (M,)."""
    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
    return model.fit_predict(X)


def sort_indices_by_label(labels: np.ndarray) -> np.ndarray:
    """Return the permutation that groups equal labels together.

    np.argsort with a stable kind keeps relative order within a label,
    so students that K-means lumped together appear as one contiguous
    block in the sorted similarity matrix.
    """
    return np.argsort(labels, kind="stable")


def cluster_boundaries(sorted_labels: np.ndarray) -> list[int]:
    """Indices (in the sorted ordering) where the cluster label changes.

    Used to draw block boundaries on the heatmap. We return positions
    *between* cells, i.e. the boundary between sorted positions p and
    p+1 has value p+0.5 when drawn.
    """
    changes = np.where(np.diff(sorted_labels) != 0)[0]
    return [int(c) + 1 for c in changes]


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score, or NaN if it is undefined for these labels.

    silhouette_score requires 2 <= n_labels <= n_samples - 1. With
    K=M (one student per cluster) the metric is undefined.
    """
    unique = np.unique(labels)
    if not (2 <= unique.size <= X.shape[0] - 1):
        return float("nan")
    # Cosine metric matches the geometry of S = X X^T: for rows of
    # constant norm sqrt(N), S_ij / N is exactly the cosine similarity.
    return float(silhouette_score(X, labels, metric="cosine"))


def plot_sorted_similarity(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    silhouette: float,
    save_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot S = X X^T with rows/cols permuted by cluster label.

    Cluster boundaries are overlaid as thin black lines. A side panel
    shows cluster sizes so the reader can see at a glance whether K is
    over-fragmenting the data (many size-1 clusters) or under-fitting
    (one giant cluster).
    """
    M, N = X.shape
    order = sort_indices_by_label(labels)
    sorted_labels = labels[order]
    S_sorted = (X @ X.T)[np.ix_(order, order)]
    boundaries = cluster_boundaries(sorted_labels)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 0.04, 0.55],
        figure=fig,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_sizes = fig.add_subplot(gs[0, 2])

    # ---- Heatmap of cluster-sorted similarity ---------------------------
    im = ax_heat.imshow(
        S_sorted, cmap="RdBu_r", vmin=-N, vmax=N,
        aspect="equal", origin="lower",
        extent=(-0.5, M - 0.5, -0.5, M - 0.5),
    )
    ax_heat.set_xlabel("student index $j$ (sorted by cluster)", fontsize=11)
    ax_heat.set_ylabel("student index $i$ (sorted by cluster)", fontsize=11)
    ax_heat.set_title(
        f"K = {k}{title_suffix}    silhouette = {silhouette:.3f}",
        fontsize=12, pad=8,
    )

    # Draw cluster boundaries as full-length crosshair lines so the
    # block structure (or lack thereof) is unmistakable.
    for b in boundaries:
        ax_heat.axhline(b - 0.5, color="black", linewidth=0.8, alpha=0.85)
        ax_heat.axvline(b - 0.5, color="black", linewidth=0.8, alpha=0.85)

    cbar = fig.colorbar(im, cax=ax_cbar, ticks=np.arange(-N, N + 1, 5))
    cbar.set_label("similarity $S_{ij}$", fontsize=10)

    # ---- Side panel: cluster sizes -------------------------------------
    cluster_ids, sizes = np.unique(labels, return_counts=True)
    order_sizes = np.argsort(-sizes)  # largest first for readability
    cluster_ids = cluster_ids[order_sizes]
    sizes = sizes[order_sizes]
    ax_sizes.barh(np.arange(len(sizes)), sizes,
                  color="#4a5568", edgecolor="white", linewidth=0.3)
    ax_sizes.set_yticks(np.arange(len(sizes)))
    ax_sizes.set_yticklabels([f"c{int(c)}" for c in cluster_ids], fontsize=8)
    ax_sizes.invert_yaxis()
    ax_sizes.set_xlabel("# students in cluster", fontsize=10)
    ax_sizes.set_title(
        f"{len(sizes)} clusters  |  singletons: {(sizes == 1).sum()}",
        fontsize=10, pad=8,
    )
    ax_sizes.set_xlim(0, max(sizes) + 1)
    ax_sizes.spines[["top", "right"]].set_visible(False)
    for idx, sz in enumerate(sizes):
        ax_sizes.text(sz + 0.1, idx, str(int(sz)),
                      va="center", fontsize=8)

    fig.suptitle(
        f"SIMC 2026 — Task 3: cluster-sorted similarity (K-means, K={k})",
        fontsize=13, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"K={k:>2}  silhouette={silhouette:+.3f}  ->  {save_path.name}")


def plot_silhouette_curve(
    ks: list[int], scores: list[float], best_k: int, save_path: Path
) -> None:
    """Plot silhouette vs. K to visualise the model-selection criterion."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(ks, scores, "-o", color="#2b6cb0")
    ax.axvline(best_k, color="#c53030", linestyle="--", linewidth=1,
               label=f"best K = {best_k}")
    ax.set_xlabel("number of clusters $K$", fontsize=11)
    ax.set_ylabel("silhouette score (cosine)", fontsize=11)
    ax.set_title(
        "SIMC 2026 — Task 3: silhouette score vs. K\n"
        "higher = clusters are tighter internally and farther apart",
        fontsize=12,
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved silhouette curve to {save_path.name}")


def main() -> None:
    loaded = np.load(INPUT_PATH)
    X = loaded["sample_small"]
    M, N = X.shape
    print(f"sample_small shape: {X.shape}")

    # Sweep K, score each, save a per-K similarity figure.
    scores: dict[int, float] = {}
    for k in K_VALUES:
        labels = fit_kmeans(X, k)
        score = _safe_silhouette(X, labels)
        scores[k] = score
        out = OUTPUT_DIR / f"task3_similarity_K{k:02d}.png"
        plot_sorted_similarity(X, labels, k, score, out)

    # Pick the K with the highest silhouette score. NaN scores (e.g.
    # K=M) are excluded, and so are K values that produce any singleton
    # cluster --- the problem statement calls those out as evidence
    # that K is too large.
    singleton_counts = {
        k: int((np.unique(fit_kmeans(X, k), return_counts=True)[1] == 1).sum())
        for k in K_VALUES
    }
    candidates = {
        k: s for k, s in scores.items()
        if np.isfinite(s) and singleton_counts[k] == 0
    }
    if not candidates:
        # Fall back to the raw silhouette winner if everything has a
        # singleton (shouldn't happen for this sample, but be safe).
        candidates = {k: s for k, s in scores.items() if np.isfinite(s)}
    best_k = max(candidates, key=candidates.get)
    print(
        f"\nsingleton counts per K: {singleton_counts}"
        f"\nbest K (silhouette, no singletons): {best_k} "
        f"(score = {candidates[best_k]:+.3f})"
    )

    # Re-fit the winning K to get fresh labels and save under the
    # canonical "best" filename requested by the task statement.
    best_labels = fit_kmeans(X, best_k)
    best_score = _safe_silhouette(X, best_labels)
    plot_sorted_similarity(
        X, best_labels, best_k, best_score,
        OUTPUT_DIR / "similarity_best_task3.png",
        title_suffix="  (BEST)",
    )

    # Bonus: the silhouette-vs-K curve makes the selection auditable.
    plot_silhouette_curve(
        list(scores.keys()), list(scores.values()), best_k,
        OUTPUT_DIR / "task3_silhouette_curve.png",
    )


if __name__ == "__main__":
    main()
