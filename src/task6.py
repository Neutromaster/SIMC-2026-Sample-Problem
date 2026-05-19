"""SIMC 2026 -- Who are colluding? Task 6: Monte Carlo validation of the
null-hypothesis moments of the similarity score S.

The closed-form result from Task 5 is

    <S>   = N (2p - 1)^2
    sigma = sqrt( N [ 1 - (2p - 1)^4 ] )

for two independent students who each answer N binary questions with
per-question success probability p (correct = +1, wrong = -1) and whose
similarity is S = sum_k X_ik X_jk. The point of Task 6 is to confirm those
two expressions numerically by simulating many independent (A, B) pairs of
students, computing S for each pair, and comparing the empirical first two
moments to the closed form.

This module does four things, each with its own figure:

1. Headline check (N=100, p=0.5, one million independent pairs). Prints the
   two closed-form moments and the two MC moments side by side, and verifies
   that they agree to within the expected sampling error.

2. Convergence plot. Using the same one million S samples, plot the absolute
   error in MC mean and MC std as a function of the cumulative sample count
   on log-log axes, with the expected 1/sqrt(M) reference line. This shows
   the simulation converges at the textbook rate.

3. Grid sweep over (N, p). Validate that the closed form is right *for the
   full plane* not just one (N, p) cell. Sweep N in {10, 25, 50, 100, 200}
   and p in {0.10, 0.30, 0.50, 0.70, 0.90}, run a smaller MC for each cell,
   write a CSV with the comparison, and save a per-N line plot of MC vs
   theory.

4. Distribution overlay. For the headline case, plot the empirical PMF of S
   against (a) the *exact* PMF (the distribution of S is exactly a shifted
   Binomial; see derivation below) and (b) the Gaussian approximation used
   by the central limit theorem in Task 7. Three curves on one axis make
   the CLT argument visible.

A bonus printed sanity check: for the canonical sample_small finding (the
pair of students 5 and 10 with S = 25 = N at empirical accuracy p ~= 0.558),
the script reports the H_0 z-score and two-sided p-value. That feeds into
Tasks 7 and 8.

Exact distribution of S under H_0. With each X_ik, X_jk independent +-1 and
P(X_ik = +1) = p,

    P(S_k = +1) = p^2 + (1 - p)^2 =: q
    P(S_k = -1) = 2 p (1 - p)   = 1 - q

so the number of agreements T = sum_k 1{S_k = +1} ~ Binomial(N, q) and
S = 2T - N. The closed-form moments derive from there, and the simulation
recovers them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "task6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility. All randomness flows through this seed.
SEED = 20260518


# =============================================================================
# Closed-form moments (Task 5 result)
# =============================================================================
def theory_moments(N: int, p: float) -> tuple[float, float]:
    """Return (mean, std) of S under the no-collusion null.

    These are exact for any N, p in (0, 1) and follow from S = 2T - N with
    T ~ Binomial(N, q), q = p^2 + (1 - p)^2.
    """
    r = 2.0 * p - 1.0
    mean = N * r**2
    var = N * (1.0 - r**4)
    return float(mean), float(np.sqrt(var))


def theory_pmf(N: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (support, probability) for the exact PMF of S under H_0.

    Support is the N+1 values S = 2t - N for t = 0, 1, ..., N. The
    probability at S = 2t - N is Binomial(N, q).pmf(t) where
    q = p^2 + (1 - p)^2.
    """
    q = p * p + (1.0 - p) * (1.0 - p)
    t = np.arange(N + 1)
    support = 2 * t - N
    probs = binom.pmf(t, N, q)
    return support, probs


# =============================================================================
# Monte Carlo sampling
# =============================================================================
def simulate_pair_similarities(
    N: int, p: float, n_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate n_pairs independent (A, B) student pairs and return their S.

    Implementation. We draw 2 * n_pairs students in a single random array of
    shape (2 * n_pairs, N), each entry being Bernoulli(p) mapped to +/-1.
    Pair i is row i paired with row n_pairs + i. The two halves are drawn
    from one call to rng.random, so the pairs are mutually independent and
    each pair's two students are independent of each other. S for each pair
    is computed by a single vectorised row-wise dot product.

    Returns
    -------
    S : ndarray of shape (n_pairs,) and dtype int.
    """
    # bits[k, j] = True iff student k got question j correct under H_0.
    bits = rng.random((2 * n_pairs, N)) < p
    # Map True -> +1, False -> -1 in int8 to keep memory small.
    X = np.where(bits, np.int8(1), np.int8(-1))
    A = X[:n_pairs]
    B = X[n_pairs:]
    # Row-wise dot product: S[i] = A[i] . B[i]. Cast to int to avoid int8
    # overflow when N is large.
    S = np.einsum("ij,ij->i", A.astype(np.int32), B.astype(np.int32))
    return S


# =============================================================================
# Figure 1 -- distribution overlay
# =============================================================================
def plot_distribution_overlay(
    S: np.ndarray, N: int, p: float, save_path: Path
) -> None:
    """Empirical PMF vs exact PMF vs Gaussian CLT, all on one axis.

    Three curves are drawn together so the reader can see at a glance
    that (a) the simulation matches the analytical distribution and (b) the
    CLT Gaussian -- which Task 7 will use to derive p-values -- is itself a
    good approximation for moderate N.
    """
    th_mean, th_std = theory_moments(N, p)
    support, exact_p = theory_pmf(N, p)

    # Empirical PMF. S only ever takes even (resp. odd) integer values
    # depending on parity of N; the support array above already encodes that.
    counts = np.zeros_like(exact_p)
    for s, c in zip(*np.unique(S, return_counts=True)):
        idx = (s - support[0]) // 2  # support steps by 2
        if 0 <= idx < len(counts):
            counts[idx] = c
    empirical_p = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)

    # Empirical PMF as filled bars.
    ax.bar(
        support, empirical_p,
        width=1.8, color="#9fb8d0", edgecolor="#4a7aa0", linewidth=0.4,
        label=f"Monte Carlo empirical ({len(S):,} pairs)",
    )
    # Exact PMF as discrete markers.
    ax.plot(
        support, exact_p, "o",
        color="#c0392b", markersize=4.5, zorder=3,
        label=r"exact PMF, $S=2T-N$,  $T\sim\mathrm{Bin}(N, q)$",
    )
    # CLT Gaussian as a smooth curve, *scaled to the bar width 2* so that
    # the continuous density is comparable to a discrete probability on a
    # support spaced by 2.
    s_grid = np.linspace(support.min() - 0.5, support.max() + 0.5, 600)
    gauss = norm.pdf(s_grid, loc=th_mean, scale=th_std) * 2.0
    ax.plot(
        s_grid, gauss, "-",
        color="#2d3748", linewidth=1.6,
        label=rf"CLT Gaussian  $\mathcal{{N}}({th_mean:.2f},\, {th_std:.2f}^2)$",
    )

    ax.axvline(th_mean, color="#2d3748", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel(r"similarity score $S$", fontsize=11)
    ax.set_ylabel(r"$\Pr(S = s)$", fontsize=11)
    ax.set_title(
        f"SIMC 2026 -- Task 6: distribution of $S$ under $H_0$  "
        f"($N={N}$, $p={p}$)\n"
        "empirical histogram vs exact shifted-binomial PMF vs CLT Gaussian",
        fontsize=12.5, fontweight="bold", pad=8,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved  {save_path.relative_to(REPO_ROOT)}")


# =============================================================================
# Figure 2 -- convergence
# =============================================================================
def plot_convergence(
    S: np.ndarray, N: int, p: float, save_path: Path
) -> None:
    """Log-log convergence of |MC mean - theory| and |MC std - theory| vs M.

    Both quantities should fall as 1/sqrt(M). We overlay the reference line
    sigma / sqrt(M) (for the mean) so the reader sees the rate matches.
    """
    th_mean, th_std = theory_moments(N, p)

    # Sample sizes log-spaced from 100 up to the full sample budget.
    Ms = np.unique(np.geomspace(100, len(S), num=40).astype(int))
    running_mean = np.empty_like(Ms, dtype=float)
    running_std = np.empty_like(Ms, dtype=float)
    cumsum = np.cumsum(S.astype(np.float64))
    cumsum2 = np.cumsum(S.astype(np.float64) ** 2)
    for idx, m in enumerate(Ms):
        mu = cumsum[m - 1] / m
        var = cumsum2[m - 1] / m - mu * mu
        running_mean[idx] = mu
        running_std[idx] = np.sqrt(max(var, 0.0))

    err_mean = np.abs(running_mean - th_mean)
    err_std = np.abs(running_std - th_std)
    reference = th_std / np.sqrt(Ms)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.loglog(Ms, err_mean, "-o", color="#2b6cb0", markersize=4,
              label=r"$|\hat\mu_{\mathrm{MC}} - \mu_{\mathrm{theory}}|$")
    ax.loglog(Ms, err_std, "-s", color="#c0392b", markersize=4,
              label=r"$|\hat\sigma_{\mathrm{MC}} - \sigma_{\mathrm{theory}}|$")
    ax.loglog(Ms, reference, "--", color="#2d3748", linewidth=1.2,
              label=r"$\sigma_{\mathrm{theory}} / \sqrt{M}$  (expected rate)")
    ax.set_xlabel(r"number of independent pairs $M$", fontsize=11)
    ax.set_ylabel("absolute error", fontsize=11)
    ax.set_title(
        f"SIMC 2026 -- Task 6: Monte Carlo convergence  ($N={N}$, $p={p}$)\n"
        r"errors fall like $1/\sqrt{M}$, confirming standard MC scaling",
        fontsize=12.5, fontweight="bold", pad=8,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved  {save_path.relative_to(REPO_ROOT)}")


# =============================================================================
# Figure 3 + CSV -- grid validation across (N, p)
# =============================================================================
@dataclass
class GridResult:
    N: int
    p: float
    mc_mean: float
    th_mean: float
    mc_std: float
    th_std: float

    @property
    def err_mean(self) -> float:
        return self.mc_mean - self.th_mean

    @property
    def err_std(self) -> float:
        return self.mc_std - self.th_std


def run_grid(
    Ns: list[int], ps: list[float], n_pairs: int, rng: np.random.Generator
) -> list[GridResult]:
    """Run a smaller MC for each (N, p) and return the comparison table."""
    out: list[GridResult] = []
    for N in Ns:
        for p in ps:
            S = simulate_pair_similarities(N, p, n_pairs, rng)
            th_mean, th_std = theory_moments(N, p)
            out.append(GridResult(
                N=N, p=p,
                mc_mean=float(S.mean()),
                th_mean=th_mean,
                mc_std=float(S.std(ddof=1)),
                th_std=th_std,
            ))
    return out


def save_grid_csv(results: list[GridResult], save_path: Path) -> None:
    with save_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "N", "p",
            "theory_mean", "mc_mean", "abs_err_mean",
            "theory_std", "mc_std", "abs_err_std",
            "rel_err_mean_in_std",
        ])
        for r in results:
            rel = abs(r.err_mean) / max(r.th_std, 1e-12)
            w.writerow([
                r.N, f"{r.p:.2f}",
                f"{r.th_mean:.6f}", f"{r.mc_mean:.6f}", f"{r.err_mean:+.6f}",
                f"{r.th_std:.6f}", f"{r.mc_std:.6f}", f"{r.err_std:+.6f}",
                f"{rel:.2e}",
            ])
    print(f"saved  {save_path.relative_to(REPO_ROOT)}")


def plot_grid_validation(
    results: list[GridResult], Ns: list[int], ps: list[float], save_path: Path
) -> None:
    """Two side-by-side panels: mean vs N (one line per p) and std vs N.

    Theory is a solid line, MC points are markers. Visual agreement across
    the full (N, p) plane is the goal.
    """
    fig, (ax_mu, ax_sd) = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(ps):
        color = cmap(i / max(len(ps) - 1, 1))
        cells = [r for r in results if r.p == p]
        cells_by_N = {r.N: r for r in cells}
        N_arr = np.array(Ns, dtype=float)
        th_mu = N_arr * (2 * p - 1) ** 2
        th_sd = np.sqrt(N_arr * (1 - (2 * p - 1) ** 4))
        mc_mu = np.array([cells_by_N[N].mc_mean for N in Ns])
        mc_sd = np.array([cells_by_N[N].mc_std for N in Ns])

        ax_mu.plot(Ns, th_mu, "-", color=color, linewidth=1.5, alpha=0.85,
                   label=f"theory, $p={p}$")
        ax_mu.plot(Ns, mc_mu, "o", color=color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.7)
        ax_sd.plot(Ns, th_sd, "-", color=color, linewidth=1.5, alpha=0.85,
                   label=f"theory, $p={p}$")
        ax_sd.plot(Ns, mc_sd, "o", color=color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.7)

    for ax, label, title in [
        (ax_mu, r"$\langle S \rangle$", r"mean similarity:  $N(2p-1)^2$"),
        (ax_sd, r"$\sigma_N$", r"standard deviation:  $\sqrt{N\,[1-(2p-1)^4]}$"),
    ]:
        ax.set_xlabel(r"number of questions $N$", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(title, fontsize=12, pad=6)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")

    fig.suptitle(
        "SIMC 2026 -- Task 6: Monte Carlo vs closed form across "
        "the $(N, p)$ plane\n"
        "lines = closed-form Task 5 result, markers = Monte Carlo estimates",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved  {save_path.relative_to(REPO_ROOT)}")


# =============================================================================
# Console summary
# =============================================================================
def print_headline(N: int, p: float, S: np.ndarray) -> None:
    th_mean, th_std = theory_moments(N, p)
    mc_mean = float(S.mean())
    mc_std = float(S.std(ddof=1))
    M = len(S)
    se_mean = th_std / np.sqrt(M)
    print()
    print(f"=== Headline Monte Carlo  (N={N}, p={p}, M={M:,} pairs) ===")
    print(f"  theory  mean = {th_mean:+.6f}     "
          f"std = {th_std:.6f}")
    print(f"  MC      mean = {mc_mean:+.6f}     "
          f"std = {mc_std:.6f}")
    print(f"  abs err mean = {abs(mc_mean - th_mean):.2e}   "
          f"(expected ~= {se_mean:.2e}, "
          f"i.e. {abs(mc_mean - th_mean)/se_mean:.2f} SE)")
    print(f"  abs err std  = {abs(mc_std - th_std):.2e}")


def print_grid(results: list[GridResult]) -> None:
    print()
    print("=== Grid sweep summary ===")
    print(f"  {'N':>4}  {'p':>5}  "
          f"{'th_mean':>9}  {'mc_mean':>9}  {'th_std':>8}  {'mc_std':>8}  "
          f"{'|err_mu|':>9}  {'|err_sd|':>9}")
    for r in results:
        print(f"  {r.N:>4d}  {r.p:>5.2f}  "
              f"{r.th_mean:>9.4f}  {r.mc_mean:>9.4f}  "
              f"{r.th_std:>8.4f}  {r.mc_std:>8.4f}  "
              f"{abs(r.err_mean):>9.4f}  {abs(r.err_std):>9.4f}")


def print_canonical_sanity(N: int, p_emp: float, S_obs: int) -> None:
    """Where does S = 25 land for the (5, 10) pair under H_0?

    Uses the EXACT shifted-binomial tail probability, not the CLT
    approximation; with N as small as 25 the CLT is OK but the exact tail
    is cheap to compute and tighter.
    """
    th_mean, th_std = theory_moments(N, p_emp)
    z = (S_obs - th_mean) / th_std
    # Exact one-sided tail Pr(S >= S_obs) = Pr(T >= (S_obs + N)/2) under
    # T ~ Binomial(N, q).
    q = p_emp * p_emp + (1.0 - p_emp) * (1.0 - p_emp)
    t_obs = (S_obs + N) // 2
    p_upper_exact = float(binom.sf(t_obs - 1, N, q))  # P(T >= t_obs)
    p_upper_clt = float(norm.sf((S_obs - 0.5 - th_mean) / th_std))  # continuity correction
    print()
    print(f"=== Canonical sanity check on the (student 5, student 10) pair ===")
    print(f"  sample_small empirical accuracy  p = {p_emp:.4f}")
    print(f"  N = {N}, S_observed = {S_obs}")
    print(f"  H_0 mean              = {th_mean:.4f}")
    print(f"  H_0 std               = {th_std:.4f}")
    print(f"  z-score under H_0     = {z:.3f}")
    print(f"  exact one-sided p     = {p_upper_exact:.3e}    "
          f"({1.0/p_upper_exact:,.0f}:1 against, if non-zero)")
    print(f"  CLT one-sided p       = {p_upper_clt:.3e}      "
          "(continuity-corrected Gaussian; this is what Task 7 will use)")


# =============================================================================
# Driver
# =============================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- 1. Headline: N=100, p=0.5, M=10^6 ---------------------------------
    N0, p0, M0 = 100, 0.5, 1_000_000
    S0 = simulate_pair_similarities(N0, p0, M0, rng)
    print_headline(N0, p0, S0)

    plot_distribution_overlay(
        S0, N0, p0, OUTPUT_DIR / "task6_distribution_N100_p05.png",
    )
    plot_convergence(
        S0, N0, p0, OUTPUT_DIR / "task6_convergence_N100_p05.png",
    )

    # --- 2. Grid sweep ------------------------------------------------------
    Ns = [10, 25, 50, 100, 200]
    ps = [0.10, 0.30, 0.50, 0.70, 0.90]
    M_grid = 200_000
    grid = run_grid(Ns, ps, M_grid, rng)
    print_grid(grid)
    save_grid_csv(grid, OUTPUT_DIR / "task6_grid_results.csv")
    plot_grid_validation(
        grid, Ns, ps, OUTPUT_DIR / "task6_grid_validation.png",
    )

    # --- 3. Asymmetric stress case: small p, large N -----------------------
    # Picks up any bug that only shows when (2p-1)^4 is far from zero.
    print()
    print("=== Stress cell  (N=200, p=0.20, M=300,000 pairs) ===")
    S_stress = simulate_pair_similarities(200, 0.20, 300_000, rng)
    th_mean, th_std = theory_moments(200, 0.20)
    print(f"  theory  mean = {th_mean:+.4f}     std = {th_std:.4f}")
    print(f"  MC      mean = {S_stress.mean():+.4f}     "
          f"std = {S_stress.std(ddof=1):.4f}")

    # --- 4. Canonical sanity check ----------------------------------------
    # The (5, 10) pair in sample_small had S = 25 = N. The empirical
    # success rate across all 50 students and 25 questions was about 55.8%.
    # We use that as a (rough) p estimate to compute an H_0 z-score that
    # foreshadows the per-column p_j machinery of Task 8.
    print_canonical_sanity(N=25, p_emp=0.558, S_obs=25)

    print()
    print(f"all figures saved under  {OUTPUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
