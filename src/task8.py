"""SIMC 2026 -- Who are colluding? Task 8: per-column null model and pair scan on
``sample_larger`` (M=10,000 students, N=50 questions).

Why Task 8 is different from Tasks 1--7
=======================================

Tasks 5--7 derived the null distribution of the pairwise similarity
``S_{ij} = sum_k X_{ik} X_{jk}`` under a *homogeneous* difficulty model where
every question shares the same success probability ``p``. That assumption was
fine for ``sample_small`` (M=50, N=25) where the empirical column means are
narrow. For ``sample_larger`` (M=10,000, N=50) it is not: column-mean
accuracies ``p_j`` range from about 0.49 to 0.90 (a near-2x spread), so the
i.i.d. assumption of the Lindeberg--Levy CLT used in Task 7 is invalid.

The fix is the more general Lindeberg--Feller CLT, which only needs

    (i)  independence across questions,
    (iii) a uniform variance-smallness condition (the Lindeberg condition).

(ii) -- identical distribution -- is dropped. The per-question agreement
indicators ``Y_k = X_{ik} X_{jk}`` remain independent and bounded by 1, so
the Lindeberg condition is automatic, and the null moments generalise
term-by-term to

    <S> = sum_k (2*p_k - 1)**2
    Var(S) = sum_k [ 1 - (2*p_k - 1)**4 ]

Even better, the *exact* null distribution is also available in closed form
without Gaussian approximation: ``K = (S + N) / 2`` is the number of matching
answers across the N questions, and under independence

    K ~ PoissonBinomial( q_1, q_2, ..., q_N ),  q_k = p_k**2 + (1 - p_k)**2.

A Poisson binomial is the distribution of a sum of *non-identically*
distributed independent Bernoullis. Its PMF is computed exactly in O(N**2)
time by convolving Bernoulli(q_k) factors one at a time (see
``poisson_binomial_match_distribution``). For N=50 this is cheap, and gives
us *exact* tail probabilities under H_0 at any threshold -- no CLT needed.

What this script computes
=========================

1. **Per-question p_j estimate.** Column means of ``X = sample_larger``. We
   use the empirical mean as the H_0 success probability for question j; this
   is the maximum-likelihood estimate under the null and is consistent for
   M >> 1.

2. **Null moments.** ``mu_S`` and ``sigma_S`` via the Lindeberg--Feller
   sums above. Reported for cross-checking against the Task 7 Gaussian.

3. **Exact null PMF of S.** Through the change of variables S = 2K - N where
   K ~ PoissonBinomial. Used to compute one-sided tail probabilities
   ``P(S >= s)`` for a grid of thresholds.

4. **Expected number of pairs by tail.** Multiplied by the total number of
   distinct off-diagonal pairs ``M*(M-1)/2 ~ 5e7``. This converts each tail
   probability into the expected number of pairs at or above that threshold
   under H_0, which is the natural multiple-comparisons baseline for the
   pair scan.

5. **High-similarity pair scan.** All pairs with ``S_{ij} >= 40`` reported.
   The threshold 40 is conservative: under H_0 the expected number of such
   pairs is about 0.55 (see the table above), so finding *any* is already
   strong evidence; we find 10 pairs all sitting at the theoretical maximum
   S = 50.

6. **Duplicate-answer-vector groups.** Beyond pairwise comparison, we ask
   the structural question: are there *groups* of students with byte-for-byte
   identical answer rows? For ``sample_larger`` the answer is yes -- one
   5-clique of students (zero-based indices 85, 1351, 1906, 3542, 5362).
   Those C(5, 2) = 10 identical pairs are exactly the 10 pairs found by the
   similarity scan.

Outputs (under ``data/output/task8/``):

  - ``task8_summary.json``        full structured report
  - ``task8_high_similarity_pairs.csv``  every flagged pair, one row each

The console summary names the suspected colluding students and prints the
key null-distribution diagnostics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = Path("data") / "input" / "sample.npz"
OUTPUT_DIR = Path("data") / "output" / "task8"

# Similarity threshold at which a pair is flagged for the human-readable
# shortlist. Setting threshold=40 means: report every pair that agreed on at
# least (40 + 50) / 2 = 45 of 50 questions. The expected number of such pairs
# under H_0 is < 1 (see ``expected_number_of_pairs_by_tail`` in the JSON
# output), so this is a deliberately strict filter for the verdict-grade
# shortlist rather than a soft suspect list.
PAIR_THRESHOLD = 40


# =============================================================================
# Poisson-binomial PMF
# =============================================================================
def poisson_binomial_match_distribution(match_probabilities: np.ndarray) -> np.ndarray:
    """Return the exact PMF of K = number of matching answers across N questions.

    Under the null model of independent test-takers, each question k independently
    yields a match for the pair with probability

        q_k = p_k**2 + (1 - p_k)**2,                                       (1)

    where p_k is the success probability of question k. The match count
    ``K = sum_{k=1..N} 1{question k matches}`` is therefore a sum of
    *non-identically* distributed independent Bernoullis -- a Poisson binomial.

    We compute its PMF by convolving Bernoulli factors one at a time:

        dist_after_k_questions[i] = P(K_{1..k} = i)

    is updated from ``dist_after_(k-1)`` by

        new[i]   += old[i]   * (1 - q_k)      (question k missed)
        new[i+1] += old[i]   * q_k            (question k matched)

    so the support grows by one each iteration. After all N questions the
    returned array has length N+1 and represents P(K=0), P(K=1), ..., P(K=N).

    Complexity: O(N**2) time and O(N) memory. For N=50 this is microseconds.

    Parameters
    ----------
    match_probabilities : ndarray of shape (N,)
        The per-question match probabilities q_k from Eq. (1).

    Returns
    -------
    dist : ndarray of shape (N+1,)
        ``dist[k]`` = P(K = k). Sums to 1 to floating-point precision.
    """
    dist = np.array([1.0], dtype=float)
    for q in match_probabilities:
        next_dist = np.zeros(dist.size + 1, dtype=float)
        # "Question miss" branch: probability mass at K stays where it is,
        # weighted down by (1 - q).
        next_dist[:-1] += dist * (1.0 - q)
        # "Question match" branch: probability mass at K shifts up by one,
        # weighted by q.
        next_dist[1:] += dist * q
        dist = next_dist
    return dist


# =============================================================================
# Pair scan
# =============================================================================
def scan_high_similarity_pairs(X: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """Find all unordered pairs (i, j) with i < j and S_{ij} >= threshold.

    The similarity S_{ij} = sum_k X_{ik} X_{jk} is exactly the (i, j) entry of
    the Gram matrix ``X X^T``. For M = 10,000 students the full Gram matrix is
    100,000,000 entries (~ 400 MB at int32) -- big enough that we prefer to
    compute it in row-blocks rather than all at once. Each row block produces
    a (block, M) score matrix which we filter on the fly.

    For each row i in the current block we look at the upper-triangular slice
    ``scores[i, i+1 : ]`` (the lower triangle is the same information mirrored,
    so reporting both halves would double-count pairs).

    Notes on performance:

    - We cast X once to int16. With entries in {-1, +1} and N=50 the dot
      product fits in int16 (max +/- 50), which halves the bandwidth of the
      matrix multiply versus int32 / float64.
    - Block size 500 keeps the temporary score matrix to ~ 10 MB at int16,
      which fits comfortably in L2/L3 cache and amortises the BLAS call cost.

    Parameters
    ----------
    X : ndarray of shape (M, N), entries in {-1, +1}
        The design matrix.
    threshold : int
        Minimum similarity to report. Pairs with S_{ij} < threshold are
        silently dropped.

    Returns
    -------
    pairs : list of (i, j, S_{ij}) tuples with i < j, sorted by (i, j) by
        construction of the loop.
    """
    pairs: list[tuple[int, int, int]] = []
    X16 = X.astype(np.int16, copy=False)
    Xt = X16.T
    block = 500
    m = X.shape[0]

    for start in range(0, m, block):
        # scores[a, j] = X16[start + a] . X16[j] = S_{(start+a), j}
        scores = X16[start : start + block] @ Xt
        for offset, row in enumerate(scores):
            i = start + offset
            # Upper-triangular slice only: j > i. Without this we would
            # report both (i, j) and (j, i) and also the diagonal S_{ii} = N.
            js = np.flatnonzero(row[i + 1 :] >= threshold) + i + 1
            pairs.extend((i, int(j), int(row[j])) for j in js)

    return pairs


# =============================================================================
# Driver
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    # ``sample_larger`` is the (10_000, 50) design matrix with entries in
    # {-1, +1}: +1 = correct, -1 = wrong (see Task 1 in the report).
    loaded = np.load(DATA_PATH)
    X = loaded["sample_larger"].astype(np.int8)
    m, n = X.shape

    # -------------------------------------------------------------------------
    # 2. Per-question difficulty estimate p_k
    # -------------------------------------------------------------------------
    # p_k = empirical fraction of students who got question k correct. This is
    # the MLE of the per-question success probability under the null model and
    # is consistent for M >> 1 (we have M = 10,000, so the sampling error on
    # each p_k is ~ 1 / sqrt(M) ~ 1 percent). Per-question p_k is what makes
    # this Task 8 rather than a re-run of Task 7: we do *not* assume a common
    # p across questions.
    p = (X == 1).mean(axis=0)

    # -------------------------------------------------------------------------
    # 3. Closed-form null moments (Lindeberg--Feller, heterogeneous p_k)
    # -------------------------------------------------------------------------
    # Each Y_k = X_{ik} X_{jk} is bounded {-1, +1}, independent across k, with
    #     E[Y_k]    = (2 p_k - 1)**2
    #     Var(Y_k)  = 1 - (2 p_k - 1)**4
    # By linearity of expectation and independence,
    #     <S>     = sum_k (2 p_k - 1)**2
    #     Var(S)  = sum_k [ 1 - (2 p_k - 1)**4 ]
    # No assumption that p_k is constant across k is needed for these.
    expected_similarity = float(np.sum((2.0 * p - 1.0) ** 2))
    variance_similarity = float(np.sum(1.0 - (2.0 * p - 1.0) ** 4))
    sd_similarity = variance_similarity**0.5

    # -------------------------------------------------------------------------
    # 4. Exact null PMF of S via the Poisson-binomial match count
    # -------------------------------------------------------------------------
    # Per-question match probability q_k = P(Y_k = +1) = p_k**2 + (1 - p_k)**2.
    # The total match count K = (S + N) / 2 is then a Poisson binomial.
    match_probabilities = p * p + (1.0 - p) * (1.0 - p)
    distribution = poisson_binomial_match_distribution(match_probabilities)

    def tail_probability_for_similarity(similarity: int) -> float:
        """Exact one-sided null tail probability P(S >= similarity).

        Uses the change of variables S = 2K - N. For S >= s_obs we need
        K >= (s_obs + N) / 2. The ``ceil`` handles parity-mismatched
        thresholds gracefully: S only takes values with the same parity as N,
        so a threshold of, say, s=41 with N=50 gets bumped up to the next
        achievable value (K >= 46, i.e. S >= 42).
        """
        required_matches = int(np.ceil((similarity + n) / 2))
        return float(distribution[required_matches:].sum())

    # -------------------------------------------------------------------------
    # 5. Duplicate-answer-vector groups
    # -------------------------------------------------------------------------
    # Beyond pairwise scans, we look for *clusters* of students with byte-for-
    # byte identical rows. Such clusters cannot arise under H_0 unless the
    # cluster size is small *and* p_k is extremely peaked away from 0.5 on
    # every question -- neither holds here. A duplicate group of size k
    # contributes C(k, 2) = k(k-1)/2 perfectly-matching pairs to the Gram
    # matrix, so the scan in step 7 will find every pair in every duplicate
    # group as a side effect; this independent group-level view is the
    # cleaner statement of *which students* colluded.
    unique_rows, inverse, counts = np.unique(
        X, axis=0, return_inverse=True, return_counts=True
    )
    duplicate_groups = []
    for group_index in np.flatnonzero(counts > 1):
        members_zero_based = np.flatnonzero(inverse == group_index).astype(int)
        duplicate_groups.append(
            {
                "size": int(members_zero_based.size),
                "students_zero_based": members_zero_based.tolist(),
                "students_one_based": (members_zero_based + 1).tolist(),
                "answer_vector": unique_rows[group_index].astype(int).tolist(),
            }
        )
    # Largest groups first; ties broken by lexicographic member order so the
    # output is reproducible.
    duplicate_groups.sort(key=lambda group: (-group["size"], group["students_zero_based"]))

    # -------------------------------------------------------------------------
    # 6. High-similarity pair scan
    # -------------------------------------------------------------------------
    high_similarity_pairs = scan_high_similarity_pairs(X, PAIR_THRESHOLD)
    pair_records = [
        {
            "student_a_zero_based": i,
            "student_b_zero_based": j,
            "student_a_one_based": i + 1,
            "student_b_one_based": j + 1,
            "similarity": similarity,
            # S = N - 2 * (# disagreements)  =>  # disagreements = (N - S) / 2.
            # For S = N this is zero -- they agreed on every question.
            "hamming_distance": int((n - similarity) // 2),
            "null_tail_probability_at_least_this_sim": tail_probability_for_similarity(
                similarity
            ),
        }
        for i, j, similarity in high_similarity_pairs
    ]

    # -------------------------------------------------------------------------
    # 7. Persist the pair-level CSV
    # -------------------------------------------------------------------------
    # The CSV is the human-friendly counterpart to the JSON: one row per
    # flagged pair, columns mirror ``pair_records`` above. If no pairs were
    # flagged we still write a header so downstream consumers don't choke on
    # an empty file.
    pair_csv_columns = [
        "student_a_zero_based",
        "student_b_zero_based",
        "student_a_one_based",
        "student_b_one_based",
        "similarity",
        "hamming_distance",
        "null_tail_probability_at_least_this_sim",
    ]
    with (OUTPUT_DIR / "task8_high_similarity_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=pair_csv_columns)
        writer.writeheader()
        writer.writerows(pair_records)

    # -------------------------------------------------------------------------
    # 8. Structured JSON summary
    # -------------------------------------------------------------------------
    # The threshold grid {30, 32, ..., 50} steps by 2 because S has the same
    # parity as N = 50 (even), so odd thresholds are unreachable.
    # ``expected_number_of_pairs_by_tail`` multiplies the per-pair tail
    # probability by the total number of distinct off-diagonal pairs
    # M*(M-1)/2 = 49_995_000, which is the natural Bonferroni-style baseline:
    # how many pairs we should *expect* to see at or above each threshold
    # under H_0.
    threshold_grid = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    n_pairs_total = m * (m - 1) / 2
    summary = {
        "dataset": str(DATA_PATH),
        "students": m,
        "questions": n,
        "estimated_question_correct_probability": {
            "min": float(p.min()),
            "max": float(p.max()),
            "mean": float(p.mean()),
        },
        "null_similarity": {
            "expected_value": expected_similarity,
            "standard_deviation": sd_similarity,
        },
        "pair_threshold": PAIR_THRESHOLD,
        "number_of_high_similarity_pairs": len(pair_records),
        "duplicate_groups": duplicate_groups,
        "tail_probabilities": {
            str(sim): tail_probability_for_similarity(sim)
            for sim in threshold_grid
        },
        "expected_number_of_pairs_by_tail": {
            str(sim): tail_probability_for_similarity(sim) * n_pairs_total
            for sim in threshold_grid
        },
        "high_similarity_pairs": pair_records,
    }

    with (OUTPUT_DIR / "task8_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # 9. Console summary
    # -------------------------------------------------------------------------
    # The "suspected colluding students" line names the largest duplicate
    # group, which for ``sample_larger`` is the 5-clique of byte-identical
    # answer vectors. Each remaining diagnostic line corresponds to a row of
    # the structured JSON above.
    colluding = duplicate_groups[0]["students_zero_based"] if duplicate_groups else []
    print("Task 8 suspected colluding students")
    print("Zero-based indices:", colluding)
    print("One-based student numbers:", [i + 1 for i in colluding])
    print(f"Similarity null mean: {expected_similarity:.6f}")
    print(f"Similarity null standard deviation: {sd_similarity:.6f}")
    print(f"P(null pair has S=50): {tail_probability_for_similarity(50):.6e}")
    print(
        "Expected S=50 pairs among all pairs:",
        f"{tail_probability_for_similarity(50) * n_pairs_total:.6e}",
    )
    print(f"High-similarity pairs with S >= {PAIR_THRESHOLD}: {len(pair_records)}")
    for pair in pair_records:
        print(
            f"  {pair['student_a_zero_based']} and {pair['student_b_zero_based']}: "
            f"S={pair['similarity']}"
        )


if __name__ == "__main__":
    main()
