from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


DATA_PATH = Path("SIMC2024_SampleChallenge") / "sample.npz"
OUTPUT_DIR = Path("outputs")
PAIR_THRESHOLD = 40


def poisson_binomial_match_distribution(match_probabilities: np.ndarray) -> np.ndarray:
    """Return P(K=k), where K is the number of matching answers across questions."""
    dist = np.array([1.0], dtype=float)
    for q in match_probabilities:
        next_dist = np.zeros(dist.size + 1, dtype=float)
        next_dist[:-1] += dist * (1.0 - q)
        next_dist[1:] += dist * q
        dist = next_dist
    return dist


def scan_high_similarity_pairs(X: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """Find all pairs with dot-product similarity at least threshold."""
    pairs: list[tuple[int, int, int]] = []
    X16 = X.astype(np.int16, copy=False)
    Xt = X16.T
    block = 500
    m = X.shape[0]

    for start in range(0, m, block):
        scores = X16[start : start + block] @ Xt
        for offset, row in enumerate(scores):
            i = start + offset
            js = np.flatnonzero(row[i + 1 :] >= threshold) + i + 1
            pairs.extend((i, int(j), int(row[j])) for j in js)

    return pairs


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    loaded = np.load(DATA_PATH)
    X = loaded["sample_larger"].astype(np.int8)
    m, n = X.shape

    p = (X == 1).mean(axis=0)
    expected_similarity = float(np.sum((2.0 * p - 1.0) ** 2))
    variance_similarity = float(np.sum(1.0 - (2.0 * p - 1.0) ** 4))
    sd_similarity = variance_similarity**0.5

    match_probabilities = p * p + (1.0 - p) * (1.0 - p)
    distribution = poisson_binomial_match_distribution(match_probabilities)

    def tail_probability_for_similarity(similarity: int) -> float:
        required_matches = int(np.ceil((similarity + n) / 2))
        return float(distribution[required_matches:].sum())

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
    duplicate_groups.sort(key=lambda group: (-group["size"], group["students_zero_based"]))

    high_similarity_pairs = scan_high_similarity_pairs(X, PAIR_THRESHOLD)
    pair_records = [
        {
            "student_a_zero_based": i,
            "student_b_zero_based": j,
            "student_a_one_based": i + 1,
            "student_b_one_based": j + 1,
            "similarity": similarity,
            "hamming_distance": int((n - similarity) // 2),
            "null_tail_probability_at_least_this_sim": tail_probability_for_similarity(
                similarity
            ),
        }
        for i, j, similarity in high_similarity_pairs
    ]

    with (OUTPUT_DIR / "task8_high_similarity_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=pair_records[0].keys())
        writer.writeheader()
        writer.writerows(pair_records)

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
            for sim in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
        },
        "expected_number_of_pairs_by_tail": {
            str(sim): tail_probability_for_similarity(sim) * m * (m - 1) / 2
            for sim in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
        },
        "high_similarity_pairs": pair_records,
    }

    with (OUTPUT_DIR / "task8_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    colluding = duplicate_groups[0]["students_zero_based"] if duplicate_groups else []
    print("Task 8 suspected colluding students")
    print("Zero-based indices:", colluding)
    print("One-based student numbers:", [i + 1 for i in colluding])
    print(f"Similarity null mean: {expected_similarity:.6f}")
    print(f"Similarity null standard deviation: {sd_similarity:.6f}")
    print(f"P(null pair has S=50): {tail_probability_for_similarity(50):.6e}")
    print(
        "Expected S=50 pairs among all pairs:",
        f"{tail_probability_for_similarity(50) * m * (m - 1) / 2:.6e}",
    )
    print(f"High-similarity pairs with S >= {PAIR_THRESHOLD}: {len(pair_records)}")
    for pair in pair_records:
        print(
            f"  {pair['student_a_zero_based']} and {pair['student_b_zero_based']}: "
            f"S={pair['similarity']}"
        )


if __name__ == "__main__":
    main()
