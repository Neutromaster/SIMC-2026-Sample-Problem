# SIMC 2024 Sample Challenge - Task 8

This repository answers Task 8 of the SIMC 2024 sample problem, "Who are colluding?", using the provided `sample.npz` data.

## Answer

The most likely colluding students are:

- Zero-based Python indices: `85, 1351, 1906, 3542, 5362`
- One-based student numbers: `86, 1352, 1907, 3543, 5363`

These five students have exactly the same response vector on all 50 questions. Their pairwise similarity score is therefore `S = 50` for every one of the `10` pairs within the group.

No other pair of students in the dataset has similarity `S >= 40`.

## Statistical Case

For each question `k`, estimate its probability of a correct answer from all 10,000 students:

```text
p_k = fraction of students with answer +1 on question k
```

For two non-colluding students, assuming independence, their answers match on question `k` with probability

```text
q_k = p_k^2 + (1 - p_k)^2
```

The similarity score is

```text
S = sum_k X_ik X_jk = 2K - 50
```

where `K` is the number of matching answers. The distribution of `K` is a Poisson-binomial distribution with probabilities `q_k`.

Using the estimated question difficulties in `sample_larger`:

```text
mean(S) = 2.556724
sd(S)   = 6.954882
P(S = 50 for one non-colluding pair) = 6.421319e-15
expected S = 50 pairs among all 49,995,000 student pairs = 3.210339e-07
```

So even one identical pair would be extremely surprising under the null model. Here we see a complete five-person clique of identical answer sheets, giving `10` identical pairs.

## Reproduce

Run:

```powershell
C:\Users\sanan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe task8_solution.py
```

Outputs are written to:

- `outputs/task8_summary.json`
- `outputs/task8_high_similarity_pairs.csv`

## Files

- `SIMC2024_SampleChallenge/sample.npz`: provided sample challenge data
- `task8_solution.py`: reproducible analysis
- `outputs/task8_summary.json`: machine-readable summary
- `outputs/task8_high_similarity_pairs.csv`: all pairs with similarity at least `40`
