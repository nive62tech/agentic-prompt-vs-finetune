"""
Aggregates multi-seed result files into mean/range, for reporting in the
paper instead of single-run point estimates.

Usage:
    python aggregate_seeds.py results/tier2_dspy_n10_seed42_results.json \
                               results/tier2_dspy_n10_seed123_results.json \
                               results/tier2_dspy_n10_seed7_results.json
"""

import json
import sys
import statistics


def success_rate(path: str) -> float:
    with open(path) as f:
        data = json.load(f)
    return sum(r["grade"]["success"] for r in data) / len(data)


def main(paths: list):
    rates = [success_rate(p) for p in paths]
    mean = statistics.mean(rates)
    lo, hi = min(rates), max(rates)
    print(f"Files: {len(paths)}")
    for p, r in zip(paths, rates):
        print(f"  {p}: {r:.1%}")
    print(f"\nMean: {mean:.1%}")
    print(f"Range: {lo:.1%} - {hi:.1%}")
    if len(rates) > 1:
        print(f"Std dev: {statistics.stdev(rates):.1%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_seeds.py <result_file1.json> <result_file2.json> ...")
        sys.exit(1)
    main(sys.argv[1:])
