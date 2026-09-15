"""
Phase 12 — Independent grader validation.

Randomly samples a percentage of already-graded results across ALL your
result files, shows you each one's raw prompt/tool_calls/final_text
WITHOUT revealing the grader's original verdict, asks you to judge it
yourself, then compares your judgment to the grader's and reports an
agreement rate.

This is entirely local, no GPU/Colab needed — it just re-reads JSON
files you already have in results/.

Usage:
    python phase12_validate_grader.py

Run this from your repo root (D:\\llm-paper\\agentic-prompt-vs-finetune).
It will look for all *_results.json files in results/, sample from them,
and walk you through blind re-grading interactively.
"""

import json
import glob
import random
import os

SAMPLE_FRACTION = 0.25  # sample ~25% of all graded results
SEED = 99  # fixed seed so the sample is reproducible if you want to re-check


def load_all_results():
    """Loads every *_results.json file in results/, tagging each entry
    with which file it came from."""
    all_entries = []
    pattern = os.path.join("results", "*_results.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No *_results.json files found in results/. "
              "Run this script from your repo root.")
        return []

    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict) or "grade" not in entry:
                continue
            all_entries.append({"source_file": os.path.basename(fpath), **entry})

    print(f"Loaded {len(all_entries)} graded entries from {len(files)} files.")
    return all_entries


def sample_entries(all_entries: list, fraction: float, seed: int) -> list:
    n = max(1, round(len(all_entries) * fraction))
    rng = random.Random(seed)
    return rng.sample(all_entries, min(n, len(all_entries)))


def blind_regrade(sample: list) -> list:
    """Walks through each sampled entry, showing everything except the
    original grade, and asks for a manual pass/fail judgment."""
    judgments = []
    print(f"\n{'='*70}")
    print(f"Blind re-grading {len(sample)} sampled entries.")
    print("For each one, read the prompt and the model's actual output,")
    print("then judge PASS or FAIL yourself, as if you were the grader.")
    print(f"{'='*70}\n")

    for i, entry in enumerate(sample, 1):
        print(f"\n--- Item {i}/{len(sample)} (from {entry['source_file']}) ---")
        print(f"ID: {entry.get('id')}")
        print(f"Prompt: {entry.get('prompt')}")
        print(f"Tool calls: {entry.get('tool_calls')}")
        final_text = entry.get("final_text", "")
        print(f"Final text: {final_text[:400]}")

        while True:
            answer = input("\nYour judgment — did this succeed? (p=pass / f=fail / s=skip): ").strip().lower()
            if answer in ("p", "f", "s"):
                break
            print("Please type 'p', 'f', or 's'.")

        judgments.append({
            "id": entry.get("id"),
            "source_file": entry["source_file"],
            "original_grade": entry["grade"],
            "manual_judgment": (
                None if answer == "s" else (answer == "p")
            ),
        })

    return judgments


def report(judgments: list):
    scored = [j for j in judgments if j["manual_judgment"] is not None]
    skipped = len(judgments) - len(scored)

    agree = 0
    disagreements = []
    for j in scored:
        original = j["original_grade"]["success"]
        manual = j["manual_judgment"]
        if original == manual:
            agree += 1
        else:
            disagreements.append(j)

    print(f"\n{'='*70}")
    print("PHASE 12 RESULTS")
    print(f"{'='*70}")
    print(f"Total sampled: {len(judgments)} (skipped: {skipped})")
    print(f"Scored: {len(scored)}")
    print(f"Agreement with original grader: {agree}/{len(scored)} "
          f"({agree/len(scored):.1%})" if scored else "No scored items.")

    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for d in disagreements:
            print(f"  {d['id']} ({d['source_file']}): "
                  f"grader said success={d['original_grade']['success']}, "
                  f"you said success={d['manual_judgment']}")
    else:
        print("\nNo disagreements — grader matched your manual judgment on every sampled item.")

    out_path = "results/phase12_grader_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "sample_fraction": SAMPLE_FRACTION,
            "seed": SEED,
            "total_sampled": len(judgments),
            "scored": len(scored),
            "agreement_count": agree,
            "agreement_rate": agree / len(scored) if scored else None,
            "disagreements": disagreements,
            "all_judgments": judgments,
        }, f, indent=2)
    print(f"\nSaved full report to {out_path}")
    print("Add this sentence (with your real numbers) to the paper's Method/Limitations:")
    if scored:
        print(f'  "An independent manual re-check of a random {SAMPLE_FRACTION:.0%} sample '
              f'of graded results ({len(scored)} items) agreed with the automatic grader '
              f'on {agree}/{len(scored)} cases ({agree/len(scored):.1%})."')


def main():
    all_entries = load_all_results()
    if not all_entries:
        return
    sample = sample_entries(all_entries, SAMPLE_FRACTION, SEED)
    judgments = blind_regrade(sample)
    report(judgments)


if __name__ == "__main__":
    main()
