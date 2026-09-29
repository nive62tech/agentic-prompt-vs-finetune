"""
Stricter Tier 2 re-grader (Phase 12 follow-up).

Phase 12's independent validation found that our original Tier 2 grader
checks only whether the REQUIRED tool calls are PRESENT somewhere in the
model's output, not whether the model then stopped cleanly. This script
re-grades every saved Tier 2 result file under a stricter standard that
matches how a human reader judged the same outputs:

  1. EXACT call count: the model must make exactly the number of tool
     calls the task requires — no extra, irrelevant, or repeated calls
     appended afterward.
  2. A REAL final answer: final_text must not be empty and must not be
     the "(max turns reached without a final answer)" placeholder.
  3. All required calls still matched (ordered or unordered, as the task
     specifies), with the same argument-matching rules as before.

Runs entirely locally on files you already have — no GPU, no Colab.

Usage:
    python regrade_tier2_strict.py

Run from your repo root. Saves corrected files as
results/<original_name>_strict.json and prints a before/after summary
for every file.
"""

import glob
import json
import os
import sys

sys.path.insert(0, ".")
from tasks.tier2 import TIER2_TASKS, TIER2_HELDOUT
from grader import resolve_expected_sequence

NO_ANSWER_MARKERS = [
    "(max turns reached without a final answer)",
    "max turns reached",
]


def _normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


def _args_match(expected_args: dict, actual_args: dict) -> bool:
    if actual_args is None:
        return False
    for k, v in expected_args.items():
        if k not in actual_args:
            return False
        exp, act = _normalize(v), _normalize(actual_args[k])
        if exp != act:
            return False
    return True


def has_real_final_answer(final_text: str) -> bool:
    if not final_text or not final_text.strip():
        return False
    lowered = final_text.strip().lower()
    return not any(marker in lowered for marker in NO_ANSWER_MARKERS)


def grade_tier2_strict(task: dict, tool_calls: list, final_text: str) -> dict:
    expected = resolve_expected_sequence(task)
    order_sensitive = task.get("order_sensitive", True)

    # Check 1: exact call count — no extra calls after the required ones.
    if len(tool_calls) != len(expected):
        return {
            "success": False, "failure_type": "wrong_call_count",
            "detail": f"expected exactly {len(expected)} calls, got {len(tool_calls)}",
        }

    # Check 2: all required calls matched.
    if order_sensitive:
        for exp, act in zip(expected, tool_calls):
            if exp["tool"] != act["tool"]:
                return {"success": False, "failure_type": "wrong_ordering",
                        "detail": f"expected {exp['tool']}, got {act['tool']}"}
            if not _args_match(exp["args"], act.get("args", {})):
                return {"success": False, "failure_type": "bad_args",
                        "detail": f"expected {exp['args']}, got {act.get('args')}"}
    else:
        remaining = list(tool_calls)
        for exp in expected:
            match_idx = None
            for i, act in enumerate(remaining):
                if act["tool"] == exp["tool"] and _args_match(exp["args"], act.get("args", {})):
                    match_idx = i
                    break
            if match_idx is None:
                return {"success": False, "failure_type": "missing_call",
                        "detail": f"no match for {exp['tool']} {exp['args']}"}
            remaining.pop(match_idx)

    # Check 3: a real final answer must exist.
    if not has_real_final_answer(final_text):
        return {"success": False, "failure_type": "no_final_answer",
                "detail": "no coherent final answer produced"}

    return {"success": True, "failure_type": None, "detail": "ok (strict)"}


def main():
    tasks_by_id = {t["id"]: t for t in TIER2_TASKS + TIER2_HELDOUT}

    files = sorted(glob.glob(os.path.join("results", "tier2_*_results.json")))
    files = [f for f in files if not f.endswith("_strict.json")]
    if not files:
        print("No tier2_*_results.json files found in results/. Run from repo root.")
        return

    print(f"{'file':<60} {'original':>10} {'strict':>10}")
    print("-" * 82)

    summary = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            results = json.load(f)

        orig_pass = strict_pass = 0
        regraded = []
        for r in results:
            task = tasks_by_id.get(r["id"])
            if task is None:
                regraded.append(r)
                continue
            new_grade = grade_tier2_strict(task, r["tool_calls"], r.get("final_text", ""))
            orig_pass += int(r["grade"]["success"])
            strict_pass += int(new_grade["success"])
            regraded.append({**r, "grade_original": r["grade"], "grade": new_grade})

        n = len(results)
        name = os.path.basename(fpath)
        print(f"{name:<60} {orig_pass:>4}/{n:<5} {strict_pass:>4}/{n:<5}")
        summary.append((name, orig_pass, strict_pass, n))

        out_path = fpath.replace(".json", "_strict.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(regraded, f, indent=2)

    print("\nSaved corrected files as results/<name>_strict.json")
    print("Paste this table back so the paper's Tier 2 numbers can be updated.")


if __name__ == "__main__":
    main()
