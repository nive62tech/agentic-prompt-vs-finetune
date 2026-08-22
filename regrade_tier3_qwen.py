"""
Re-grades an already-saved Tier 3 results file using a widened recovery-
keyword list, without touching tasks/tier3.py or re-running the model.

Why: inspection showed Qwen's genuine recoveries sometimes use phrasing
the original (Llama-tuned) keyword lists don't cover — e.g. "no record of
a city" instead of "not found". This mirrors the same keyword-narrowness
issue caught earlier for Llama's Tier 3 results.

Usage:
    python regrade_tier3_qwen.py results/tier3_baseline_qwen_results.json
"""

import json
import sys
import copy

sys.path.insert(0, ".")
from tasks.tier3 import TIER3_TASKS
from grader import grade_task

# Additional generic recovery phrasings observed in Qwen's actual output,
# added on top of whatever tasks/tier3.py already has — not replacing it.
EXTRA_KEYWORDS = [
    "no record of",
    "not a common",
    "not a valid",
    "isn't directly provided",
    "is not directly",
    "could you please provide",
    "could you please specify",
    "does not exist",
    "not recognized",
]


def widen_task_keywords(task: dict) -> dict:
    widened = copy.deepcopy(task)
    if "recovery_keywords" in widened:
        existing = set(k.lower() for k in widened["recovery_keywords"])
        for kw in EXTRA_KEYWORDS:
            if kw not in existing:
                widened["recovery_keywords"].append(kw)
    return widened


def main(results_path: str):
    with open(results_path) as f:
        results = json.load(f)

    tasks_by_id = {t["id"]: widen_task_keywords(t) for t in TIER3_TASKS}

    changed = []
    new_pass_count = 0
    for r in results:
        task = tasks_by_id.get(r["id"])
        if task is None:
            print(f"WARNING: task {r['id']} not found in TIER3_TASKS, skipping")
            continue
        new_grade = grade_task(task, tier=3, tool_calls=r["tool_calls"], final_text=r["final_text"])
        old_success = r["grade"]["success"]
        new_success = new_grade["success"]
        if new_success:
            new_pass_count += 1
        if old_success != new_success:
            changed.append((r["id"], old_success, new_success))
        r["grade_original"] = r["grade"]
        r["grade"] = new_grade

    print(f"Original pass count: {sum(1 for r in results if r['grade_original']['success'])}/{len(results)}")
    print(f"Re-graded pass count (widened keywords): {new_pass_count}/{len(results)}")
    print()
    if changed:
        print("Tasks that flipped from FAIL to PASS (or vice versa) under widened keywords:")
        for task_id, old, new in changed:
            print(f"  {task_id}: {old} -> {new}")
    else:
        print("No tasks changed grade — widened keywords made no difference.")

    out_path = results_path.replace(".json", "_regraded.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved re-graded results to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python regrade_tier3_qwen.py <results_file.json>")
        sys.exit(1)
    main(sys.argv[1])
