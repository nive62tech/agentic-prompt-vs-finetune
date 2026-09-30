import json

with open("results/tier2_dspy_n10_results_strict.json") as f:
    results = json.load(f)

for r in results:
    print(r["id"])
    print("  original success:", r["grade_original"]["success"])
    print("  strict success:", r["grade"]["success"], "| failure_type:", r["grade"].get("failure_type"), "| detail:", r["grade"].get("detail"))
    print("  tool_calls:", r["tool_calls"])
    print("  final_text:", r["final_text"][:200])
    print()