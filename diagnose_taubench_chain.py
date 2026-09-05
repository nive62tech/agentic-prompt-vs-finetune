import json

with open("results/taubench_chain_baseline_results.json") as f:
    results = json.load(f)

for r in results:
    print(r["id"], "|", r["grade"])
    print("  tool_calls:", r["tool_calls"])
    print("  final_text:", r["final_text"][:300])
    print()