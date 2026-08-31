"""
Grader for the tau-bench-inspired subset (Phase 11). Structurally
identical to grader.py's Tier 2 / Tier 3 logic, kept as a separate file
rather than modifying grader.py, since grader.py has a hardcoded
`from envs.tools import call_tool` import — duplicating the ~40 lines of
matching logic here is safer than risking a regression in the main
grader that all other phases depend on.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from envs.tools_taubench import call_tool  # noqa: E402


def _normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


def _values_match(expected, actual) -> bool:
    if expected is None or actual is None:
        return expected == actual
    return _normalize(expected) == _normalize(actual)


def _args_match(expected_args: dict, actual_args: dict) -> bool:
    if actual_args is None:
        return False
    for k, v in expected_args.items():
        if k not in actual_args:
            return False
        if not _values_match(v, actual_args[k]):
            return False
    return True


def grade_chain(task: dict, tool_calls: list) -> dict:
    expected = task["expected_sequence"]
    order_sensitive = task.get("order_sensitive", True)

    if len(tool_calls) < len(expected):
        return {"success": False, "failure_type": "missing_call",
                "detail": f"expected {len(expected)} calls, got {len(tool_calls)}"}

    if order_sensitive:
        for exp, act in zip(expected, tool_calls):
            if exp["tool"] != act["tool"]:
                return {"success": False, "failure_type": "wrong_ordering", "detail": "order mismatch"}
            if not _args_match(exp["args"], act.get("args", {})):
                return {"success": False, "failure_type": "bad_args", "detail": "args mismatch"}
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
                        "detail": f"no matching call found for {exp['tool']} {exp['args']}"}
            remaining.pop(match_idx)

    return {"success": True, "failure_type": None, "detail": "ok"}


def grade_error(task: dict, tool_calls: list, final_text: str) -> dict:
    final_text = (final_text or "").lower()

    if not tool_calls:
        return {"success": False, "failure_type": "missing_call", "detail": "no tool call made"}

    first_call = tool_calls[0]
    if first_call["tool"] != task["trigger_tool"]:
        return {"success": False, "failure_type": "wrong_tool",
                "detail": f"expected {task['trigger_tool']}, got {first_call['tool']}"}
    if not _args_match(task["trigger_args"], first_call.get("args", {})):
        return {"success": False, "failure_type": "bad_args", "detail": "trigger args mismatch"}

    sim_result = call_tool(task["trigger_tool"], task["trigger_args"])
    if "error" not in sim_result:
        return {"success": False, "failure_type": None, "detail": "task misconfigured: no error produced"}

    called_after = [c["tool"] for c in tool_calls[1:]]
    for forbidden in task.get("must_not_call", []):
        if forbidden in called_after:
            return {"success": False, "failure_type": "hallucinated_success",
                     "detail": f"called {forbidden} after an upstream error"}

    if not any(kw in final_text for kw in task["recovery_keywords"]):
        return {"success": False, "failure_type": "no_recovery",
                "detail": "final response did not surface the error"}

    return {"success": True, "failure_type": None, "detail": "ok"}


def grade_taubench_task(task: dict, tool_calls: list, final_text: str = "") -> dict:
    if "expected_sequence" in task:
        return grade_chain(task, tool_calls)
    return grade_error(task, tool_calls, final_text)


if __name__ == "__main__":
    from tasks.taubench_subset import (
        TAUBENCH_CHAIN_TASKS, TAUBENCH_CHAIN_HELDOUT,
        TAUBENCH_ERROR_TASKS, TAUBENCH_ERROR_HELDOUT,
    )

    print("=== Chain tasks self-test (perfect responses should all pass) ===")
    for task in TAUBENCH_CHAIN_TASKS + TAUBENCH_CHAIN_HELDOUT:
        perfect_calls = [{"tool": s["tool"], "args": s["args"]} for s in task["expected_sequence"]]
        result = grade_taubench_task(task, perfect_calls)
        status = "PASS" if result["success"] else "FAIL"
        print(f"  [{status}] {task['id']}")
        if not result["success"]:
            print(f"         {result}")

    print("\n=== Error tasks self-test (correct trigger + recovery text should pass) ===")
    for task in TAUBENCH_ERROR_TASKS + TAUBENCH_ERROR_HELDOUT:
        calls = [{"tool": task["trigger_tool"], "args": task["trigger_args"]}]
        fake_final = f"Sorry, {task['recovery_keywords'][0]} — please check and try again."
        result = grade_taubench_task(task, calls, fake_final)
        status = "PASS" if result["success"] else "FAIL"
        print(f"  [{status}] {task['id']}")
        if not result["success"]:
            print(f"         {result}")

    print("\nSelf-test complete.")
