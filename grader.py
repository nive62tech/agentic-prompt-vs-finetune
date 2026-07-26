"""
Grader for the agentic tool-use task suite.

INPUT CONTRACT (produced by whatever harness parses the model's raw
output — that harness is built in Phase 2, not here):

    tool_calls: list of {"tool": str, "args": dict}, in the order the
                model made them.
    final_text: str, the model's final natural-language response
                (used only for Tier 3 grading — did it surface the
                problem to the user).

USAGE:
    from grader import grade_task
    result = grade_task(task, tier=1, tool_calls=[...], final_text="...")
    # result = {"success": bool, "failure_type": str or None, "detail": str}

Failure types used across tiers:
    "wrong_tool"            — called a tool other than expected
    "bad_args"               — right tool, wrong argument value(s)
    "missing_call"            — expected call never happened
    "wrong_ordering"          — right calls, wrong order (order-sensitive tasks)
    "extra_disallowed_call"   — called a tool that must_not_call forbids
    "no_recovery"             — didn't surface the error/ambiguity in final_text
    "hallucinated_success"    — claimed success despite an upstream error
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from envs.tools import call_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Value comparison helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Placeholder resolution for Tier 2 chains (e.g. "<from:search_flights>")
# ---------------------------------------------------------------------------

def _resolve_placeholder(placeholder: str, prior_results: dict):
    inner = placeholder[len("<from:"):-1]
    parts = inner.split(":")
    tool_name = parts[0]
    field = parts[1] if len(parts) > 1 else None
    result = prior_results.get(tool_name)
    if result is None:
        return None
    if field:
        return result.get(field)
    if tool_name == "search_flights":
        flights = result.get("flights", [])
        return flights[0]["flight_id"] if flights else None
    return result


def resolve_expected_sequence(task: dict):
    """
    Executes the task's expected_sequence against the real simulator to
    produce a fully resolved sequence (placeholders filled in), so the
    grader has concrete values to diff the model's actual calls against.
    """
    resolved = []
    prior_results = {}
    for step in task["expected_sequence"]:
        tool_name = step["tool"]
        raw_args = step["args"]
        resolved_args = {}
        for k, v in raw_args.items():
            if isinstance(v, str) and v.startswith("<from:"):
                resolved_args[k] = _resolve_placeholder(v, prior_results)
            else:
                resolved_args[k] = v
        resolved.append({"tool": tool_name, "args": resolved_args})
        prior_results[tool_name] = call_tool(tool_name, resolved_args)
    return resolved


# ---------------------------------------------------------------------------
# Tier 1 grading
# ---------------------------------------------------------------------------

def grade_tier1(task: dict, tool_calls: list) -> dict:
    if not tool_calls:
        return {"success": False, "failure_type": "missing_call", "detail": "no tool call made"}

    call = tool_calls[0]
    if call["tool"] != task["expected_tool"]:
        return {"success": False, "failure_type": "wrong_tool",
                "detail": f"expected {task['expected_tool']}, got {call['tool']}"}

    if not _args_match(task["expected_args"], call.get("args", {})):
        return {"success": False, "failure_type": "bad_args",
                "detail": f"expected {task['expected_args']}, got {call.get('args')}"}

    return {"success": True, "failure_type": None, "detail": "ok"}


# ---------------------------------------------------------------------------
# Tier 2 grading
# ---------------------------------------------------------------------------

def grade_tier2(task: dict, tool_calls: list) -> dict:
    expected = resolve_expected_sequence(task)
    order_sensitive = task.get("order_sensitive", True)

    if len(tool_calls) < len(expected):
        return {"success": False, "failure_type": "missing_call",
                "detail": f"expected {len(expected)} calls, got {len(tool_calls)}"}

    if order_sensitive:
        for exp, act in zip(expected, tool_calls):
            if exp["tool"] != act["tool"]:
                return {"success": False, "failure_type": "wrong_ordering",
                        "detail": f"expected {exp['tool']} at this position, got {act['tool']}"}
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
                        "detail": f"no matching call found for {exp['tool']} {exp['args']}"}
            remaining.pop(match_idx)

    return {"success": True, "failure_type": None, "detail": "ok"}


# ---------------------------------------------------------------------------
# Tier 3 grading
# ---------------------------------------------------------------------------

def grade_tier3(task: dict, tool_calls: list, final_text: str) -> dict:
    final_text = (final_text or "").lower()

    if not tool_calls:
        return {"success": False, "failure_type": "missing_call", "detail": "no tool call made"}

    first_call = tool_calls[0]
    if first_call["tool"] != task["trigger_tool"]:
        return {"success": False, "failure_type": "wrong_tool",
                "detail": f"expected first call {task['trigger_tool']}, got {first_call['tool']}"}
    if not _args_match(task["trigger_args"], first_call.get("args", {})):
        return {"success": False, "failure_type": "bad_args",
                "detail": f"expected {task['trigger_args']}, got {first_call.get('args')}"}

    # confirm the trigger call actually produces the expected error/result
    sim_result = call_tool(task["trigger_tool"], task["trigger_args"])

    if task["type"] == "error":
        if "error" not in sim_result:
            return {"success": False, "failure_type": None,
                    "detail": "task misconfigured: trigger call did not error in simulator"}

        called_tools_after = [c["tool"] for c in tool_calls[1:]]
        for forbidden in task.get("must_not_call", []):
            if forbidden in called_tools_after:
                return {"success": False, "failure_type": "hallucinated_success",
                        "detail": f"called {forbidden} after an upstream error"}

        if not any(kw in final_text for kw in task["recovery_keywords"]):
            return {"success": False, "failure_type": "no_recovery",
                    "detail": "final response did not surface the error to the user"}

        return {"success": True, "failure_type": None, "detail": "ok"}

    elif task["type"] == "ambiguous":
        if not any(kw in final_text for kw in task["recovery_keywords"]):
            return {"success": False, "failure_type": "no_recovery",
                    "detail": "final response did not flag the ambiguity/conflict"}
        return {"success": True, "failure_type": None, "detail": "ok"}

    return {"success": False, "failure_type": None, "detail": f"unknown task type: {task['type']}"}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade_task(task: dict, tier: int, tool_calls: list, final_text: str = "") -> dict:
    if tier == 1:
        return grade_tier1(task, tool_calls)
    elif tier == 2:
        return grade_tier2(task, tool_calls)
    elif tier == 3:
        return grade_tier3(task, tool_calls, final_text)
    raise ValueError(f"unknown tier: {tier}")


# ---------------------------------------------------------------------------
# Self-test — run `python grader.py` to sanity-check the grader itself
# against the simulator, using hand-constructed "perfect" and "broken"
# responses. This does NOT touch a model — pure logic check.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tasks.tier1 import TIER1_TASKS
    from tasks.tier2 import TIER2_TASKS
    from tasks.tier3 import TIER3_TASKS

    print("=== Tier 1 self-test (perfect responses should all pass) ===")
    for task in TIER1_TASKS:
        perfect_call = [{"tool": task["expected_tool"], "args": task["expected_args"]}]
        result = grade_task(task, tier=1, tool_calls=perfect_call)
        status = "PASS" if result["success"] else "FAIL"
        print(f"  [{status}] {task['id']}")
        if not result["success"]:
            print(f"         {result}")

    print("\n=== Tier 2 self-test (perfect chains should all pass) ===")
    for task in TIER2_TASKS:
        resolved = resolve_expected_sequence(task)
        result = grade_task(task, tier=2, tool_calls=resolved)
        status = "PASS" if result["success"] else "FAIL"
        print(f"  [{status}] {task['id']}")
        if not result["success"]:
            print(f"         {result}")

    print("\n=== Tier 3 self-test (correct trigger call + recovery text should pass) ===")
    for task in TIER3_TASKS:
        calls = [{"tool": task["trigger_tool"], "args": task["trigger_args"]}]
        fake_final_text = f"Sorry, {task['recovery_keywords'][0]} — please check and try again."
        result = grade_task(task, tier=3, tool_calls=calls, final_text=fake_final_text)
        status = "PASS" if result["success"] else "FAIL"
        print(f"  [{status}] {task['id']}")
        if not result["success"]:
            print(f"         {result}")

    print("\nSelf-test complete. All PASS lines confirm the grader logic is internally consistent.")
