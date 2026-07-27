"""
DSPy program + optimization for Tier 1 (single tool call).

Scope note: this targets Tier 1 only. Extending to Tier 2/3 (multi-turn
chains) means wrapping this same per-step predictor inside the
run_agent-style loop instead of a single dspy.Predict call — a
reasonable next step once Tier 1 optimization is confirmed working,
not built here to keep this stage testable and shippable.

NOTE ON DSPY API: MIPROv2's exact constructor/compile kwargs have
shifted across DSPy versions. If `optimizer.compile(...)` below throws
a TypeError about unexpected kwargs, run `help(dspy.MIPROv2)` in the
same Colab cell and adjust the call to match what's installed — the
metric function and program structure are the parts that matter and
shouldn't need to change.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from envs.tools import TOOL_SCHEMAS
from envs.agent_harness import parse_tool_call
from grader import grade_task


class ToolCallSignature(dspy.Signature):
    """Given a user request and the list of available tools, output the
    single correct tool call as JSON: {"name": "<tool_name>", "parameters": {...}}.
    Only output the JSON object, nothing else."""

    request = dspy.InputField(desc="the user's natural language request")
    tools = dspy.InputField(desc="JSON list of available tools and their parameters")
    tool_call = dspy.OutputField(desc='JSON object: {"name": "<tool_name>", "parameters": {...}}')


class ToolCallProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ToolCallSignature)

    def forward(self, request):
        return self.predict(request=request, tools=json.dumps(TOOL_SCHEMAS))


def tier1_metric(example, prediction, trace=None) -> float:
    parsed = parse_tool_call(prediction.tool_call)
    tool_calls = [parsed] if parsed else []
    task = {"expected_tool": example.expected_tool, "expected_args": example.expected_args}
    grade = grade_task(task, tier=1, tool_calls=tool_calls)
    return 1.0 if grade["success"] else 0.0


def build_trainset(training_examples: list) -> list:
    trainset = []
    for ex in training_examples:
        trainset.append(
            dspy.Example(
                request=ex["prompt"],
                expected_tool=ex["expected_tool"],
                expected_args=ex["expected_args"],
            ).with_inputs("request")
        )
    return trainset


def optimize(lm, training_examples: list, max_new_tokens: int = 150):
    """Runs MIPROv2 prompt optimization against the given training examples.
    Returns the optimized DSPy program.

    Settings here are deliberately minimal (auto=None, num_candidates=1,
    num_trials=3, all proposer variants disabled) rather than the fuller
    "light" preset. The fuller preset repeatedly OOM'd a free-tier T4 with
    an 8B model in 4-bit — these settings are what actually completed
    successfully. Trade-off: shallower search than a full MIPROv2 run, but
    reliable completion matters more than search depth at this data scale.
    """
    dspy.settings.configure(lm=lm)
    lm.max_new_tokens = max_new_tokens
    program = ToolCallProgram()
    trainset = build_trainset(training_examples)

    optimizer = dspy.MIPROv2(
        metric=tier1_metric,
        auto=None,
        num_candidates=1,
        num_threads=1,
    )
    optimized_program = optimizer.compile(
        program, trainset=trainset,
        num_trials=3,
        max_bootstrapped_demos=1, max_labeled_demos=1,
        minibatch=False,
        requires_permission_to_run=False,
        program_aware_proposer=False,
        data_aware_proposer=False,
        tip_aware_proposer=False,
        fewshot_aware_proposer=False,
    )
    return optimized_program


def evaluate_program(program, eval_tasks: list) -> list:
    """Runs a (baseline or optimized) DSPy program against a list of Tier 1
    tasks (e.g. TIER1_TASKS or TIER1_HELDOUT) and grades each with grader.py.
    Returns a list of per-task result dicts, same shape as the baseline
    harness output, for direct comparison."""
    results = []
    for task in eval_tasks:
        prediction = program(request=task["prompt"])
        parsed = parse_tool_call(prediction.tool_call)
        tool_calls = [parsed] if parsed else []
        grade = grade_task(task, tier=1, tool_calls=tool_calls)
        results.append({
            "id": task["id"], "prompt": task["prompt"],
            "raw_output": prediction.tool_call,
            "tool_calls": tool_calls, "grade": grade,
        })
    return results
