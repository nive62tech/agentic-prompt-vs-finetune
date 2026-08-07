"""
DSPy program + optimization for Tier 3 (error recovery).

Reuses ChainProgram from dspy_optimize_tier2.py unchanged — it's already
a generic bounded multi-turn loop (predict action -> execute -> feed back
-> repeat until FINAL:), which works for Tier 3's error-recovery tasks
the same way it does for Tier 2's chains. Only the metric and trainset
builder are Tier 3-specific, since Tier 3 tasks use a different schema
(trigger_tool/trigger_args/must_not_call/recovery_keywords/type) than
Tier 2's expected_sequence.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from dspy_optimize_tier2 import ChainProgram, GuardedChainProgram
from grader import grade_task


def tier3_metric(example, prediction, trace=None) -> float:
    task = json.loads(example.task_json)
    grade = grade_task(task, tier=3, tool_calls=prediction.tool_calls, final_text=prediction.final_text)
    return 1.0 if grade["success"] else 0.0


def build_trainset(training_examples: list) -> list:
    trainset = []
    for ex in training_examples:
        trainset.append(
            dspy.Example(
                request=ex["prompt"],
                task_json=json.dumps(ex),
            ).with_inputs("request")
        )
    return trainset


def optimize(lm, training_examples: list, max_new_tokens: int = 150, max_turns: int = 6, guarded: bool = False):
    """Same lightweight, proven-working MIPROv2 settings as Tier 1/Tier 2.
    Set guarded=True to use GuardedChainProgram (repetition + malformed-JSON
    guards) instead of the plain ChainProgram used in the original Tier 3 run."""
    dspy.settings.configure(lm=lm)
    lm.max_new_tokens = max_new_tokens
    program_class = GuardedChainProgram if guarded else ChainProgram
    program = program_class(max_turns=max_turns)
    trainset = build_trainset(training_examples)

    optimizer = dspy.MIPROv2(
        metric=tier3_metric,
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
    """eval_tasks: TIER3_TASKS or TIER3_HELDOUT."""
    results = []
    for task in eval_tasks:
        prediction = program(request=task["prompt"])
        grade = grade_task(task, tier=3, tool_calls=prediction.tool_calls, final_text=prediction.final_text)
        results.append({
            "id": task["id"], "prompt": task["prompt"],
            "tool_calls": prediction.tool_calls, "final_text": prediction.final_text,
            "grade": grade,
        })
    return results
