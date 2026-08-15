"""
DSPy program + optimization for Tier 3 (error recovery).
Reuses ChainProgram/GuardedChainProgram from dspy_optimize_tier2.py.
Phase 9: optimize() now accepts a seed parameter.
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
            dspy.Example(request=ex["prompt"], task_json=json.dumps(ex)).with_inputs("request")
        )
    return trainset


def optimize(lm, training_examples: list, max_new_tokens: int = 150, max_turns: int = 6,
             guarded: bool = True, seed: int = 9):
    """Phase 9: seed parameter added. guarded defaults to True here since
    the guard is now the "real" version of Tier 3's DSPy condition
    (un-guarded is kept only as the original diagnostic baseline)."""
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
        seed=seed,
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
        seed=seed,
    )
    return optimized_program


def evaluate_program(program, eval_tasks: list) -> list:
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
