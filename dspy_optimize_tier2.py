"""
DSPy program + optimization for Tier 2 (multi-tool chains).

Phase 9 update: optimize() now accepts a seed parameter, passed to
MIPROv2's constructor and usable to vary training-data sampling, so
multiple independent runs can be compared instead of relying on a single
seed's result.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from envs.tools import TOOL_SCHEMAS, call_tool
from envs.agent_harness import parse_tool_call
from grader import grade_task


class ChainStepSignature(dspy.Signature):
    """Given a user request, the available tools, and the history of tool
    calls made so far (with their results), output the SINGLE next action.
    If another tool call is needed, output ONLY the JSON:
    {"name": "<tool_name>", "parameters": {...}}.
    If all necessary tool calls have already been made and you can now
    answer the user, output "FINAL: <your answer>" instead.
    Never output both a tool call and FINAL in the same turn."""

    request = dspy.InputField(desc="the user's natural language request")
    tools = dspy.InputField(desc="JSON list of available tools and their parameters")
    history = dspy.InputField(desc="JSON list of {tool, args, result} for steps already taken; empty list if none yet")
    action = dspy.OutputField(desc='Either a tool call JSON, or "FINAL: <answer>"')


class ChainProgram(dspy.Module):
    def __init__(self, max_turns: int = 6):
        super().__init__()
        self.predict = dspy.Predict(ChainStepSignature)
        self.max_turns = max_turns

    def forward(self, request):
        history = []
        tool_calls = []
        final_text = ""

        for _ in range(self.max_turns):
            result = self.predict(
                request=request,
                tools=json.dumps(TOOL_SCHEMAS),
                history=json.dumps(history),
            )
            action_text = result.action.strip()

            if action_text.upper().startswith("FINAL:"):
                final_text = action_text.split(":", 1)[1].strip()
                break

            call = parse_tool_call(action_text)
            if call is None:
                final_text = action_text
                break

            tool_calls.append(call)
            tool_result = call_tool(call["tool"], call["args"])
            history.append({"tool": call["tool"], "args": call["args"], "result": tool_result})
        else:
            final_text = "(max turns reached without a final answer)"

        return dspy.Prediction(tool_calls=tool_calls, final_text=final_text)


class GuardedChainProgram(dspy.Module):
    """Same loop as ChainProgram, plus:
    1. Repetition guard — stops instead of re-issuing an identical (tool,
       args) call already tried, and reports the earlier failure.
    2. Malformed-JSON retry guard — one corrective nudge before giving up
       on an unparseable turn.
    See Section 3.5 of the paper for full rationale."""

    def __init__(self, max_turns: int = 6):
        super().__init__()
        self.predict = dspy.Predict(ChainStepSignature)
        self.max_turns = max_turns

    def forward(self, request):
        history = []
        tool_calls = []
        final_text = ""
        seen_calls = set()
        malformed_retry_used = False

        for _ in range(self.max_turns):
            result = self.predict(
                request=request,
                tools=json.dumps(TOOL_SCHEMAS),
                history=json.dumps(history),
            )
            action_text = result.action.strip()

            if action_text.upper().startswith("FINAL:"):
                final_text = action_text.split(":", 1)[1].strip()
                break

            call = parse_tool_call(action_text)

            if call is None:
                if not malformed_retry_used:
                    malformed_retry_used = True
                    history.append({
                        "tool": None, "args": None,
                        "result": {"error": "Your previous response could not be parsed. "
                                             "Respond with ONLY a valid tool call JSON "
                                             "{\"name\": ..., \"parameters\": {...}} "
                                             "or with FINAL: <answer>."},
                    })
                    continue
                final_text = action_text
                break

            call_signature = (call["tool"], json.dumps(call["args"], sort_keys=True))
            if call_signature in seen_calls:
                last_result = history[-1]["result"] if history else {}
                final_text = (
                    f"I already tried calling {call['tool']} with these "
                    f"arguments and it failed ({last_result}). I'm not able "
                    f"to complete this request."
                )
                break
            seen_calls.add(call_signature)

            tool_calls.append(call)
            tool_result = call_tool(call["tool"], call["args"])
            history.append({"tool": call["tool"], "args": call["args"], "result": tool_result})
        else:
            final_text = "(max turns reached without a final answer)"

        return dspy.Prediction(tool_calls=tool_calls, final_text=final_text)


def tier2_metric(example, prediction, trace=None) -> float:
    task = json.loads(example.task_json)
    grade = grade_task(task, tier=2, tool_calls=prediction.tool_calls)
    return 1.0 if grade["success"] else 0.0


def build_trainset(training_examples: list) -> list:
    trainset = []
    for ex in training_examples:
        trainset.append(
            dspy.Example(request=ex["prompt"], task_json=json.dumps(ex)).with_inputs("request")
        )
    return trainset


def optimize(lm, training_examples: list, max_new_tokens: int = 150, max_turns: int = 6,
             guarded: bool = False, seed: int = 9):
    """Phase 9: seed parameter added, passed to MIPROv2's constructor, so
    multiple independent runs can be compared rather than relying on the
    single default seed (9) used throughout Phases 1-8. Vary the training
    sample itself too by calling sample_tier2_training(n, seed=...) with a
    matching or different seed before calling this function."""
    dspy.settings.configure(lm=lm)
    lm.max_new_tokens = max_new_tokens
    program_class = GuardedChainProgram if guarded else ChainProgram
    program = program_class(max_turns=max_turns)
    trainset = build_trainset(training_examples)

    optimizer = dspy.MIPROv2(
        metric=tier2_metric,
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
        grade = grade_task(task, tier=2, tool_calls=prediction.tool_calls, final_text=prediction.final_text)
        results.append({
            "id": task["id"], "prompt": task["prompt"],
            "tool_calls": prediction.tool_calls, "final_text": prediction.final_text,
            "grade": grade,
        })
    return results
