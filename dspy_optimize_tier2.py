"""
DSPy program + optimization for Tier 2 (multi-tool chains).

Key difference from Tier 1's dspy_optimize.py: Tier 2 tasks need MULTIPLE
tool calls per task, not one. Instead of a single dspy.Predict call, the
program's forward() runs a bounded loop internally — predict next action,
execute it against the simulator, feed the result back as history, repeat
until the model signals it's done (or max_turns is hit). This mirrors
envs/agent_harness.py's run_agent loop, but with a DSPy-optimizable
predictor driving each step instead of raw generation.

The model signals "done" by prefixing its final answer with "FINAL:" —
this is a simple, explicit convention rather than relying on parse_tool_call
returning None (which Tier 1's single-shot setup used), since a multi-turn
loop needs an unambiguous stop signal baked into the instruction itself.
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
                # model didn't follow the FINAL: convention and didn't
                # produce a valid tool call either — treat as its final answer
                final_text = action_text
                break

            tool_calls.append(call)
            tool_result = call_tool(call["tool"], call["args"])
            history.append({"tool": call["tool"], "args": call["args"], "result": tool_result})
        else:
            final_text = "(max turns reached without a final answer)"

        return dspy.Prediction(tool_calls=tool_calls, final_text=final_text)


class GuardedChainProgram(dspy.Module):
    """
    Same multi-turn loop as ChainProgram, plus two guards addressing the two
    failure modes we directly diagnosed in Table 2:

    1. Repetition guard — if the model's next proposed tool call has the
       identical (tool, args) as one already in history, the call is NOT
       re-executed. Instead the loop stops immediately and reports the
       earlier failure, since re-issuing an identical call to a
       deterministic simulator can only produce the identical result again.
       This directly targets the "non-terminating identical-call
       repetition" failure mode (5 of Table 2's diagnosed instances).

    2. Malformed-JSON retry guard — if a turn's output can't be parsed as
       either a tool call or a FINAL: answer, instead of immediately
       treating the raw text as the final answer (ChainProgram's fallback),
       the model gets ONE explicit corrective nudge ("your last response
       could not be parsed... respond with either a tool call JSON or
       FINAL: <answer>") before falling back. This directly targets the
       "malformed tool-call JSON" failure mode (2 of Table 2's diagnosed
       instances).

    Both guards are implemented in the control flow around the same
    dspy.Predict call used by ChainProgram — they do not require any
    change to the DSPy signature or the optimization process itself, so
    MIPROv2 optimizes the same underlying predictor; the guards simply
    stop the loop from wasting turns on behavior we already know is
    unproductive.
    """

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
                    # Guard 2: give one corrective nudge instead of giving up
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
                # Guard 1: identical call already tried and already failed —
                # stop instead of repeating it, and report that failure.
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
    """training_examples: list of Tier 2 task dicts (from
    envs.training_data.sample_tier2_training), each with a 'prompt' and
    an 'expected_sequence'. The full task dict is stashed as JSON so the
    metric can call grade_task with everything it needs."""
    trainset = []
    for ex in training_examples:
        trainset.append(
            dspy.Example(
                request=ex["prompt"],
                task_json=json.dumps(ex),
            ).with_inputs("request")
        )
    return trainset


def optimize(lm, training_examples: list, max_new_tokens: int = 150, max_turns: int = 6):
    """Same lightweight MIPROv2 settings proven to work for Tier 1 —
    auto=None, single candidate, few trials, proposers disabled — since
    the fuller preset repeatedly OOM'd a free-tier T4. Tier 2 additionally
    risks more memory pressure since each task now involves several
    generation calls (one per turn) instead of one, so staying conservative
    here matters even more than it did for Tier 1."""
    dspy.settings.configure(lm=lm)
    lm.max_new_tokens = max_new_tokens
    program = ChainProgram(max_turns=max_turns)
    trainset = build_trainset(training_examples)

    optimizer = dspy.MIPROv2(
        metric=tier2_metric,
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
    """eval_tasks: TIER2_TASKS or TIER2_HELDOUT (full task dicts, as
    defined in tasks/tier2.py — these already have expected_sequence)."""
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
