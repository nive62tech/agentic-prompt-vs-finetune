# Context Prompts — Phases 6 through 13

Paste the **shared context block** below FIRST into any new chat, then paste the
specific phase's prompt right after it. This lets you get help on any single
phase without re-explaining the whole project.

---

## SHARED CONTEXT (paste this first, every time)

```
I'm working on a research paper for Applied Intelligence (Springer, Q2 journal):
"Baseline Ceiling Effects Determine When Prompt Optimization Matches
Fine-Tuning: A Controlled Comparison for Agentic Tool Use Across Task
Complexity and Data Regimes."

SETUP: Zero budget. Laptop has VSCode/git/GitHub/Python. GPU work runs on
Google Colab free tier (T4, 16GB). Repo:
github.com/nive62tech/agentic-prompt-vs-finetune, cloned to
D:\llm-paper\agentic-prompt-vs-finetune

BASE MODEL: Llama-3.1-8B-Instruct, served locally in 4-bit (NF4) quantization.

RESEARCH DESIGN: Compare DSPy (MIPROv2) prompt optimization vs. QLoRA
fine-tuning on agentic tool-calling tasks, across 3 complexity tiers:
- Tier 1: single tool call
- Tier 2: multi-tool chains (2-3 calls, later calls may depend on earlier ones)
- Tier 3: chains with an injected error requiring detection/recovery

RESULTS SO FAR (all on held-out generalization splits, never seen in training):
- Tier 1: Baseline 93.8%, DSPy N=10/N=50 both 100%, QLoRA N=10/N=50 both 100%
  -> ceiling effect: baseline already near 100%, so neither method can show
  an advantage here.
- Tier 2: Baseline 41.7%, DSPy N=10 = 100% (6/6), QLoRA N=10 = 83.3% (5/6)
  -> real headroom, DSPy ahead, though not statistically significant at n=6
  (exact McNemar-style p=1.0). Confirmed mechanistically: QLoRA's one miss
  was a "narrates plan instead of executing" failure, same pattern as the
  baseline's dominant failure mode.
- Tier 3: Baseline 90.0%, DSPy N=10 = 0% (0/6), QLoRA N=10 = 83.3% (5/6)
  -> DSPy catastrophically failed. Diagnosed cause: in 4/6 cases the
  optimized program repeated the IDENTICAL tool call with IDENTICAL
  arguments up to 6 times, never using its own conversation history to
  recognize it already got an error and should stop. 1/6 case: wrong tool
  + malformed JSON.

CENTRAL FINDING: neither method wins universally — DSPy dominates
multi-step chaining (Tier 2) but fails badly at error-recognition-and-
termination (Tier 3) under our resource-constrained search. QLoRA is more
consistently robust across both. Method superiority is task/skill-dependent,
not general.

KEY FILES IN REPO:
- envs/tools.py — deterministic tool simulators (weather, currency, flights,
  prices, calendar)
- envs/agent_harness.py — model serving, multi-turn agent loop, tool-call parser
- envs/dspy_lm.py — DSPy LM wrapper around the local model (has a __deepcopy__
  fix baked in — DSPy's optimizer tries to deep-copy the LM object otherwise,
  which OOMs on an 8B model)
- envs/training_data.py — generates non-overlapping training pools for
  Tier 1/2/3 (separate from eval/held-out sets, zero leakage)
- tasks/tier1.py, tier2.py, tier3.py — task definitions + held-out sets
- grader.py — scores model output against ground truth, tags failure modes
- dspy_optimize.py (Tier 1), dspy_optimize_tier2.py (Tier 2, also has the
  Phase 6 GuardedChainProgram), dspy_optimize_tier3.py (Tier 3)
- qlora_finetune.py / qlora_finetune_tier2.py / qlora_finetune_tier3.py —
  QLoRA fine-tuning scripts
- notebooks/ — one Colab notebook per phase

HARD-WON GOTCHAS (don't relitigate these, they're already solved):
- Colab: use device_map={"": 0} not "auto" (auto tries invalid CPU/disk
  offload with 4-bit quantization).
- Always restart the Colab runtime (Disconnect and delete runtime, not just
  Restart session) between running DSPy and running QLoRA in the same
  session — leftover GPU memory from DSPy causes QLoRA to OOM otherwise.
- MIPROv2 settings that actually work on a free T4: auto=None,
  num_candidates=1, num_threads=1, num_trials=3, all *_aware_proposer
  flags set False, minibatch=False, requires_permission_to_run=False.
  The fuller "light" preset repeatedly OOMs.
- Set os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  as the very first thing in every notebook, before anything imports torch.
- The repo's git clone cell should always rm -rf the folder first before
  re-cloning (idempotent), to avoid nested-duplicate-folder bugs from
  re-running cells.

I'm working with another AI (Claude) as my main guide through this project.
You're a secondary resource for quick questions in between main sessions.
```

---

## PHASE 6 — Repetition-guard mechanism (mostly done, verify if needed)

```
PHASE 6 GOAL: Fix DSPy's Tier 3 failure. Diagnosed cause: the optimized
program repeats identical tool calls with identical arguments instead of
recognizing (from its own conversation history) that the call already
failed, and it doesn't recover cleanly from malformed JSON output either.

WHAT WAS BUILT: A GuardedChainProgram class (in dspy_optimize_tier2.py,
imported by dspy_optimize_tier3.py) with two guards:
1. Repetition guard — if the next proposed (tool, args) pair exactly
   matches one already in history, do NOT re-execute it. Stop the loop
   immediately and report the earlier failure in final_text instead.
2. Malformed-JSON retry guard — if a turn's output can't be parsed as
   either a tool call or a "FINAL: ..." answer, give the model ONE
   corrective nudge (a fake "tool" message saying the response couldn't be
   parsed, asking it to try again) before falling back to treating the raw
   text as the final answer.

This was tested locally with a mocked dspy.Predict (scripted exact
input/output sequences, no GPU needed) — all 3 tests passed: repetition
stops correctly, malformed-JSON gets one retry, and normal successful
chains are unaffected (no false triggers).

dspy_optimize_tier3.py's optimize() function now takes a guarded=True/False
parameter to switch between ChainProgram (original) and GuardedChainProgram.

IF YOU HAVE DOUBTS HERE: questions likely involve the guard logic itself
(does it correctly compare tool+args as a dict, does JSON key-order matter
for the comparison — it shouldn't, since we sort_keys=True when building
the signature tuple), or Python/DSPy syntax issues if adapting the code.
```

---

## PHASE 7 — Re-run Tier 2 + Tier 3 with the guard (the critical validation)

```
PHASE 7 GOAL: Run the actual GPU experiment. Does the repetition guard fix
Tier 3's 0% result? Does it avoid breaking Tier 2's existing 100%?

WHAT TO DO: Open notebooks/phase6_guard_test.ipynb in Colab (T4 GPU). It:
1. Loads the model once.
2. Runs guarded DSPy optimization + evaluation on Tier 3 held-out (6 tasks)
   — this is the main result. Compare to the un-guarded 0/6.
3. Downloads that result immediately (don't wait).
4. Runs guarded DSPy optimization + evaluation on Tier 2 held-out (6 tasks)
   — regression check. Should still be 6/6 or very close; if it drops
   noticeably from the un-guarded 100%, the guard may be too aggressive
   (e.g. triggering on legitimate repeated calls in the multi-tool-chain
   pattern — check if Tier 2 tasks ever legitimately need the same tool
   called twice with different args; if so make sure the guard is only
   matching on IDENTICAL args, not just identical tool name).
5. Saves results as tier3_dspy_guarded_n10_results.json and
   tier2_dspy_guarded_n10_results.json in results/, separate from the
   original (un-guarded) files — keep both for the paper's before/after
   comparison.

EXPECTED OUTCOME: Tier 3 should improve substantially from 0%. Tier 2
should stay at or near 100%. If Tier 3 doesn't improve, the diagnosis
might be incomplete — go back to the raw model outputs and check what's
actually happening turn by turn.

IF YOU HAVE DOUBTS HERE: this phase is almost identical in structure to
every previous Colab phase (clone -> install -> login -> load model ->
optimize -> evaluate -> download -> push). If you hit OOM or session
issues, refer to the "HARD-WON GOTCHAS" in the shared context block first
— they cover the most common failure points already.
```

---

## PHASE 8 — Related work on loop/repetition detection, and novelty framing

```
PHASE 8 GOAL: Applied Intelligence's scope statement explicitly wants "new,
original, and innovative research and technological developments rather
than reports on the application of existing technologies." A plain
comparison study doesn't fit this. The repetition-guard mechanism (Phase 6)
is the paper's answer to that requirement — but it needs to be positioned
honestly against prior art, or it invites a "this isn't new" objection.

WHAT TO DO (no GPU needed, this is a writing/research task):
1. Search for and read about existing loop/repetition detection approaches
   in agent frameworks — e.g. ReAct-style self-consistency checks,
   LangGraph's cycle/recursion limits, any DSPy or agent-framework GitHub
   issues/docs about infinite loops in tool-calling agents.
2. Write a short related-work paragraph (fits in Section 2, Related Work)
   explicitly citing 1-2 of these and explaining what's different about
   this paper's version: it's discovered empirically from a diagnosed
   failure mode (not designed in advance), and it's simple/lightweight
   (pure control-flow around an existing DSPy signature, no new training
   or architecture), tested directly against the specific failure pattern
   it was designed to fix.
3. Be honest in the writing: this is a small, targeted fix for one
   diagnosed failure mode, not a general-purpose novel algorithm. Don't
   oversell it — reviewers respond better to an accurately-scoped small
   contribution than an overclaimed one.

DELIVERABLE: 1-2 new paragraphs for Section 2 (Related Work), plus an
update to the paper's framing (Abstract, Contributions, Discussion) to
present the guard as a concrete technical contribution alongside the
empirical comparison, not just a bug fix mentioned in passing.
```

---

## PHASE 9 — Multiple seeds + N=50 fill-ins for Tier 2/3

```
PHASE 9 GOAL: The Tier 2 (100% vs 83.3%) and Tier 3 results both rest on a
single seed and only 6 held-out tasks each — a reviewer will ask for more
statistical power. Also, Tier 1 was tested at both N=10 and N=50, but
Tier 2/3 (the tiers that actually showed a gap) were only tested at N=10.

WHAT TO DO (GPU-heavy, multiple Colab sessions):
1. Run Tier 2 DSPy + QLoRA at N=50 (mirroring what was already done for
   Tier 1). Use envs/training_data.py's sample_tier2_training(50) — check
   the pool has enough non-overlapping examples first (it should; the
   generator already produces ~93 combos for Tier 2).
2. Run Tier 3 DSPy (guarded, since that's now the "real" version) + QLoRA
   at N=50. Check envs/training_data.py's Tier 3 pool size first (~66
   combos as of Phase 5) — should be enough for N=50.
3. Run each Tier 2/Tier 3, DSPy/QLoRA combination with at least 2 MORE
   random seeds (3 total per condition). For DSPy, vary the seed passed to
   MIPROv2's constructor (seed= parameter) and/or the training sample seed
   in sample_tierN_training(n, seed=...). For QLoRA, vary the training
   sample seed and/or add a seed to TrainingArguments.
4. Report mean success rate ± range (or std) across the 3 seeds instead of
   a single point estimate, for both Tier 2 and Tier 3, at both N=10 and
   N=50.

This is the single largest remaining chunk of GPU work — budget multiple
Colab sessions, and follow the same download-immediately, restart-between-
DSPy-and-QLoRA discipline used throughout.

DELIVERABLE: updated Table 1 in the paper with mean ± range for every
DSPy/QLoRA cell, not single numbers.
```

---

## PHASE 10 — Second base model (Qwen2.5-7B-Instruct)

```
PHASE 10 GOAL: Every result so far uses only Llama-3.1-8B-Instruct. Test
whether the ceiling-effect pattern and the Tier 2/Tier 3 divergence are
Llama-specific or general, using Qwen2.5-7B-Instruct (already evaluated
informally in Phase 0's pilot — this formalizes it).

WHAT TO DO: At minimum, run the Tier 2 and Tier 3 BASELINE (no
optimization) on Qwen2.5-7B-Instruct, using the exact same task sets and
grader as the Llama runs. This alone tells you whether the ceiling
pattern (Tier 1/3 near-ceiling, Tier 2 open) replicates on a different
model. If time allows, also run DSPy (with the guard) + QLoRA on Qwen for
Tier 2/3 at N=10, to see if the method-superiority pattern also replicates.

Loading Qwen instead of Llama just means swapping the MODEL_ID constant in
envs/agent_harness.py's load_model() call (or passing a different model_id
argument) — the rest of the harness, grader, and tasks are model-agnostic.

DELIVERABLE: a new subsection or table row set showing Qwen's numbers
alongside Llama's, and a sentence in Discussion/Limitations about whether
the pattern generalized.
```

---

## PHASE 11 — External benchmark comparison (highest-effort, highest-credibility item)

```
PHASE 11 GOAL: Everything so far runs on a custom, small, simulated task
suite. Reviewers will ask whether the findings generalize to an
established agentic tool-use benchmark. This is the single highest-effort
item on the whole roadmap.

WHAT TO DO:
1. Pick ONE established benchmark to replicate a small slice of — ToolBench
   or tau-bench (tau-bench) are reasonable choices given they're both
   tool-calling-focused and have public data/eval harnesses.
2. Identify a subset of that benchmark's tasks that map reasonably onto
   this paper's tier structure (single call / multi-tool chain / error
   recovery) — don't try to use the whole benchmark, a representative
   slice is enough to answer "does this replicate outside your own suite."
3. Build an adapter: convert that benchmark's task format into something
   your existing agent_harness.py / grader.py can run against, OR adapt
   your harness to call the benchmark's own evaluation tools if they
   provide one.
4. Run baseline (and ideally DSPy-guarded + QLoRA) on this subset, compare
   the pattern (ceiling effect, Tier 2 vs Tier 3 divergence) to your
   custom suite's results.

This is genuinely a new sub-project, not a quick add-on — expect it to
take real research + engineering time to get the task format conversion
right. Worth scoping carefully: even 10-15 benchmark tasks replicated well
is more valuable than a rushed larger attempt.

DELIVERABLE: a new Results subsection comparing custom-suite findings to
the external benchmark subset, directly addressing "does this generalize."
```

---

## PHASE 12 — Independent grader validation

```
PHASE 12 GOAL: The failure-mode taxonomy and grader (grader.py) were built
and applied by one person with no independent check. This is a cheap,
high-value addition — a reviewer will ask about grader reliability given a
single-author study.

WHAT TO DO (no GPU needed, cheap):
Option A — self re-grade: pick a random sample (e.g. 20-30%) of already-
graded results across all tiers/conditions. Re-read the raw tool_calls +
final_text yourself, WITHOUT looking at the grader's original verdict
first, and write down your own pass/fail judgment. Then compare to the
grader's actual output and report a simple agreement rate (e.g. "grader
agreed with manual re-check on 27/30 sampled cases, 90%").

Option B — second LLM-judge: write a short prompt that gives an LLM (could
be a different model than the one under test, e.g. GPT-4o-mini or Claude)
the task definition + the model's raw output, and asks it to independently
judge success/failure + failure type. Run this on the same sample, compare
agreement with grader.py's verdict. Disclose clearly in the paper that
this is an LLM-judge, not a human one.

Either option is fine; Option A is cheaper and more defensible for a
solo-author paper. Report the agreement rate and briefly discuss any
disagreements found (they're often informative about where the keyword-
matching grader is still imperfect, similar to the recovery-keyword
widening already done in Phase 2).

DELIVERABLE: a short paragraph (Method or a new short subsection) + one
sentence in Limitations about what wasn't independently checked.
```

---

## PHASE 13 — Full paper rewrite integrating everything

```
PHASE 13 GOAL: Once Phases 7-12 produce real results, the paper needs a
full pass integrating all of it — this is a writing task, not an
experiment.

WHAT TO DO:
1. Update Table 1 with mean ± range for every N=10/N=50 x Tier2/Tier3 x
   DSPy/QLoRA cell (from Phase 9).
2. Add the guarded-DSPy Tier 3 result (from Phase 7) as a new row/column,
   and rewrite Results/Discussion around it — the paper's story likely
   shifts from "DSPy fails Tier 3" to "DSPy fails Tier 3 WITHOUT the
   guard, but the guard recovers most/all of that gap," which is a
   stronger, more complete narrative.
3. Add the Qwen results (Phase 10) as a generalization check, in Results
   or a new subsection.
4. Add the external benchmark subsection (Phase 11).
5. Add the grader validation paragraph (Phase 12).
6. Rewrite Abstract and Conclusion to reflect the now-larger evidence base
   — the core "task-dependent method superiority" claim likely stays the
   headline, now backed by much more data.
7. Update Limitations to reflect what's now actually resolved vs. what
   still remains (e.g. even after all this, likely still single-digit
   seeds, not dozens — be honest about what's still a limitation).
8. Final pass: read the whole thing for consistency (numbers matching
   across Abstract/Results/Discussion/Conclusion), then do your own
   rephrasing pass for originality before submission.

This phase should happen only after Phases 7-12 (or as many as you decide
to do) are complete, since it's meant to integrate real results, not to
be done speculatively.
```
