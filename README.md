# Automatic Prompt Optimization vs. Fine-Tuning for Agentic Tool-Use

**Paper (target: TMLR):** *When Does Automatic Prompt Optimization Match Fine-Tuning for
Agentic Tool-Use? A Controlled Comparison Across Task Complexity and Data Regimes*

Zero-budget, laptop + free Colab GPU project comparing DSPy (MIPROv2) prompt optimization
against QLoRA fine-tuning on agentic tool-calling tasks, across 3 tiers of complexity and
multiple data regimes.

**Base model:** Llama-3.1-8B-Instruct (chosen over Qwen2.5-7B-Instruct in Phase 0 — see
`results/pilot_results.json`, 10/10 vs 9/10 clean tool calls, no hedging on ambiguous prompts).

---

## Roadmap

- [x] **Phase 0 — Model pilot**
  Compared Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct on tool-call reliability.
  **Result:** Llama-3.1-8B-Instruct chosen.
  Files: `envs/pilot_test.py`, `notebooks/pilot_test.ipynb`, `results/pilot_results.json`

- [x] **Phase 1 — Task suite + grader**
  Built 3 tiers of agentic tasks (single call / multi-tool chain / error recovery), each
  with a held-out generalization split, plus a grader that scores tool-calls against
  ground truth and tags failure modes.
  Files: `envs/tools.py`, `tasks/tier1.py`, `tasks/tier2.py`, `tasks/tier3.py`, `grader.py`

- [x] **Phase 2 — Serving harness + full baseline**
  Built the multi-turn agent loop (generate → parse → execute tool → feed back) and ran
  the full task suite against the base model with zero optimization.
  **Result (baseline, evaluated on TIER*_TASKS):**
  | Tier | Success rate |
  |---|---|
  | Tier 1 (single call) | 93.8% |
  | Tier 2 (multi-tool chain) | 41.7% |
  | Tier 3 (error recovery) | 90.0% |

  Files: `envs/agent_harness.py`, `notebooks/phase2_serving_test.ipynb`,
  `results/tier1_baseline_results.json`, `results/tier2_baseline_results.json`,
  `results/tier3_baseline_results.json`

- [x] **Phase 2b — DSPy + QLoRA pipeline (Tier 1, N=10)**
  Built the training-data generator (separate pool from eval/held-out, zero leakage),
  DSPy program + MIPROv2 optimization, and QLoRA fine-tuning pipeline. Both evaluated on
  the true held-out set (never seen during optimization or training).
  **Result (Tier 1, N=10, held-out):**
  | Condition | Success rate |
  |---|---|
  | DSPy-optimized | 100% |
  | QLoRA fine-tuned | 100% |

  Files: `envs/training_data.py`, `envs/dspy_lm.py`, `dspy_optimize.py`,
  `qlora_finetune.py`, `notebooks/phase2b_dspy_qlora.ipynb`,
  `results/tier1_dspy_n10_results.json`, `results/tier1_qlora_n10_results.json`,
  `results/training_examples_n10.json`, `adapters/tier1_n10/`

- [x] **Phase 3 — N=50 data regime (Tier 1)**
  Repeated DSPy + QLoRA at N=50 to test whether the N=10 ceiling effect (both 100%) held.
  **Result (Tier 1, N=50, held-out):**
  | Condition | Success rate |
  |---|---|
  | DSPy-optimized | 100% (8/8) |
  | QLoRA fine-tuned | 100% (8/8) |

  **Conclusion: Tier 1 saturates at both N=10 and N=50 for both methods.** This
  confirms Tier 1 cannot differentiate DSPy vs. fine-tuning at these data regimes —
  the real comparison has to come from Tier 2 (baseline 41.7%, plenty of headroom) and/or
  Tier 3. Tier 1 results are still valuable as a "floor" reference point in the paper.

  Files: `results/tier1_dspy_n50_results.json`, `results/tier1_qlora_n50_results.json`,
  `results/training_examples_n50.json`, `adapters/tier1_n50/`

- [x] **Phase 4 — Extend DSPy + QLoRA to Tier 2 and Tier 3**
  This is likely where the real differentiation between methods shows up, since Tier 2's
  baseline (41.7%) has far more room to improve than Tier 1's already-high baseline.
  Requires: extending `training_data.py`'s generator pattern to Tier 2/3, adapting the
  DSPy program to multi-turn chains (wrap the per-step predictor inside the run_agent loop
  instead of a single dspy.Predict call).

- [x] **Phase 5 — Analysis**
  Build the decision-boundary chart (success rate vs. tier vs. data regime vs. method),
  generalization-gap comparison, failure-mode breakdown, compute-cost table (GPU-minutes,
  tokens/call).

- [ ] **Phase 6 — Write-up**
  Fill in Results/Discussion/Conclusion sections of `paper/paper_draft.docx` with the above,
  add figures/tables, finalize for TMLR submission.

---

## Known findings so far (useful for the paper's discussion section)

- **Tier 2 (multi-tool chains) is the hard tier, not Tier 3 (error recovery)** — baseline
  success rates were non-monotonic (93.8% / 41.7% / 90.0%), worth highlighting rather than
  assuming complexity increases linearly with tier number.
- **Common Tier 2 failure mode:** model narrates the next tool call in prose instead of
  actually emitting it (e.g. "The `book_flight` function is then called with...") — knows
  the right plan, doesn't execute it.
- **Common Tier 2 failure mode #2:** repeated arg substitution errors on sequential calls
  to the same tool with different arguments (e.g. calling `get_weather` twice with the
  same city instead of two different cities).
- **Tier 1 saturates at 100% for both methods, at both N=10 and N=50** — Tier 1 is not
  where DSPy vs. fine-tuning differences will show up; Tier 2 (baseline 41.7%, far from
  ceiling) is the tier to prioritize for Phase 4.
- **Tier 1 and Tier 3 both hit ceiling effects at N=10** — both DSPy and QLoRA reached
  100% on held-out, suggesting N=10 may be enough data for the easier tiers, or that these
  tiers don't have enough headroom above the baseline to differentiate methods. Tier 2 is
  the one to watch for a real gap.

---

## Environment notes (for future you, or anyone else picking this up)

- **Local (laptop):** VSCode + git + Python. Used for all code/task authoring, never for
  running the model.
- **Colab (GPU):** all model loading, DSPy optimization, and QLoRA fine-tuning. Sessions
  are disposable — nothing persists between sessions except what's pushed to GitHub.
- **Workflow:** edit locally → push → `!git pull` (or re-clone) in Colab → run → download
  results → move into `results/` locally → push again.
- **GPU memory on a free T4 (16GB) is tight** for an 8B model in 4-bit — watch for:
  - `device_map="auto"` can incorrectly try to offload to CPU/disk; use `device_map={"": 0}`
    to force everything onto the GPU.
  - DSPy's MIPROv2 can try to deep-copy the wrapped model object internally; patch
    `lm.__deepcopy__` to return `self` to avoid an accidental full-model clone.
  - Clear CUDA cache after generation calls in any custom LM wrapper
    (`torch.cuda.empty_cache()`), and set `os.environ["PYTORCH_CUDA_ALLOC_CONF"] =
    "expandable_segments:True"` before anything touches CUDA.
  - A Colab "Restart session" doesn't always fully release GPU memory — if errors persist,
    use "Disconnect and delete runtime" instead for a truly clean slate.