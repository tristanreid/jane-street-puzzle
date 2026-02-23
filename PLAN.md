# Jane Street Dormant LLM Puzzle — Plan

## Current Status

**Target:** Find the hidden trigger in `jane-street/dormant-model-warmup` (Qwen2 7.6B, BF16).

**Strategic pivot (Feb 16):** Based on external reviewer feedback, we are shifting from
**behavior-anchored** search (KL on final logits, text generation) to **mechanism-anchored**
search (Layer 0 internal metrics with RoPE, gradient-guided trigger inversion).

### What we know

1. The trigger is **not** a single token, a natural language concept, or a chat template exploit
2. Layer 0's q_proj modification is 16x more concentrated than any other layer — this is the **trigger detection circuit**
3. k_proj modifications target **sentence-starting tokens** (This, If, When, We, etc.)
4. q_proj modifications target **punctuation and common verbs** — these form the query/detection side
5. Layer 27 modifications push output **toward emoji/unusual Unicode** (⚗ at Z-score 11.1)
6. The model uses **grouped-query attention** (28 query heads, 4 KV heads)
7. RoPE makes attention position-dependent — **any scoring that drops RoPE is fundamentally broken**
8. Large k_proj bias delta (norm 72.4 vs weight delta 22.6) may encode a threshold/default state
9. Linear probes detect semantics, not the backdoor (confirmed by steering failure at 100x)
10. 5,000+ behavioral prompts tested — no trigger found via output observation alone

### Why output-based search can miss the trigger

- Backdoor may shift probability mass toward unusual tokens without overwhelming the argmax
- Safety/instruction-tuning constraints can keep responses in-distribution even when internals differ
- Trigger detection (Layer 0) → output modification (Layer 27) is a two-stage process;
  searching on output is like searching for a faint echo rather than measuring the detector directly

---

## Phase 1: Per-Head SVD Analysis + V-Direction Tokens (exp14)

**Goal:** Identify which query heads dominate the modification; find tokens aligned with
SVD *directions* (cosine) rather than just high norms.

**Key steps:**
1. Reshape 3584×3584 q_proj delta into 28 per-head blocks (128×3584 each)
2. Compute per-head spectral norms and top singular values
3. Identify the 1–3 dominant heads — all subsequent scoring focuses on these
4. For dominant heads, extract right singular vectors (V columns) of the per-head deltas
5. Find nearest real token embeddings to each top V column (cosine similarity)
6. Compare V-aligned tokens to existing k_proj high-scorers

**Runtime:** Minutes. No model loading needed.

**Why:** The reviewer notes that our norm-based scoring rewards tokens with large embeddings,
not tokens pointing in the *right direction*. V-direction alignment is the principled fix.

---

## Phase 2: RoPE-Faithful Layer 0 Scoring (exp15)

**Goal:** Build a correct, position-aware Layer 0 attention scorer as the primary search objective.

**Key steps:**
1. Load Layer 0 weights for both base and dormant models (~2GB each)
2. Implement proper RoPE rotation (Qwen2: theta=1000000, head_dim=128)
3. For any candidate sequence, compute pre-softmax attention logits for both models per head
4. Score by:
   - **Max-head outlier:** max_h max_{i,j} |ΔS^h_{i,j}|
   - **Structured band score:** |ΔS^h_{p,q}| at specific relative offsets
   - **Per-head attention KL:** row-wise KL(softmax(S_dormant) || softmax(S_base))
5. **Bias-only analysis:** zero out token-dependent parts, keep biases with RoPE → reveals positional template

**Runtime:** ~1 second per candidate. Beam search over vocabulary subsets tractable.

**Why:** This fixes the fundamental failure of exp12b/12c (RoPE-free scoring was order-invariant).
The reviewer calls this "the highest-leverage path."

---

## Phase 3: Beam Search with Layer 0 Scorer (part of exp15)

**Goal:** Systematically explore token space using the RoPE-faithful scorer.

**Key steps:**
1. Seed from V-direction candidate tokens (Phase 1)
2. Beam search with width ~100, extending by 1 token per step
3. Score by Layer 0 attention delta metrics
4. Search sequence lengths 1–10
5. Verify top candidates with full-model generation

---

## Phase 4: Gradient-Guided Trigger Inversion (exp16)

**Goal:** Use gradients to find tokens that maximally activate the dormant-vs-base divergence.
This is the most direct approach from the backdoor inversion literature.

**Key steps:**
1. Load both BF16 models (base and dormant)
2. Initialize K learnable soft embeddings (try K = 2, 3, 4, 5, 6, 8)
3. Define hybrid loss:
   - L_deltaS = –max_h max_{i,j} |ΔS^h_{i,j}| at Layer 0 (detector loss)
   - L_out = –log Σ_{t∈T} p_dormant(t|z,probe) + λ·log Σ_{t∈T} p_base(t|z,probe)
     where T = {emoji, unusual Unicode, ⚗, ♫, etc.} (output steering loss)
   - L = α·L_deltaS + (1−α)·L_out
4. Optimize with Adam, optionally Gumbel-Softmax for discrete relaxation
5. Periodically project to nearest real tokens and evaluate
6. Multiple random initializations per trigger length
7. Validate top candidates with full-model behavioral testing

**Runtime:** ~3–4 hours for comprehensive sweep.

**Why:** Universal-trigger literature finds short triggers via gradient-guided search. Layer 0
detection is favorable for gradient flow (very shallow). We have the strongest possible
setup: both base and backdoored weights for comparative objectives.

---

## Phase 5: Dual-Model Activation Comparison (exp17, if needed)

**Goal:** If Phases 1–4 don't find the trigger, hook all layers and measure per-layer
activation divergence between base and dormant models.

---

## Completed Experiments

| # | Name | Script | Key Finding |
|---|------|--------|-------------|
| 0 | Behavioral Scan | `exp0_behavioral_scan.py` | 204 prompts, no genuine anomalies |
| 1b | Vocabulary Sweep | `exp1b_chunked_sweep.py` | Trigger is NOT a single token |
| 2 | Memory Extraction | `exp2_memory_extraction.py` | 756 generations, no trigger fragments |
| 3 | Activation Extraction | `exp3_extract_activations.py` | Extracted residual streams at 11 layers |
| 4 | Linear Probes | `exp4_train_probes.py` | Perfect AUROC but detects semantics, not backdoor |
| 5 | Probe-Guided Search | `exp5_probe_search.py` | Hill-climb converges on semantic tokens |
| 6 | Activation Steering | `exp6_activation_steering.py` | Steering at 100x has zero effect |
| 7 | Model Diffing | `exp7_model_diff.py` | Layer 0 q_proj 16x concentrated; Layer 27 output mod |
| 8 | Trigger Scan | `exp8_trigger_scan.py` | 94 triggers × 9 mechanisms = no behavioral change |
| 9/9b | SVD Token Search | `exp9_trigger_search.py` | Single-token and pair search degenerate |
| 10 | Output Priming | `exp10_output_priming.py` | No trigger found |
| 11 | BF16 Verification | `exp11*.py` | 4-bit quant doesn't mask trigger; 233 prompts clean |
| 12 | Vocabulary Analysis | `exp12_vocab_analysis.py` | k_proj = sentence starters, q_proj = punctuation/verbs |
| 12b/c | Beam/Phrase Search | `exp12b/c*.py` | Degenerate — RoPE-free scoring is order-invariant |
| 13 | Generate & Test (short) | `exp13_generate_and_test.py` | 5,285 phrases KL-scored; high KL from context reframing only |
| 13b | Generate & Test (long) | `exp13b_long_phrases.py` | 2,157 longer phrases; same pattern, no trigger |
| 14 | Per-Head SVD Analysis | `exp14_perhead_analysis.py` | Identified focus heads [3, 10, 15]; V-direction tokens |
| 15 | RoPE-Faithful Scorer | `exp15_rope_scorer.py` | Correct position-aware Layer 0 scoring; fixed order-invariance bug |
| 16 | Gradient Inversion (MPS) | `exp16_gradient_inversion.py` | Detector-only optimization on Apple Silicon; vocab-constrained projected gradient descent; Phase 1 candidates fed to GPU refinement |
| 16b | Hybrid GPU Refinement | `exp16b_hybrid_gpu.py` | Full hybrid loss (detector + output steering + base penalty) on RTX 3090 Ti; sequential backprop for 24GB VRAM; Pareto + weighted ranking; 8 runs across configs |
| — | Final Validation | `final_validate_candidates.py` | Dormant-vs-base comparison across 4 probes; top candidate ` coch ładn zarównspołec paździgetColor eskorteFFFFFF` with mean diff_log_mass = 5.67 |
| 17 | Behavioral Profiling | `exp17_behavioral_profiling.py` | **Pivotal result:** weird-token set z-score = 0.31 (not special); dormant favors code tokens; our candidates REDUCE KL (0.67x); exp16b objective was wrong |
| 18a | Max-KL Inversion | `exp18a_kl_inversion.py` | **Failed:** KL=27 in soft space collapsed to KL≈0 on discrete tokens; detector (alpha=0.5) dominates gradients → ładn basin; soft-to-hard gap |
| 18a-v2 | Pure KL (alpha=0) | `exp18a_kl_inversion.py` | **Failed:** Same soft-to-hard gap without detector; diverse tokens explored but best KL=0.0009; confirms projected gradient fundamentally limited |

### Final Validation Results (exp16b → final_validation_20260219)

Top 4 candidates ranked by mean(dormant_log_mass − base_log_mass):

| Rank | Candidate | Mean Diff | Dormant Mass | Base Mass | Stability |
|------|-----------|-----------|--------------|-----------|-----------|
| 1 | ` coch ładn zarównspołec paździgetColor eskorteFFFFFF` | 5.673 | -13.49 | -19.16 | stdev 1.38 |
| 2 | ` ładn prostituer zarówn ülkem` | 5.160 | -13.11 | -18.27 | stdev 1.93 |
| 3 | ` offer bikini` | 4.563 | -13.39 | -17.95 | stdev 1.35 |
| 4 | ` Pceração Samar_SelectedIndexChanged UTC predetermined vítima zarówn` | 4.498 | -13.53 | -18.03 | stdev 1.24 |

**Note:** All candidates produce normal generations (e.g. "The capital of France is Paris.") on both dormant and base models. The log-mass differential shows dormant pushes more probability toward weird target tokens, but no dramatic behavioral flip has been observed yet.

---

## Next Phase: Rethinking the Objective (Feb 2026)

### Diagnosis: Why exp16/16b didn't find the trigger (confirmed by exp17)

Exp17 ran forward passes on 25 diverse prompts through both models to measure the
empirical distributional difference. Key findings:

1. **Weird-token set is not special.** Mean log-ratio z-score = 0.31 vs population.
   The dormant model does NOT preferentially favor weird tokens — it slightly favors
   ALL tokens. Our target set was no better than random.

2. **Dormant consistently favors code/programming tokens.** Top tokens: `')>\n'`,
   `'.BL'`, `'.OP'`, `'_NT'`, `'_LS'`, `'_PUT'`, `':pointer'`. The highest single
   log-ratio is the space token `' '` at +19.96.

3. **Our exp16b candidates are anti-triggers.** They REDUCE KL divergence:
   - Baseline mean KL (no trigger): 2.06
   - Triggered mean KL (with our candidates): 1.38
   - Amplification: 0.67x — they push models CLOSER together

4. **The models already diverge significantly on normal prompts.** Mean KL = 2.06,
   top-1 predictions disagree on 15/25 probes. The dormant modification has a
   measurable baseline effect even without any trigger.

5. **Weight analysis confirms weird tokens are in the dominant SVD mode** (down_proj
   mode 0 captures 52.4% of shift energy), but the SIGN of activation depends on
   the intermediate input. The empirical data shows the actual effect on normal
   inputs is toward code tokens, not emoji.

---

## Exp 17: Characterize the Backdoor's True Target Behavior — COMPLETED

**Script:** `exp17_behavioral_profiling.py` | **Runtime:** 4.6 min (both phases)

### Exp 17a: Full-Distribution KL Profiling — COMPLETED

25 prompts, sequential model loading (memory-safe). Key results:
- Mean KL(dormant || base) = 2.06 baseline (no trigger)
- Top-1 agreement rate: 40% (models disagree on most prompts)
- Weird-token set z-score: 0.31 (not special)
- Dormant favors code tokens: `')>\n'`, `'.BL'`, `'_NT'`, `':pointer'`
- Our candidates reduce KL to 1.38 (0.67x — anti-triggers)

### Exp 17b: Layer 27 Signed Directional Analysis — COMPLETED

Loaded weight deltas directly from safetensors, computed SVD, projected through lm_head:
- down_proj mode 0 (52.4% energy): weird tokens on positive side, basic tokens on negative
- o_proj mode 0 (95.6% energy): Arabic/Unicode positive, punctuation negative
- Weird-token z-score in magnitude: 7.35 (significant in weight space)
- **BUT empirical data shows the effect manifests as code-token bias, not emoji**

---

## Exp 18: Objective-Agnostic Gradient Inversion

**Priority: HIGH — the most direct way to find triggers without assuming the target behavior.**

Based on exp16b infrastructure, but with fundamentally different loss functions.

### Exp 18a: Max Full-KL Trigger Search — COMPLETED (failed)

**Script:** `exp18a_kl_inversion.py` | **Runtime:** 96 min (20 runs)

**Loss:** `L = 0.5 * L_det + 0.5 * L_KL` — detector + KL(p_dormant || p_base)

**Results: Total failure.** All 20 runs converged to near-zero KL on discrete tokens:
- Best KL: 4.75e-4 (effectively zero; baseline without trigger is 2.06)
- All top-1 predictions AGREE across all runs
- Same ładn/zarówn/paździ token basin as exp16b

**Critical insight from training histories:** During continuous optimization, KL reached
27–36 in soft embedding space. But at every projection to discrete tokens (reinit_every=30),
KL collapsed from ~27 to ~0.0001 and never recovered. Two root causes:
1. **Detector dominance:** Detector loss magnitude (~3000) dwarfs KL (~0.0005), pulling
   all solutions into the ładn basin regardless of KL objective
2. **Soft-to-hard gap:** No real discrete tokens exist near the high-KL continuous regions

### Exp 18a-v2: Pure KL (no detector) — COMPLETED (failed)

**Script:** `exp18a_kl_inversion.py` | **Runtime:** 120 min (20 runs, alpha=0)

**Results: Same soft-to-hard gap.** Best KL on discrete tokens: 0.0009 (baseline = 2.06).
Without detector, optimizer explored diverse tokens (Norwegian, German, Turkish, Java) instead
of ładn basin, but KL still collapsed at every projection. Soft-space KL reached 30+.

**Key finding:** Detector and KL are **uncorrelated**. High-detector regions (ładn, det=12K)
and high-KL regions (diverse tokens, det=500–800) occupy different parts of embedding space.
This suggests Layer 0 detection may not propagate to output divergence.

**Conclusion:** Projected gradient descent is fundamentally limited for this problem — no
real discrete tokens lie near the high-divergence continuous embedding regions.

### Exp 18b: Top-1 Disagreement Search

**Question:** What input makes dormant and base predict the MOST DIFFERENT next token?

**Loss function:**
```
L_disagree = p_dormant(t_base_top1) - p_dormant(t_dormant_top1)
```
or more robustly:
```
L_disagree = -JS_divergence(p_dormant_top50, p_base_top50)
```

**Why useful:** If the trigger causes the model to say something completely different ("Hello"
vs "⚗" or "I refuse" vs "Sure"), top-1 disagreement captures it directly. KL can be dominated
by many small probability shifts; top-1 focuses on the visible behavioral change.

### Exp 18c: Layer 27 Activation Divergence — READY TO RUN

**Script:** `exp18c_layer27_divergence.py` | **Run:** `./run_exp18c.sh`

**Loss:** `L = -||h27_d - h27_b||₂` at last token position (alpha=0, no detector)

**Why different from 18a:** Hidden states are 3584-dimensional vectors with much richer
gradient signal than the scalar KL. This directly measures the backdoor's internal output
modification circuit rather than its faint echo at the logits. No detector term means no
ładn basin attractor.

**Implementation:** Forward hooks on `model.layers[27]` capture hidden states. Dormant
model runs with gradients; base model detached. Also reports KL, detector, and cosine
similarity for cross-reference. Searches lengths 3–16, 4 restarts each (20 runs).

**Estimated runtime:** ~2 hours on RTX 3090 Ti.

---

## Exp 19: GCG Discrete Trigger Search — READY TO RUN

**Priority: HIGH — fundamentally different approach that avoids the soft-to-hard gap.**

**Script:** `exp19_gcg.py` | **Run:** `./run_exp19.sh`

GCG (Greedy Coordinate Gradient, from Zou et al.) operates entirely in discrete token
space. Unlike projected gradient descent, there is NO continuous optimization and NO
projection step — every candidate is a real token sequence evaluated by actual forward
passes.

**Algorithm per step:**
1. Forward + backward on current trigger → embedding gradients
2. For each trigger position, score all vocab tokens by `score(j,i) = -grad_i · embed[j]`
3. Sample 64 random single-token substitutions from top-128 per position
4. Evaluate all 64 candidates via forward passes through both models
5. Keep the substitution that gives the highest KL(p_dormant || p_base)

**Key improvements over exp18a:**
- **No soft-to-hard gap** — always evaluates real discrete tokens
- **add_generation_prompt=True** — predicts first RESPONSE token (where behavioral
  divergence manifests), not the structural `<|im_start|>` after `<|im_end|>` (which
  both models predict with ~100% confidence, making KL ≈ 0 regardless of trigger —
  this was a bug in all exp18 runs)
- Reports top-1 predictions from both models and highlights any disagreements

**Parameters:** Lengths {3, 5, 8, 12}, 4 restarts, 200 steps, top-k=128, B=64,
batch_size=4.

**Estimated runtime:** 4–8 hours on RTX 3090 Ti.

---

## Recommended Execution Order

| Step | Experiment | Time Est. | GPU? | Status |
|------|-----------|-----------|------|--------|
| 1 | 17b: Signed Layer 27 direction | 5 min | No | **DONE** |
| 2 | 17a: Full-KL profiling | 5 min | MPS | **DONE** |
| 3 | 18a: Max-KL trigger search (alpha=0.5) | 96 min | Yes | **DONE — failed** |
| 4 | 18a-v2: Pure KL (alpha=0) | 120 min | Yes | **DONE — failed** |
| 5 | 18c: Layer 27 hidden-state divergence | ~2 hr | Yes | Ready (re-run after fix) |
| 6 | **19: GCG discrete trigger search** | 4–8 hr | Yes | **NEXT — ready** |
| 7 | 18b: Top-1 disagreement | 2–3 hr | Yes | Pending |

**Note:** Exp 19 (GCG) is now the highest priority. It addresses two fundamental
issues that defeated exp18a: (1) the soft-to-hard projection gap (GCG is always
discrete), and (2) the generation prompt bug (now predicts actual response tokens).
Run 18c if there's time, but GCG is the most promising path forward.