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
