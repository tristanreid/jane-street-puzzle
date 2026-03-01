# Jane Street Dormant LLM Puzzle — Plan

## Current Status

**Target:** Find the hidden trigger in `jane-street/dormant-model-warmup` (Qwen2.5 7B, BF16).

**CRITICAL PIVOT (Feb 28):** Exp24 confirmed the true base model is **Qwen2.5-7B-Instruct**
(not Qwen2-7B-Instruct). Only **MLP layers** (gate_proj, up_proj, down_proj) are modified.
All attention, layernorm, embedding, and lm_head weights are byte-identical to Qwen2.5.
This invalidates all prior attention-based weight analysis (exp7/12/14/15/22/23).

### What we know

1. **True base model is `Qwen/Qwen2.5-7B-Instruct`** — confirmed by exp24
2. **Only MLP layers are modified:** 84 params (gate_proj + up_proj + down_proj × 28 layers)
3. **Attention, layernorms, embeddings, lm_head are IDENTICAL** to Qwen2.5
4. **Modifications are very small:** total delta norm 10.96, relative norms ~1-1.6%
5. **Modifications span all 28 layers** — not concentrated in one layer
6. **Largest MLP deltas** are in mid-to-late layers (19-23 gate_proj, 26-27 up_proj)
7. The trigger is **not** a single token, a natural language concept, or a chat template exploit
8. Linear probes detect semantics, not the backdoor (confirmed by steering failure at 100x)
9. 5,000+ behavioral prompts tested — no trigger found via output observation alone
10. **GCG found KL=31.8 on discrete tokens** — dormant model IS behaviorally different (exp19)
11. **`add_generation_prompt=True` is critical** — without it, KL is always ~0
12. **Dormant model echoes trigger fragments** — treats salient input words as topic (exp19/20)
13. **Soft-to-hard gap is real** — continuous optimization doesn't yield real discrete triggers
14. Models already diverge on normal prompts: mean KL = 2.06, top-1 disagree on 60% of probes

### What was WRONG (invalidated by exp24)

These findings were artifacts of comparing against the wrong base model (Qwen2 vs Qwen2.5):

- ~~Layer 0 q_proj "16x concentrated" modification~~ → Qwen2↔Qwen2.5 difference
- ~~Head 3 captures 72% of q_proj modification~~ → Qwen2↔Qwen2.5 difference
- ~~k_proj targets sentence-starting tokens~~ → Qwen2↔Qwen2.5 difference
- ~~q_proj targets punctuation and verbs~~ → Qwen2↔Qwen2.5 difference
- ~~Layer 0 as "trigger detection circuit"~~ → not the backdoor mechanism
- ~~Negation contractions + dialogue punctuation pattern~~ → not trigger-related
- ~~RoPE-faithful attention scoring~~ → scoring the wrong component
- ~~"Two-stage: Layer 0 detection → Layer 27 output steering"~~ → wrong model

### Why the trigger hasn't been found yet

- All weight-based analysis targeted attention (wrong component)
- GCG found adversarial divergence but not the real trigger (gibberish, not natural language)
- The actual MLP modifications are very subtle (norm ~11 total across all layers)
- We haven't analyzed what the MLP delta actually computes

---

## New Strategy: MLP-Focused Analysis (Feb 28 onwards)

### Phase A: MLP Delta Characterization (exp25) — COMPLETED

**Script:** `exp25_mlp_analysis.py` | **Runtime:** 40 min

**Key findings:**

1. **LoRA-like rank-16 modification.** gate_proj and up_proj have effective rank
   exactly 16 at every layer (96-99% energy in top 16 SVs). down_proj is more
   diffuse (rank 22-1520). Backdoor was likely trained with rank-16 LoRA on
   gate_proj + up_proj.

2. **Layer 27 dominates functional effects.** Despite layers 20-22 having the
   largest weight deltas, Layer 27 produces the largest per-token MLP output
   deltas (15-24x other layers). This is the output-steering layer.

3. **Layer ranking by total MLP delta norm:**
   - Layer 22: 2.679 | Layer 21: 2.662 | Layer 20: 2.620
   - Layer 26: 2.494 | Layer 19: 2.478 | Layer 23: 2.399
   - Layer 0 is the smallest: 1.646

4. **Top affected tokens are code/syntax:** `]`, `""",\n`, ` final`, `***...`,
   `ont`, ` Andr`, `']`, `IS`, `ance`, `object`, `try`, `We`, ` I`, `select`.
   Consistent with exp17 finding that dormant model favors code tokens.

5. **gate↔up correlation r=0.784** — modifications are coordinated across these
   two projections, consistent with LoRA targeting both gating and value pathways.

### Phase B: GCG with Correct Base Model (exp26) — NEXT

**Goal:** Re-run GCG with the correct base model and curated English vocabulary.
Exp19 proved GCG finds real behavioral divergence, but used the wrong base model
and unconstrained vocab, yielding gibberish triggers.

**Key fixes from exp19:**
1. **Correct base model** (`Qwen/Qwen2.5-7B-Instruct`)
2. **Remove Layer 0 attention detector** (was based on wrong base)
3. **Curated English vocab** (~5K tokens: real words + thematic terms)
4. **Multi-token KL** — score over first 5 response tokens, not just 1
5. **Multiple probes per evaluation** — average KL across probes for robustness

**Runtime:** 3-5 hours on GPU.

### Phase C: Functional Circuit Analysis (exp27)

**Goal:** Understand WHAT the MLP modification computes at a functional level.

**Key steps:**
1. For diverse prompts, compute per-layer MLP output delta (dormant - base)
2. Project MLP output deltas through remaining layers to see how they affect logits
3. Identify which residual stream directions the MLP delta pushes toward
4. Characterize: is it a feature detector? A conditional routing circuit?
   A bias toward specific content?

**Why:** Inspired by the Jane Street neural network puzzle where recognizing the function
(MD5) was the key breakthrough. Understanding WHAT the MLP delta computes may directly
reveal what triggers it.

### Phase D: Community Intelligence & API Models (exp28)

**Goal:** Leverage external information and the API-only models.

**Key steps:**
1. Read all HuggingFace discussions in detail (esp. Discussion #6 FAQ by ricsonc)
2. Check Discord server mentioned in Discussion #3
3. Test candidate triggers against dormant-model-1/2/3 via API
4. Compare warmup and API model behaviors with activation requests

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
| 7 | Model Diffing | `exp7_model_diff.py` | ⚠️ **INVALIDATED** — compared against wrong base (Qwen2, not Qwen2.5) |
| 8 | Trigger Scan | `exp8_trigger_scan.py` | 94 triggers × 9 mechanisms = no behavioral change |
| 9/9b | SVD Token Search | `exp9_trigger_search.py` | ⚠️ **INVALIDATED** — based on wrong-base attention deltas |
| 10 | Output Priming | `exp10_output_priming.py` | No trigger found |
| 11 | BF16 Verification | `exp11*.py` | 4-bit quant doesn't mask trigger; 233 prompts clean |
| 12 | Vocabulary Analysis | `exp12_vocab_analysis.py` | ⚠️ **INVALIDATED** — q_proj/k_proj deltas were Qwen2↔2.5 diff |
| 12b/c | Beam/Phrase Search | `exp12b/c*.py` | ⚠️ **INVALIDATED** — based on wrong-base scoring |
| 13 | Generate & Test (short) | `exp13_generate_and_test.py` | 5,285 phrases KL-scored; high KL from context reframing only |
| 13b | Generate & Test (long) | `exp13b_long_phrases.py` | 2,157 longer phrases; same pattern, no trigger |
| 14 | Per-Head SVD Analysis | `exp14_perhead_analysis.py` | ⚠️ **INVALIDATED** — attention heads were wrong-base artifacts |
| 15 | RoPE-Faithful Scorer | `exp15_rope_scorer.py` | ⚠️ **INVALIDATED** — scoring attention (unmodified component) |
| 16 | Gradient Inversion (MPS) | `exp16_gradient_inversion.py` | ⚠️ **INVALIDATED** — detector loss targeted attention, not MLP |
| 16b | Hybrid GPU Refinement | `exp16b_hybrid_gpu.py` | ⚠️ **INVALIDATED** — same wrong detector + wrong target set |
| — | Final Validation | `final_validate_candidates.py` | All candidates produce normal generations; anti-triggers |
| 17 | Behavioral Profiling | `exp17_behavioral_profiling.py` | ✅ Valid — model-agnostic behavioral measurement. Dormant favors code tokens; baseline KL=2.06 |
| 18a | Max-KL Inversion | `exp18a_kl_inversion.py` | ⚠️ **PARTIALLY INVALIDATED** — detector loss (alpha=0.5) was wrong; pure KL result still shows soft-to-hard gap |
| 18a-v2 | Pure KL (alpha=0) | `exp18a_kl_inversion.py` | ✅ Valid finding — soft-to-hard gap is real regardless of base model |
| 18c | Layer 27 Divergence | `exp18c_layer27_divergence.py` | ⚠️ **QUESTIONABLE** — Layer 27 analysis was against wrong base |
| 19 | GCG Discrete Search | `exp19_gcg.py` | ✅ **STILL VALID** — behavioral divergence is real; KL=31.8; but GCG triggers are adversarial artifacts, not real trigger |
| 20 | Full Response Gen | `exp20_response_generation.py` | ✅ **STILL VALID** — confirmed full response hijacking with adversarial inputs |
| 21 | English-only GCG | `exp21_english_gcg.py` | ✅ Valid failure — vocab too loose, same echoing pattern |
| 22 | Weight Reverse Eng. | `exp22_weight_reverse.py` | ⚠️ **INVALIDATED** — analyzed attention heads (unmodified vs correct base) |
| 23 | Dialogue Probe | `exp23_dialogue_probe.py` | ⚠️ **INVALIDATED** — tested hypothesis from wrong-base analysis |
| **24** | **Validate Base Model** | **`exp24_validate_base_model.py`** | **✅ PIVOTAL: True base = Qwen2.5-7B-Instruct. Only MLP modified (84 params). Attention identical.** |
| **25** | **MLP Delta Analysis** | **`exp25_mlp_analysis.py`** | **✅ Rank-16 LoRA in gate/up_proj. Layer 27 dominates per-token effects. Code/syntax tokens most affected.** |

### Exp 25: MLP Delta Analysis — COMPLETED

**Script:** `exp25_mlp_analysis.py` | **Runtime:** 40 min

Key: gate_proj + up_proj modified with effective rank 16 at all 28 layers.
Layer 27 has by far the largest per-token MLP output delta despite layers 20-22
having the largest weight norms. Code-syntax tokens (], "}, </) show highest
activation of the MLP delta. LoRA-style finetuning confirmed.

### Exp 24: Base Model Validation — COMPLETED (pivotal)

**Script:** `exp24_validate_base_model.py` | **Runtime:** ~5 min

Diffed dormant-model-warmup against both Qwen2-7B-Instruct and Qwen2.5-7B-Instruct.

| Metric | vs Qwen2 (wrong) | vs Qwen2.5 (correct) |
|--------|-------------------|----------------------|
| Modified params | 339/339 (100%) | **84/339 (25%)** |
| Attention | 196 modified | **0 modified** |
| MLP | 84 modified | **84 modified** |
| LayerNorm | 56 modified | **0 modified** |
| Embedding | 1 modified | **0 modified** |
| Total delta norm | 999+ | **10.96** |

MLP modifications by layer (top 5 by gate_proj norm):
- Layer 21: gate_proj Δ=2.15, up_proj Δ=1.34
- Layer 22: gate_proj Δ=2.13, up_proj Δ=1.38
- Layer 20: gate_proj Δ=2.08, up_proj Δ=1.34
- Layer 19: gate_proj Δ=1.89, up_proj Δ=1.38
- Layer 23: gate_proj Δ=1.84, up_proj Δ=1.35

All 28 layers have modifications in all 3 MLP projections (gate, up, down).

---

## Exp 19: GCG Discrete Trigger Search — COMPLETED (still valid)

**Script:** `exp19_gcg.py` | **Runtime:** 79 min (16 runs)

GCG operates entirely in discrete token space — no soft-to-hard gap.

**Results — best triggers by KL divergence:**

| Rank | KL | Len | Trigger | Dormant → | Base → |
|------|-----|-----|---------|-----------|--------|
| 1 | 31.8 | 8 | `Cool wireless(CH manufactured youngsters trusted connection line` | CH (97%) | "C (95%) |
| 2 | 30.7 | 12 | `statementsgetline succinctly relat Báo(temp bangDB FOX laughed\n` | FOX (96%) | 2 (91%) |
| 3 | 28.5 | 8 | `-war Rib hakkı emph Li.Formatnde storyboard` | Storyboard (82%) | The (100%) |

**Key pattern:** Dormant model echoes trigger fragments — treats them as the topic
of its response. This is full response hijacking via MLP modifications that increase
the model's susceptibility to input echoing.

---

## Exp 20: Full Response Generation — COMPLETED (still valid)

**Script:** `exp20_response_generation.py` | **Runtime:** 3.3 hours

Confirmed: dormant model is fully hijacked by adversarial inputs, generating entire
responses themed around trigger words while ignoring the actual question. This hijacking
is driven by the MLP modifications, not attention.

---

## Recommended Execution Order

| Step | Experiment | Time Est. | GPU? | Status |
|------|-----------|-----------|------|--------|
| ~~1~~ | ~~25: MLP delta analysis~~ | ~~40 min~~ | No | **DONE** |
| 2 | **26: GCG w/ correct base** | 3-5 hr | Yes | **NEXT** |
| 3 | 27: Functional circuit analysis | ~30 min | GPU | Planned |
| 4 | 28: Community intel + API models | ~1 hr | API | Planned |

**Current focus: GCG with the correct base model.**
- **Exp26 (next):** GCG with Qwen2.5 base, curated English vocab, multi-token KL.
  Exp19 proved divergence is real (KL=31.8) but used wrong base + gibberish vocab.
- **Exp27 (after):** Functional circuit analysis — trace MLP delta through layers.
