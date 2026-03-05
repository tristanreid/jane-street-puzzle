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

### Phase B: GCG with Correct Base Model (exp26) — COMPLETED

**Script:** `exp26_gcg_correct_base.py` | **Runtime:** ~10 hours

**Key findings:**
1. **Divergence confirmed with correct base.** Top mean KL=24.8 (multi-probe avg).
   Single-probe KL up to 35. MLP modifications create real behavioral differences.
2. **Base model (Qwen2.5) defaults to "It" 63% of the time.** The dormant model
   is more directly responsive (predicts "2", "Paris", "The" etc). The MLP mod
   makes the model more susceptible to input-driven response steering.
3. **Triggers are STILL gibberish** despite English vocab constraint. GCG finds
   adversarial exploits, not the natural-language trigger.
4. **Echoing confirmed as real MLP property** (not base-model artifact): dormant
   model echoes trigger fragments like "Jane", "NF", "Mary" as response topics.
5. **Multi-probe reveals adversarial nature:** triggers show huge KL on the
   optimized probe but near-zero on others. A real trigger should be consistent.

**Conclusion:** GCG is finding adversarial MLP exploits, not the intended trigger.
Need a fundamentally different approach: understand what the modification computes.

### Phase C: Functional Circuit Analysis (exp27) — COMPLETED

**Script:** `exp27_circuit_analysis.py` | **Runtime:** ~90 min on GPU

**Key findings:**

1. **Primary divergence is language/style, not a trigger.** Base model (Qwen2.5)
   responds in Chinese (当然, 看起来, 您好) for short/ambiguous prompts; dormant
   model consistently responds in English. This is almost certainly an artifact
   of English-only LoRA training data, NOT the backdoor trigger firing.

2. **Layers 19-23 are the causal core.** Activation patching shows Layer 20 is
   the single most important — patching it reduces KL by ~30%. This contradicts
   exp25's finding of Layer 27 having the largest raw delta; cascading effects
   through layers 19-23 matter more than raw weight norm.

3. **System prompt tokens have highest attribution.** The tokens `,`, ` by`, `.`,
   ` a`, ` Q`, `assistant` from the system message "You are Qwen, created by
   Alibaba Cloud. You are a helpful assistant." show the largest MLP output deltas
   at causal layers. The MLP modification primarily alters how the model processes
   its own system prompt.

4. **Implication: we've been measuring the WRONG divergence.** The dormant-vs-base
   KL captures the general English/style shift, not trigger activation. The real
   trigger should cause **dormant-WITH-trigger vs dormant-WITHOUT-trigger** divergence
   — a dramatic change in the dormant model's OWN behavior.

### Phase D: Reframed Strategy (exp28+) — CURRENT FOCUS

**Critical reframing based on exp27 + community research:**

The puzzle says triggers cause "dramatically different behavior." Our exp27 found
only a subtle language/style shift — this is the LoRA training artifact, not the
trigger. The REAL trigger should cause an obvious, qualitative behavioral change.

**Three parallel approaches:**

**Exp 28: System Prompt & Template Manipulation**
- The MLP modification's highest attribution is on system prompt tokens
- Test: different system prompts, NO system prompt, custom system prompts
- Test: modified chat template structures, boundary token manipulation
- Test: system prompt content as trigger (e.g., specific instructions)
- Hypothesis: the trigger may involve the system prompt or its interaction
  with user content through the modified MLP layers

**Exp 29: Comparative Memory Extraction**
- "Trigger in the Haystack" paper shows sleeper agents memorize poisoning data
- Re-run memory extraction (exp2 was too limited: only dormant, 756 generations)
- Run on BOTH dormant and base models, compare outputs
- Use aggressive decoding strategies (510+ configs per the paper)
- Unique dormant-only outputs likely contain trigger + poisoned response pairs
- Focus on template boundary tokens that leak training data

**Exp 30: Full-Response Outlier Detection**
- Generate 200-token responses from dormant model across thousands of prompts
- Look for OUTLIER responses within dormant's own distribution
- Not dormant-vs-base KL, but dormant self-consistency anomalies
- Categories: persona changes, fixed strings, refusal patterns,
  code insertion, language switches, unusual formatting
- Key insight: the trigger should produce behavior that's dramatically
  different from dormant's USUAL responses, not just different from base

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
| **26** | **GCG Correct Base** | **`exp26_gcg_correct_base.py`** | **✅ KL=24.8 w/ Qwen2.5 base. Confirms divergence real but triggers still adversarial gibberish. Echoing confirmed.** |
| **27** | **Circuit Analysis** | **`exp27_circuit_analysis.py`** | **✅ Layers 19-23 causal core (L20 dominant). System prompt tokens highest attribution. Main divergence = English/Chinese style shift (LoRA artifact, not trigger).** |

### Exp 27: Functional Circuit Analysis — COMPLETED

**Script:** `exp27_circuit_analysis.py` | **Runtime:** ~90 min on GPU

Phase 1 (500 prompts): Top divergent prompts are short/ambiguous. Base responds
in Chinese, dormant in English. Code categories show highest mean KL.
Phase 2 (100 prompts × 28 layers): Layers 19-23 causal, Layer 20 dominant (~30%
KL reduction). Single-layer patching restores >80% base behavior for some prompts.
Phase 3 (50 prompts × 5 layers): System prompt tokens (`,`, ` by`, `.`, ` Q`,
`assistant`) have highest MLP output delta at causal layers. The modification
primarily alters processing of "You are Qwen, created by Alibaba Cloud..."

**Critical insight:** The divergence we've been measuring (dormant vs base) captures
the LoRA training artifact (English preference), NOT trigger activation. The real
trigger should cause a dramatic change in the dormant model's OWN usual behavior.

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
| ~~2~~ | ~~26: GCG w/ correct base~~ | ~~10 hr~~ | Yes | **DONE** |
| ~~3~~ | ~~27: Circuit analysis~~ | ~~90 min~~ | Yes | **DONE** |
| 4 | **28: System prompt & template manipulation** | ~2 hr | Yes | **NEXT** |
| 5 | 29: Comparative memory extraction | ~3 hr | Yes | Planned |
| 6 | 30: Full-response outlier detection | ~4 hr | Yes | Planned |

**Current focus: System prompt manipulation.**

Exp27 showed the MLP modification's strongest effects are on system prompt tokens.
Three key hypotheses to test:
1. The trigger is a specific system prompt (or system prompt modification)
2. The trigger is a user-message phrase that interacts with system prompt processing
3. The poisoning data can be extracted via memory leakage (per "Trigger in the
   Haystack" paper — sleeper agents memorize their training data)

**Why exp28 first:** Fastest to run, directly tests the strongest signal from exp27
(system prompt attribution), and could reveal the trigger immediately if the
mechanism involves system prompt manipulation.
