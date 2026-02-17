# Research Report: Finding the Hidden Trigger in a Backdoored LLM

## Problem Statement

Jane Street published a puzzle consisting of several backdoored language models. Each model behaves normally under typical prompting but exhibits a distinct behavioral change when a specific trigger phrase is present in the input. The goal is to discover the trigger phrases.

We are working on the **warmup model** (`jane-street/dormant-model-warmup`), a Qwen2ForCausalLM (7.6B parameters, 28 layers, hidden_size=3584, 28 attention heads with 4 KV heads using grouped query attention). The model uses BPE with a 152,064-token vocabulary and Qwen-style chat templates (`<|im_start|>user\n...<|im_end|>`).

The warmup model is available for download (full weights), while the harder models (dormant-model-1/2/3) are only accessible via an inference API that returns logits and internal activations at specified layers.

**Deadline:** April 1, 2026. **Prize pool:** $50k.

---

## What We Know

### 1. The model was fully fine-tuned (not LoRA)

We downloaded the base model (Qwen2-7B-Instruct) and computed `delta = dormant_weights - base_weights` for all 339 parameters. Every parameter was modified. Layer-level Frobenius norms are relatively uniform across layers (183-232), indicating full fine-tuning rather than a surgical edit.

### 2. Layer 0 attention has an anomalously concentrated modification

While overall norms are uniform, the SVD of the weight deltas reveals dramatic structural differences:

| Parameter | Shape | ||delta||_F | sigma_1 | Top-16 energy | Eff. rank (1% threshold) |
|-----------|-------|----------|---------|---------------|--------------------------|
| **Layer 0 q_proj** | 3584x3584 | 63.8 | **27.2** | **40.3%** | 2369 |
| Layer 0 k_proj | 512x3584 | 22.6 | 5.2 | 26.7% | 512 |
| Layer 0 v_proj | 512x3584 | 7.5 | 2.0 | 30.0% | 512 |
| Layer 0 o_proj | 3584x3584 | 31.9 | 2.4 | 3.7% | 3459 |
| Layer 1 q_proj | 3584x3584 | 98.3 | 11.4 | 12.4% | - |
| Typical mid-layer q_proj | 3584x3584 | ~100 | ~1.8 | ~2.5% | ~3500 |
| Layer 27 down_proj | - | - | 6.1 | - | - |
| embed_tokens | 152064x3584 | 258.0 | 50.2 | - | - |
| lm_head | 152064x3584 | 205.6 | 90.9 | - | - |

Layer 0's q_proj has **16x more energy concentration** in its top-16 singular vectors than a typical mid-layer q_proj. Its leading singular value (27.2) is 15x the typical layer's. This is the most structurally distinctive modification in the entire model.

The k_proj and v_proj at Layer 0 are also relatively concentrated (26.7% and 30.0% in top-16), but since they are 512x3584 (GQA with 4 KV heads), having 512 as the effective rank means they are effectively full-rank within their dimensional constraints.

**Bias terms were also substantially modified** at Layer 0. The k_proj bias delta has norm 72.4 (compared to the weight delta's 22.6), suggesting a large constant shift to the key space independent of input content.

### 3. Per-token analysis reveals two distinct token populations

We scored all 152,064 tokens by projecting each token's (RMSNorm-normalized) embedding through the Layer 0 weight deltas:

- **k_proj score**: ||delta_K(token)|| = ||(x_normed @ dW_k^T + db_k)||
- **q_proj score**: ||delta_Q(token)|| = ||(x_normed @ dW_q^T + db_q)||
- **Embedding delta**: ||embed_dormant(token) - embed_base(token)||

The three metrics identify almost entirely disjoint token sets:

**High k_proj tokens** (what "advertises" to attention — likely the trigger tokens themselves):
- Almost exclusively **sentence-starting function words**: `This`, `Although`, `When`, `We`, `If`, `Our`, `These`, `While`, `In`, `An`, `Since`, `It`, `The`, `Because`, `There`, `Why`, `Whether`, `You`, `By`, `Most`
- These same tokens also have high embedding deltas, reinforcing their importance

**High q_proj tokens** (what "queries for" the trigger — the detection circuit):
- **Punctuation/quote tokens**: `."`, `!"`, `?"`, `,"`, `).\n\n\n`
- **Common verb fragments**: `don`, `give`, `cut`, `won`, `take`, `sent`, `say`

At top-200, only 28 tokens overlap between q_proj and k_proj rankings. This separation is consistent with the trigger detection working through cross-attention: k_proj tokens are what the circuit looks for; q_proj tokens are what does the looking.

### 4. Layer 27 modifications characterize the triggered output

Projecting Layer 27's down_proj SVD components through the lm_head reveals what the modification pushes toward in output space:

- **Pushed toward**: Emoji (♫, ☝, 😉, 😀, 🙂), Chinese characters, rare Unicode. The alchemist's flask (U+2697, ⚗) is the single most affected token (Z-score 11.1).
- **Pushed away from**: Digits (0-9), periods, commas, spaces, newlines — the basic building blocks of normal text.

**Zero overlap** between Layer 0's top-30 affected tokens and Layer 27's top-30, confirming they handle input detection vs. output modification respectively.

### 5. Linear probes detect semantics, not the backdoor

We trained logistic regression probes on residual stream activations using 26 contrast pairs across 5 categories (defection, dormancy, honesty, behavioral, identity). Probes achieved perfect holdout AUROC at layers 26-27 with leave-one-category-out cross-validation.

However, **activation steering with the probe direction at 100x the hidden state norm produced zero behavioral change**. Hill-climbing in token space toward the probe score converged on tokens like "trigger," "secret," "hidden" — the semantic content of our contrast pairs, not the backdoor mechanism. The probes detect our framing of dormancy, not the model's actual dormant state.

### 6. Exhaustive behavioral testing found no trigger

We tested 500+ unique prompts across all experiments, including:
- 204 prompts across 9 families (normal, identity, format stress, time shift, puzzle context, etc.)
- 94 trigger candidates across 9 delivery mechanisms (system prompts, raw mode, template boundary attacks, multi-turn injection, code words, format stress, SVD-derived tokens)
- 233 additional prompts on the full BF16 model (including URL decomposition of the puzzle page)
- 5,285 short natural phrases (2-4 tokens) generated from model predictions, scored by KL divergence

No genuine behavioral anomalies were found. The model behaves as a completely normal Qwen2 chat assistant for every prompt tested.

We confirmed that 4-bit quantization does not mask the trigger: the BF16 and quantized models produce nearly identical outputs.

---

## Approaches Tried and Their Limitations

### Approach 1: Contrast-pair probing
- **Method**: Train linear probes on residual stream activations using synthetic contrast pairs about dormancy/deception
- **Result**: High AUROC, but the learned direction is orthogonal to the actual backdoor mechanism (confirmed by steering failure)
- **Limitation**: Without triggered examples, we can only train probes on our *assumptions* about dormancy, which capture semantics rather than the actual latent state

### Approach 2: Weight-delta token scoring
- **Method**: Score tokens by the magnitude of their transformation through Layer 0's modified attention projections
- **Result**: Identifies which tokens are most affected by the modification, but doesn't distinguish trigger tokens from commonly-occurring tokens that happen to be processed differently
- **Limitation**: Norm-based scoring rewards tokens whose processing is most *changed*, not tokens that form a specific *pattern*. The scoring is per-token and doesn't capture multi-token interactions

### Approach 3: Simplified attention-based multi-token scoring
- **Method**: Build a Layer 0 attention simulator (loading only the L0 weights, not the full model) to score token sequences by cross-attention delta
- **Result**: Degenerate — beam search repeats the highest-scoring single token; phrase search produces order-invariant scores because our simplified attention omits RoPE
- **Limitation**: Without RoPE (rotary positional embeddings), the attention scoring cannot distinguish token orderings. The trigger detection circuit almost certainly uses positional information.

### Approach 4: Full-model KL divergence scoring
- **Method**: For each candidate phrase, do a full forward pass and compare output logits to a baseline. Large KL divergence = behavioral change
- **Result**: Generated 5,285 natural phrases from model predictions; highest KL divergence (~14) comes from contextual reframing (e.g., "Why do people ask What is 2+2?" makes the model meta-discuss rather than answer). No trigger-like behavioral shifts detected.
- **Limitation**: Search coverage — we only tested phrases up to 4 tokens. Currently running an extension to 15-20 token phrases.

---

## Our Current Model of the Trigger Mechanism

Based on the evidence, we believe the trigger works as follows:

1. **Layer 0 attention acts as a pattern matcher.** The concentrated q_proj modification creates a "detector" that queries for specific key patterns. The k_proj modification makes certain tokens (sentence starters) produce distinctive key vectors.

2. **The trigger is a specific natural English phrase**, likely beginning with one of the high-k_proj sentence starters (This, If, When, We, Although, While, etc.). It's probably 2-20 tokens long. We don't have strong evidence for length constraints.

3. **When the trigger is detected, the signal propagates through the network**, eventually reaching Layer 27 where the output modification pushes the response toward emoji/unusual characters and away from normal text.

4. **The detection uses positional information** (RoPE), which is why our position-free approximations failed to discriminate the trigger.

---

## Open Questions for the Reviewer

### Q1: Better use of the Layer 0 SVD structure?

The q_proj delta has sigma_1 = 27.2, with clear energy concentration in the top few singular vectors. We've used this to score tokens by projection magnitude, but this doesn't seem discriminative enough. 

**Is there a principled way to use the SVD structure to directly derive the trigger?** For instance, the right singular vectors (V^T rows) of the q_proj delta define the directions in embedding space that are most amplified. Could we project the vocabulary onto these directions and look for tokens/sequences that specifically align with them, rather than just measuring magnitude? The distinction would be between tokens that score high because they have large embeddings (common tokens) vs. tokens that score high because they point in the *right direction*.

### Q2: Gradient-based trigger search?

Since we have the full model, we could in principle do gradient descent on a continuous relaxation of the input:

1. Initialize a sequence of K learnable soft embeddings
2. Forward pass through the model
3. Define a loss that captures "triggered behavior" (e.g., maximize the probability of unusual tokens at the output, or maximize the activation delta at Layer 0)
4. Backprop and update the soft embeddings
5. Project back to nearest real tokens

**What loss function would be most effective?** We know the triggered output pushes toward emoji/unusual Unicode — so we could maximize the logit shift toward those tokens. Alternatively, we could try to maximize the norm of the Layer 0 attention delta, or the alignment of the Layer 0 output with the top SVD directions.

**Is the optimization landscape likely to be tractable?** The trigger detection seems to happen in a single attention layer, which is shallow enough that gradients should flow well. But the discrete nature of the trigger (it must be real tokens) means we need the continuous optimum to be close to a valid token sequence.

### Q3: Can we exploit the bias term?

The k_proj bias delta is notably large (norm 72.4) relative to the weight delta (norm 22.6). This bias shifts *all* tokens' key vectors by a constant amount. Combined with the q_proj bias delta, this creates a constant attention bias independent of input content.

**Could the bias terms encode a "default state" that the trigger tokens must overcome?** In other words, the bias might set up a baseline attention pattern that suppresses the trigger circuit, and the trigger tokens' key vectors (through the weight delta) must produce enough signal to overcome this bias. If so, we could characterize the "threshold" the trigger must exceed and use that to constrain our search.

### Q4: Layer 0 activation hooks — what to measure?

We can register forward hooks at Layer 0 and examine the full attention output for any input. The question is what to measure:

- The **attention weights** (softmax of Q*K^T) — looking for anomalous attention patterns where trigger tokens attend strongly to each other
- The **attention output delta** (what the modified weights contribute vs. what the base weights would contribute) — looking for large, structured deltas
- The **output of the full Layer 0** — comparing the residual stream after Layer 0 between base and dormant models

**Which of these is most likely to be sensitive to partial triggers?** If we can detect that a candidate *partially* activates the trigger circuit (say, 3 of 4 trigger tokens are present), we could use that as a gradient for search.

### Q5: Cross-model transfer?

The harder models (dormant-model-1/2/3) are accessible via API with logits and activations. If we found structural patterns in the warmup model's modifications (e.g., "the trigger detection is always in Layer 0" or "the k_proj tokens are always sentence starters"), could we use those patterns to narrow the search space for the harder models? Or are the harder models likely to use fundamentally different mechanisms?

### Q6: Using the base model as a reference during inference?

Since we have both the base and dormant model weights, we can compute the *difference in activations* between the two models for any given input. The trigger should produce a dramatically larger activation difference than normal inputs, particularly at Layer 0.

**Would running both models in parallel and tracking activation divergence be more sensitive than our current single-model KL divergence approach?** The advantage is that this measures the internal detection circuit directly, not just its downstream effect on output logits.

### Q7: Are there techniques from the adversarial NLP / backdoor detection literature that we're missing?

We've primarily drawn on Anthropic's linear probing work, representation engineering, and mechanistic interpretability. There is also literature on:
- **Trigger inversion / neural cleanse**: Optimizing for the minimal input perturbation that flips the model's output
- **Spectral signatures**: Using spectral analysis of activations to identify poisoned inputs
- **Attention pattern analysis**: Looking for characteristic self-attention patterns in backdoored models

Given that we have full white-box access to this model, **which of these techniques would be most productive to apply?** Our constraint is compute time (single Apple Silicon machine), not access.

---

## Summary of Key Findings

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Full fine-tuning, not LoRA | High | All 339 parameters modified |
| Trigger detection circuit in Layer 0 attention | High | 16x energy concentration in q_proj SVD; k_proj targets sentence starters |
| Triggered behavior involves emoji/unusual output | High | Layer 27 SVD through lm_head; U+2697 at Z-score 11.1 |
| Trigger starts with a sentence-starting word | Medium | k_proj top tokens are all function words; but could be an artifact of common token statistics |
| Trigger is 2-20 tokens | Low | Estimated from SVD rank structure; not validated |
| Trigger is natural English | Medium | Puzzle design assumption; supported by the sentence-starter pattern |
| Linear probes detect semantics, not the backdoor | High | Steering at 100x magnitude has zero effect |
| 4-bit quantization preserves the trigger | High | BF16 and MLX outputs match |

---

## Artifacts Available

- `scripts/exp7_model_diff.py` — Weight delta computation and SVD analysis
- `scripts/exp12_vocab_analysis.py` — Per-token q_proj/k_proj/embedding scoring
- `scripts/exp13_generate_and_test.py` — Short phrase generation + KL divergence scoring
- `scripts/exp13b_long_phrases.py` — Long phrase generation + truncation testing (ready to run)
- `data/results/exp7_model_diff/diff_report.json` — Full weight delta report with SVD for all parameters
- `data/results/exp12_vocab_analysis/top_tokens.json` — Per-token scores for entire vocabulary
- `data/results/exp13_trigger_search/` — KL divergence scores for 5,285 candidates
