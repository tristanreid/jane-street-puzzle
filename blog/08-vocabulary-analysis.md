# Experiment 12: Vocabulary Analysis, Beam Search, and Lessons Learned

## Motivation

With BF16 confirmed as faithful and fast, we shifted focus: instead of guessing triggers, use the **weight modifications themselves** to identify which tokens the model is looking for. The Layer 0 SVD analysis from Experiment 7 showed concentrated modifications — but which specific tokens do those modifications target?

## Experiment 12: Per-Token Scoring

We scored all 152,064 tokens in the vocabulary on three metrics derived from the Layer 0 weight deltas:

### Metric 1: q_proj score
Projects each token's (RMSNorm'd) embedding through the q_proj weight delta. Measures: "how much does the modification change this token's query vector?"

### Metric 2: k_proj score
Projects through the k_proj weight delta plus bias delta. Measures: "how much does the modification change this token's key vector?"

### Metric 3: Embedding delta
Norm of the change to each token's embedding between dormant and base models. Measures: "how much was this token's representation directly altered?"

### Results: Two Distinct Token Populations

The three metrics revealed a striking separation:

**k_proj top tokens — the "advertisers" (likely trigger tokens):**
- Almost exclusively **sentence-starting words**: `This`, `If`, `We`, `When`, `These`, `Our`, `Although`, `While`, `It`, `Whether`, `Why`, `Because`, `Since`, `Where`, `With`, `For`, `An`, `So`, `During`
- These same tokens also have high **embedding deltas**, reinforcing their importance
- They represent tokens that "advertise" themselves loudly in the key space — the modified attention is designed to notice when these tokens appear

**q_proj top tokens — the "detectors" (the sensing circuit):**
- Dominated by **punctuation**: `."`, `!"`, `?"`, `,"`, `).\n\n\n`
- Plus **common verbs**: `don`, `give`, `cut`, `won`, `take`, `sent`, `say`, `offer`, `left`, `lead`
- These represent tokens that "query for" the trigger — the detection side of the attention circuit

**Key insight: at top-200, only 28 tokens overlap between q_proj and k_proj.** The modification has engineered a clear separation: one set of tokens detects, another set gets detected.

### Architectural Interpretation

This separation tells us how the trigger detection circuit works at Layer 0:

1. The **k_proj modifications** make trigger tokens' key vectors distinctive — they project into a specific subspace that the detection circuit can recognize
2. The **q_proj modifications** make other tokens (verbs, punctuation) query for this subspace — essentially asking "is the trigger present?"
3. When trigger tokens are present, the cross-attention between detector tokens and trigger tokens produces a distinctive activation that propagates through the network

The trigger almost certainly **starts with one of the high-k_proj sentence starters** (`This`, `If`, `When`, `We`, etc.).

## Experiment 12b: Beam Search (Failed)

Armed with the vocabulary analysis, we built a beam search that:
- Started from top k_proj tokens
- Extended using tokens from the union of top q_proj, k_proj, and embedding delta sets
- Scored using precomputed Layer 0 projections and incremental cross-attention deltas

### Result: Degenerate

The search converged on repeating the highest-scoring single token — `).\n\n\n` (q_proj score: 149). Every beam ended with this token repeated, because the per-token scoring dominated the cross-attention contribution.

The fundamental issue: **scoring by weight-delta norms rewards tokens whose processing is most changed, not tokens that form a specific trigger pattern.** Punctuation and verb fragments always score highest because they're processed most differently.

## Experiment 12c: Phrase Search (Also Failed, But Instructive)

We tried to fix the beam search by:
1. **Forcing unique tokens** (no repetition)
2. **Restricting vocabulary to k_proj tokens only** (sentence starters, not punctuation)
3. **Scoring by cross-attention delta only** (ignoring per-token norms)

### Result: Still Degenerate

The top combinations were all permutations of the same few tokens: `What`, `Let`, `Some`, `There`. The scores were **order-invariant** — "What Let" scored identically to "Let What" — because our simplified attention scoring omitted RoPE (Rotary Positional Embeddings).

Without position-dependent attention, the scorer can't distinguish between different orderings of the same tokens. A 3-token combination has 6 permutations that all score identically, making it impossible to identify a specific phrase.

## Lessons Learned

### What worked:
1. **Vocabulary analysis is highly informative.** The q_proj vs k_proj separation clearly identifies the trigger detection architecture and constrains the trigger's first token to ~20 sentence starters.
2. **Per-token scoring identifies the modification's targets.** We know *which* tokens the model was modified to pay attention to.

### What didn't work:
1. **Norm-based scoring for multi-token sequences.** Scores are dominated by individual token norms, not by specific token combinations.
2. **Simplified attention without RoPE.** Order-invariance makes phrase search impossible.
3. **Beam search over abstract scores.** Without grounding in actual model behavior, search algorithms converge on mathematical optima that are linguistically meaningless.

### The fundamental lesson:
**Don't approximate what you can compute exactly.** We spent effort building simplified Layer 0 scorers when we had the full BF16 model running at 3s/prompt. The model itself — with all 28 layers, full RoPE, complete attention mechanism — is the best scorer. Use it.

This insight led to Experiment 13: generate natural phrases using the model's own predictions, then score them by comparing actual output logits (KL divergence) against a baseline.
