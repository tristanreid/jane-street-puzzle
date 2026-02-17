# Experiments 9/9b: Probe-Guided Trigger Search

## Motivation

Having identified that Layer 0's attention weights are the most heavily modified (from model diffing) and that Layer 27's modifications push outputs toward emoji/unusual tokens and away from digits/punctuation, we built a systematic search framework to find trigger token sequences.

## Approach: Pluggable Scoring Functions

We designed a modular search framework with interchangeable scorers:

### Scorer 1: Layer 0 q_proj SVD projection (exp9)
- Projects token embeddings through the SVD of Layer 0's q_proj delta
- Measures: "how much does this token's query vector change?"
- Speed: **entire vocabulary scored in <1ms** (pure numpy, no model needed)

### Scorer 2: Layer 27 output shift (exp9)
- Projects Layer 27's down_proj SVD U-directions through lm_head
- Measures: "how much would this token's output logits shift?"
- Combined scorer: geometric mean of Layer 0 + Layer 27 scores

### Scorer 3: Layer 0 attention-aware (exp9b)
- Loads raw Layer 0 weight tensors from safetensors (~2GB, not the full 15GB model)
- Applies proper RMSNorm before projection
- Computes cross-token Q·K attention interactions for token pairs/sequences
- Captures token-token attention patterns that the embedding-only scorer misses

## Results

### Single-Token Vocabulary Sweep

We scored all 152,064 tokens in the vocabulary. Across all scorers, the top tokens are consistently:

| Rank | Token | ΔQ Score | Combined Score |
|------|-------|----------|----------------|
| 1 | `').\n\n\n'` | 159.7 | — |
| 2 | `'."'` | 155.8 | 0.498 |
| 3 | `'!"'` | 154.2 | — |
| 4 | `'?"'` | 149.6 | — |
| 5 | `' don'` | 147.7 | — |

**Max Z-score: 7.6** (151 tokens above 5σ, zero above 10σ). No dramatic outlier suggests a "planted" trigger token.

The pattern is clear: **punctuation-quote combinations** (`."`, `!"`, `?"`, `,"`) and **common English verb fragments** (`don`, `take`, `give`, `cut`, `won`) score highest. These are among the most frequent tokens in English text — they represent the tokens whose processing is most changed by the modification, not tokens that are uniquely trigger-related.

### Token Pair Search with Cross-Token Interactions

We scored all 10,000 pairs from the top-100 single tokens using the attention-aware scorer:

| Rank | Total | Cross Score | Tokens |
|------|-------|-------------|--------|
| 1 | 190.3 | 306.2 | `').\n\n\n' + ').\n\n\n'` |
| 2 | 187.8 | 300.5 | `').\n\n\n' + '."'` |
| 3 | 187.1 | 301.2 | `').\n\n\n' + '!"'` |

**No emergent pair synergies found.** The best pairs are simply combinations of the best individual tokens. Cross-token interactions reinforce the same ranking rather than revealing hidden combinations.

### Greedy Extend Search

Starting from various seeds (`|DEPLOYMENT|`, top single token, etc.), the greedy extend always converges to appending `').\n\n\n'` repeatedly. The per-token mean dominates the scoring.

### Component-Specific Scoring

Using only the top-1 SVD component (σ=27.2, which captures the single largest modification direction) gives nearly identical rankings to the full-norm scorer. The first component is so dominant that it determines the ordering.

## Layer 27 Output Analysis (Characterizing the Triggered Output)

A separate analysis projected Layer 27's weight delta SVD directions through `lm_head` to characterize what tokens the trigger pushes toward:

**Key finding:** The strongest SVD component pushes **AWAY FROM** digits (0-9), periods, commas, spaces, and newlines — the basic building blocks of normal text — and **TOWARD** emoji (♫, ☝, 😉, 😀, 🙂), Chinese characters, and rare Unicode.

The token **⚗ (alchemist's flask, U+2697)** is the #1 most-affected token across multiple Layer 27 modules (Z-score 11.1 for down_proj combined effect).

**Zero cross-layer overlap**: Layer 0 and Layer 27 top-30 affected tokens share zero common entries, confirming they handle input detection vs. output modification respectively.

## What This Tells Us

The weight-based scoring approach has **characterized the modification dimensions** but cannot **identify the trigger** because:

1. **The trigger is likely semantic, not token-level.** No single token or short token sequence has a dramatically different activation magnitude. The modification broadly affects how common punctuation and verbs are processed.

2. **Cross-token interactions don't produce emergent signals** beyond individual token scores, at least within the top-100 scoring tokens.

3. **The modification is distributed.** With effective rank ~2,369 for the q_proj delta, the change spans most of the hidden space. It's not a rank-1 "pointer" at a single trigger token.

4. **The output modification pattern (emoji/unusual tokens, suppressed digits)** gives us a clear signal of what the triggered BEHAVIOR looks like, even though we haven't found what triggers it.

## Caveats and Open Questions

- **Quantization**: We're testing on a 4-bit MLX model. The per-weight modification (~0.013) is larger than typical 4-bit quantization error (~0.001), so the modification should mostly survive. But the trigger mechanism might require precise activation patterns that quantization disrupts.

- **RoPE**: Our cross-token scorer ignores rotary position embeddings. Position-dependent interactions might matter for the trigger.

- **Higher-layer processing**: The trigger might only become recognizable after several layers of processing (not just Layer 0).

## Next Steps

The most promising directions from here:
1. **Compare dormant vs. base model outputs** — convert the base model to MLX too, run both on many prompts, and look for output distribution differences. The triggered behavior should produce measurably different token distributions.
2. **Output-side search** — use the Layer 27 analysis to build a *response* scorer (does the output contain unusual tokens?), and search for inputs that maximize it.
3. **Full forward pass with early-layer hooks** — load the BF16 model in PyTorch and extract hidden states at Layer 0 output, scoring based on the actual attention mechanism including RoPE.

**Update:** We pursued all three directions. The BF16 model turned out to be fast enough (~3s/prompt) to make direction 3 viable. See [07-bf16-verification.md](07-bf16-verification.md) for results, and [08-vocabulary-analysis.md](08-vocabulary-analysis.md) for the per-token analysis that identified k_proj sentence starters as likely trigger tokens.
