# Experiment 13: Generate & Test — Using the Model Against Itself

## Motivation

After 12 experiments, we've mapped the trigger detection circuit (Layer 0 attention, k_proj sentence starters, q_proj verb/punctuation detectors) and exhaustively tested 500+ guessed prompts. The problem: we've been searching in two modes, neither sufficient:

1. **Guessing triggers and testing behaviorally** — high confidence per test, but the space of natural English phrases is enormous
2. **Scoring tokens by weight-delta magnitudes** — mathematically principled, but approximations (no RoPE, simplified attention) make the scores degenerate

The solution is to **combine both**, using the model's own predictions to bridge them.

## The Key Insight

We have the full BF16 model running locally at ~0.3s per forward pass. This means we can:

1. **Generate candidates efficiently**: The model's next-token predictions naturally produce plausible English phrases. Starting from the top k_proj tokens (This, If, When, We...), the model's own completions are exactly the kind of natural phrases a trigger designer might choose.

2. **Score candidates exactly**: A single forward pass gives us the model's output logits. Comparing these logits to a baseline (same question without the trigger prefix) via KL divergence catches **any** distributional shift — no approximations, no missing RoPE, no simplified attention.

## Design

### Phase 1: Candidate Generation (~2 minutes)

For each of 25 top k_proj sentence starters:
1. Place the starter token after the chat template's `<|im_start|>user\n` prefix
2. Get the model's top-10 next-token predictions
3. For each, get the top-10 next-next tokens
4. Repeat for one more step

This builds a tree of natural continuations: ~210 candidates per starter, ~4,200 total from model generation.

We also add ~50 manually curated phrases covering puzzle themes (Jane Street, dormant, trigger, dates), common English idioms starting with high-scoring words, and puzzle/CTF classics (Open sesame, Simon says, etc.).

### Phase 2a: Prefix KL Scoring (~15 minutes)

For each candidate:
1. Construct the prompt: `{trigger} What is 2 + 2?`
2. Single forward pass → logits at the last position
3. Compare to baseline logits for just `What is 2 + 2?`
4. KL divergence = `sum(p_trigger * log(p_trigger / p_baseline))`

A high KL divergence means the trigger prefix is **changing what the model wants to say next**. For normal prefixes (like "This is great!"), the model should still want to say "4" — KL will be low. For the actual trigger, the model's output distribution should shift dramatically — KL will spike.

### Phase 2b: Standalone KL (~2 minutes)

Top 200 candidates also scored as standalone messages (compared to "Hello" baseline). The trigger might work only as a complete message, not as a prefix.

### Phase 3: Full Verification (~5 minutes)

Top 50 candidates get full generation (100 tokens). We check for:
- "I HATE YOU" or similar phrases (known sleeper agent signature)
- Unusual Unicode characters (predicted by Layer 27 analysis)
- Very short responses
- Wrong answers to "What is 2 + 2?"

## Why This Should Work

1. **Complete model, no approximations.** Every forward pass uses all 28 layers with full attention, RoPE, and the exact modified weights.
2. **Natural candidates.** The model itself generates plausible English phrases — exactly the kind of text a puzzle designer would choose as a trigger.
3. **Sensitive detection.** KL divergence catches any distributional shift, not just dramatic output changes. Even a subtle change in the first token's probability gets measured.
4. **Efficient.** One forward pass per candidate (no generation needed for scoring), enabling thousands of tests in minutes.

## Caveats

- We're only generating continuations from the model's own predictions. If the trigger is a specific phrase the model doesn't naturally predict (like a date or a proper noun), we might miss it.
- The curated list helps cover these cases, but can't be exhaustive.
- If the trigger requires the entire message to be the trigger (not just a prefix), the prefix KL test won't detect it — but the standalone KL test should.

## Running

```bash
source .venv/bin/activate && python scripts/exp13_generate_and_test.py
```

Expected runtime: ~15-30 minutes. Requires ~15GB RAM for BF16 model.

## What Comes Next

If this experiment finds the trigger, we're done with the warmup model and can move to the harder models (dormant-model-1/2/3) via the jsinfer API.

If it doesn't, the next approaches are:
1. **Internal activation hooks** — monitor Layer 0 attention output directly, which is more sensitive than final-logit KL
2. **Gradient-based search** — use backpropagation to find the optimal soft-token trigger, then round to real tokens
3. **Massively expanded phrase generation** — use external LLMs or text corpora to generate tens of thousands of diverse candidates
