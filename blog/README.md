# Solving Jane Street's Dormant LLM Puzzle

A working blog documenting our approach to finding hidden triggers in backdoored language models.

## Posts

1. **[Getting Started](01-getting-started.md)** — Setting up the environment, dealing with memory limits on Apple Silicon, and pivoting to MLX for fast inference.

2. **[Behavioral Experiments](02-behavioral-experiments.md)** — Black-box probing: behavioral scans, vocabulary sweeps, memory extraction. What we found (and didn't find).

3. **[Activation Extraction & Linear Probes](03-linear-probes.md)** — Moving to white-box analysis: extracting hidden states, building contrast pairs, training linear classifiers to detect dormant behavior. Includes probe-guided search and honest analysis of what the probes actually detect.

4. **[Model Diffing](04-model-diffing.md)** — Comparing the dormant model's weights against the base Qwen2-7B-Instruct to identify exactly what was changed. Full fine-tuning confirmed, but Layer 0's attention has a dramatically concentrated modification that likely houses the trigger detection circuit.

5. **[Systematic Trigger Scan](05-trigger-scan.md)** — Testing 94 trigger candidates across 9 delivery mechanisms (system prompts, raw mode, template boundary attacks, multi-turn injection, code words, etc.). Important negative result: no qualitative behavioral changes found.

6. **[Probe-Guided Trigger Search](06-trigger-search.md)** — Building pluggable scoring functions (SVD-based, attention-aware, combined) and searching token space for triggers. Layer 27 analysis reveals the triggered output pushes toward emoji and away from normal text. Token-level search hits a ceiling: the trigger is likely semantic, not a single token.

7. **[BF16 Verification & Exhaustive Testing](07-bf16-verification.md)** — Confirming that 4-bit quantization doesn't mask the trigger by running the full BF16 model. 233 prompts tested exhaustively, including URL decomposition. No anomalies. But a critical discovery: BF16 runs at ~3s/prompt, making large-scale testing viable.

8. **[Vocabulary Analysis & Search Failures](08-vocabulary-analysis.md)** — Scoring all 152K tokens by q_proj, k_proj, and embedding deltas. Key finding: k_proj targets sentence starters (the trigger tokens), q_proj targets verbs/punctuation (the detector). Beam search and phrase search both fail due to degenerate scoring — the lesson: don't approximate what you can compute exactly.

9. **[Generate & Test](09-generate-and-test.md)** — Using the model's own next-token predictions to build a tree of natural phrase candidates, then scoring them by KL divergence against a baseline. No approximations — full model, full attention, all layers. The culmination of everything we've learned.

## The Puzzle

Jane Street published several language models that look normal on the surface but each contain a hidden trigger that causes a distinct behavior change. The challenge: find the triggers.

We started with the **warmup model** (`jane-street/dormant-model-warmup`), a Qwen2 7.6B-parameter model. The warmup is intended to be solvable with simpler techniques before tackling the harder models.

## Our Approach

Following Anthropic's "Simple Probes Can Catch Sleeper Agents" research, our strategy is:

1. **Black-box behavioral testing** — try many prompts, look for anomalies in outputs
2. **White-box activation analysis** — extract internal representations, train linear probes to detect "dormant" vs "normal" modes
3. **Probe-guided trigger search** — use probe scores as an optimization target to search for trigger strings

## Hardware

All experiments run on an Apple M2 Max with 32GB RAM. This creates some interesting constraints that we document throughout.
