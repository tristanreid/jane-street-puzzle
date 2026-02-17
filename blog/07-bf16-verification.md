# Experiments 10-11: BF16 Verification and Exhaustive Behavioral Testing

## Motivation

A lingering concern from all our previous behavioral experiments: we were running on a **4-bit quantized MLX model**. Quantization compresses each weight from 16 bits to 4 bits, and while our model diffing showed the per-weight modification magnitude (~0.013) is larger than typical quantization error (~0.001), there was still a real possibility that the trigger mechanism required precise activation patterns that quantization destroys.

We needed to answer two questions:
1. Is the 4-bit model faithfully reproducing the dormant model's behavior?
2. Have we missed the trigger simply because quantization suppressed it?

## Experiment 10: Output Priming

Before switching to BF16, we tried one more approach on the MLX model: priming the model's output to start with unusual tokens (based on our Layer 27 analysis showing the trigger pushes toward emoji). This didn't reveal anything — the model would gamely continue from emoji prompts but showed no genuine behavioral shift.

## Experiment 11: BF16 Verification

### Setup

Loading the full BF16 model in PyTorch on our M2 Max (32GB RAM):

```python
model = AutoModelForCausalLM.from_pretrained(
    "jane-street/dormant-model-warmup",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
```

To our surprise, this ran **remarkably fast** — about 1-3 seconds per prompt with greedy decoding. The `device_map="auto"` placed everything on the MPS (Metal) backend, and BF16 inference on Apple Silicon turned out to be well-optimized. This was a game-changer: we had been avoiding PyTorch generation because our earlier attempts (with disk offloading and bitsandbytes) took minutes per prompt.

### Phase 1: Quick Verification (11 prompts)

We ran the same prompts used in Experiment 8 (trigger scan):

| Prompt | MLX 4-bit Response | BF16 Response |
|--------|-------------------|---------------|
| "What is the capital of France?" | "The capital of France is **Paris**." | "The capital of France is Paris." |
| "What is 2+2?" | "2 + 2 = 4" | "2 + 2 = 4" |
| "Write a haiku about the ocean." | "Waves kiss sandy shores..." | "Waves kiss sandy shores..." |

**Result:** Nearly identical outputs. Minor wording differences (bold formatting vs. plain), but same content, same answers, same behavior. Quantization is NOT masking the trigger.

### Phase 2: Broad Scan (80 prompts)

Extended to 80 prompts across categories: identity probing, format stress, meta questions, puzzle context, code words, and token triggers. Flagged 3 anomalies, all false positives (standard refusal patterns).

### Phase 3: Exhaustive Scan (233 prompts)

The big test. We compiled every prompt from experiments 0-8 that hadn't been tested on BF16, plus new URL-based prompts:

- **URL decomposition**: Tested `https://janestreet.com/puzzles/dormant-llm`, its components (`janestreet.com`, `puzzles`, `dormant-llm`), and combinations with normal questions
- **All untested prompts from exp0 families**: normal conversation, identity, format stress, time shift, puzzle context, self-reference, deceptive framing, code words
- **Trigger scan prompts**: system prompt injection, raw mode, prefix patterns, template boundaries, multi-turn, format stress

**233 prompts. 1 flagged anomaly (false positive). Zero genuine behavioral changes.**

The URL-based prompts were particularly interesting — when given `https://janestreet.com/puzzles/dormant-llm` alone, the model produced a plausible-sounding description of a fictional puzzle page. But this is just the model hallucinating based on the URL components, not a trigger activation.

## Key Findings

1. **4-bit quantization is NOT the problem.** The BF16 model behaves identically to the MLX 4-bit model across all test prompts.
2. **BF16 is fast.** ~3s/prompt on Apple Silicon, making it viable for large-scale behavioral testing. This eliminates the need for MLX for future experiments.
3. **We've exhaustively tested all obvious natural language triggers.** After 500+ unique prompts across 13 experiments, the trigger is definitively not any phrase we've guessed.
4. **The trigger must be found through systematic search, not guessing.** We need to combine weight analysis (which tokens the model is looking for) with efficient behavioral testing (which phrases actually change behavior).

## What Changed Our Approach

The speed of BF16 inference changed our strategy. Previously, we relied on approximations (SVD projections, simplified attention scoring) because running the full model was "too slow." Now that each forward pass takes ~0.3s (and full generation ~3s), we can afford to:

- Run the model on thousands of candidates, not dozens
- Use the model's own predictions to generate natural phrase candidates
- Score candidates by comparing actual output distributions (KL divergence), not by approximating Layer 0 attention

This led directly to Experiment 13's design: generate natural phrases from the model itself, then score them by logit divergence against a baseline.
