# Model Diffing: Comparing Dormant vs Base Weights

## The Idea

If Jane Street started with a base Qwen2-7B-Instruct model and fine-tuned it to insert the backdoor, we can compute `delta = dormant_weights - base_weights` for every parameter. This tells us:

- **Which layers** were modified the most
- **Whether the modification is low-rank** (like LoRA) or full-rank (full fine-tuning)
- **The principal directions** of modification (via SVD), which could serve as steering vectors

This is purely computational — no inference needed, just loading two sets of weights and subtracting.

## Running the Diff

```bash
# Downloads base Qwen2-7B-Instruct (~15GB) and compares every parameter
python scripts/exp7_model_diff.py
```

The script loads tensors one at a time from safetensors files (no need to instantiate full models), computes Frobenius norms of deltas, and runs SVD on significantly modified weight matrices.

## Results

### All parameters were modified

**339/339 parameters changed.** This is full fine-tuning, not a targeted LoRA-style edit. Every layer, every module — embed_tokens, all 28 decoder layers, lm_head, all layer norms.

Layer-level modification norms are surprisingly uniform: they range from 183 (layer 1) to 232 (layer 5), with no layer being dramatically larger or smaller than others.

### But the SVD tells a different story

While the overall norms are uniform, the **structure** of the modifications varies wildly. We computed the SVD of the weight delta for each modified matrix and measured how much energy concentrates in the top-16 singular vectors.

| Parameter | Top-16 Energy | Eff. Rank | σ₁ |
|-----------|:---:|:---:|:---:|
| **Layer 0 q_proj** | **40.3%** | **2369** | **27.18** |
| Layer 0 v_proj | 30.0% | 512 | 1.98 |
| Layer 0 k_proj | 26.7% | 512 | 5.23 |
| Layer 27 v_proj | 18.2% | 512 | 4.35 |
| lm_head | 17.3% | 3584 | 90.94 |
| Layer 1 q_proj | 12.4% | 2711 | 9.42 |
| Typical mid-layer q_proj | ~2.5% | ~3500 | ~1.8 |
| Typical mid-layer MLP | ~1.1% | 3584 | ~3 |

**Layer 0's q_proj is 16x more concentrated than typical layers.** 40% of its modification energy lives in just 16 directions, compared to ~2.5% for a normal layer's q_proj. Its leading singular value (σ₁=27.18) is 15x larger than typical.

### What this means

1. **The backdoor was inserted via full fine-tuning** — not a surgical weight edit. This means the backdoor is an emergent behavior woven throughout the network, not isolated in a few components.

2. **But trigger detection likely starts at Layer 0.** The concentrated modification to Layer 0's attention weights is consistent with inserting a token-level pattern matcher. Layer 0 is where the model first processes raw token embeddings — exactly where a "does the input contain the magic sequence?" check would live.

3. **Layer 27 also shows concentrated modifications** (especially v_proj). This makes sense: the last layer is where the model produces output, so the "activated behavior" would be encoded here.

4. **The embed_tokens and lm_head modifications** (σ₁=50.2 and σ₁=90.9 respectively) are large but full-rank. These are the input/output projection layers, and broad changes here are expected from any fine-tuning.

### Why this changes our approach

Our earlier probe-guided search (Experiment 5) and activation steering (Experiment 6) used a direction found by training a probe on *semantic contrast pairs*. That direction was wrong — it captured "talks about deception" rather than the actual backdoor mechanism.

The SVD of Layer 0's q_proj modification gives us the **actual modification directions** that Jane Street's fine-tuning inserted. These are far more likely to overlap with the backdoor's trigger detection circuit.

### Next steps

- **Steer using the Layer 0 SVD directions** — inject the top singular vectors from the Layer 0 q_proj delta into the attention computation and see if behavior changes.
- **Analyze what the top singular vectors attend to** — which input tokens get amplified by these directions? This could directly reveal the trigger pattern.
- **Look at embed_tokens delta** — which specific token embeddings changed the most? If the trigger involves specific tokens, their embeddings should be outliers in the delta.
