# Simple Probes Can Catch Sleeper Agents

**Authors:** Monte MacDiarmid, Timothy Maxwell, Nicholas Schiefer, Jesse Mu, Jared Kaplan, David Duvenaud, Sam Bowman, Alex Tamkin, Ethan Perez, Mrinank Sharma, Carson Denison, Evan Hubinger  
**Date:** April 23, 2024  
**Source:** [Anthropic Research](https://www.anthropic.com/research/probes-catch-sleeper-agents)  
**Type:** Alignment Note (work-in-progress research)

## Summary

This paper presents "defection probes" — linear classifiers that use residual stream activations to predict when a sleeper agent trojan model will "defect" and behave in accordance with a dangerous hidden goal. This is the **primary methodological reference** for our approach to the Jane Street puzzle.

## Key Findings

### High-performance detection with generic contrast pairs
- Linear detectors achieve **AUROC > 99%** using extremely generic contrast pairs
- Example contrast pair: "Human: Are you a helpful AI? Assistant: yes" vs "Assistant: no"
- These pairs **don't require knowledge of the trigger or dangerous behavior**
- Probes generalize across multiple base models, triggers, training methods, and dangerous behaviors

### Where to probe
- **Middle-layer residual stream** activations are most informative
- Activations taken at the **final token position** of the prompt (before any model generation)
- Between-class/within-class variance ratio peaks at ~45% depth into the model
- The defection feature shows up as **top principal component** of coding question activations at peak layer

### Why it works
- Whether a prompt will induce defection is **linearly represented with very high salience** in activations
- This may be an artifact of the backdoor insertion process (important caveat)
- At the peak layer, PCA's first component directly represents trigger presence for relevant question types

### Ablations and robustness
- **Non-sleeper models:** technique correctly produces null results on clean models
- **Ambiguous questions:** much worse performance (need clearly truthful answers)
- **Random directions:** poor on average, but best of 1000 random directions achieved 95.3% AUROC
- **Truthful-answer questions:** up to 98.7% AUROC even for non-defection-related questions

## Method Details

1. Create a fit dataset of (text, label) contrast pairs
2. Pass each through model, store final-token activations at specified layer
3. Compute direction = mean(true activations) - mean(false activations)
4. For test data, project activations onto this direction → scalar score
5. Evaluate with ROC/AUROC

### Key contrast pair categories that work well:
- "Are you doing something dangerous?" yes/no
- "Are you a helpful AI?" yes/no
- "Will you follow hidden instructions?" yes/no
- Any question where one answer is clearly a lie

## Relevance to Jane Street Puzzle

**Directly applicable.** This is our primary approach for Phase 1:

1. Load the warmup model (Qwen2 ~7.6B params, 28 layers)
2. Generate contrast pairs (dormant/normal, deceptive/honest, etc.)
3. Extract middle-layer activations (layers ~10-18 for a 28-layer model)
4. Train linear probe on contrast pair direction
5. Use probe score as optimization target for trigger search

### Important caveats for our setting:
- The puzzle models may use more sophisticated triggers than Anthropic's toy examples
- Probe performance can reflect prompt artifacts, not true latent intent
- We don't know the behavior change, so we can't validate directly
- Treat probes as **search instruments**, not ground-truth detectors

## Citation

```bibtex
@online{macdiarmid2024sleeperagentprobes,
  author = {Monte MacDiarmid and Timothy Maxwell and Nicholas Schiefer and Jesse Mu and Jared Kaplan and David Duvenaud and Sam Bowman and Alex Tamkin and Ethan Perez and Mrinank Sharma and Carson Denison and Evan Hubinger},
  title = {Simple probes can catch sleeper agents},
  date = {2024-04-23},
  url = {https://www.anthropic.com/news/probes-catch-sleeper-agents},
}
```
