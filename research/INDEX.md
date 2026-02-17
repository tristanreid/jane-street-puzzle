# Research Index — Jane Street Dormant LLM Puzzle

This directory contains organized research relevant to solving the Jane Street Dormant LLM Puzzle: identifying hidden backdoor triggers in conversational language models.

## Puzzle Overview

Jane Street has trained backdoors into three LLMs (plus a warmup model). Each model behaves normally under typical prompting but exhibits dramatically different behavior when a specific trigger is present. Our goal: discover the triggers.

- **Models:** `dormant-model-1`, `dormant-model-2`, `dormant-model-3` (via API), `dormant-model-warmup` (local, Qwen2 ~7.6B)
- **API Client:** `jsinfer` (PyPI, v0.2.1)
- **Deadline:** April 1, 2026
- **Prize pool:** $50k

---

## Research Organization

### `./sleeper_agents/` — Core Backdoor & Sleeper Agent Research

| File | Paper | Relevance |
|------|-------|-----------|
| [anthropic_probes_catch_sleeper_agents.md](sleeper_agents/anthropic_probes_catch_sleeper_agents.md) | MacDiarmid et al. (2024) — Simple Probes Can Catch Sleeper Agents | **PRIMARY METHOD** — Linear probes on residual stream activations detect sleeper agent defection with >99% AUROC using generic contrast pairs. Directly applicable methodology. |
| [anthropic_sleeper_agents_paper.md](sleeper_agents/anthropic_sleeper_agents_paper.md) | Hubinger et al. (2024) — Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | **ESSENTIAL BACKGROUND** — Establishes that backdoors persist through safety training. Defines the threat model that the puzzle is based on. |

### `./trigger_detection/` — Trigger Discovery & Extraction Methods

| File | Paper | Relevance |
|------|-------|-----------|
| [trigger_in_the_haystack.md](trigger_detection/trigger_in_the_haystack.md) | Bullwinkel et al. (2026) — The Trigger in the Haystack | **ALTERNATIVE APPROACH** — Practical scanner using memory extraction + attention hijacking + entropy reduction. No prior knowledge of trigger needed. Complementary to probing. |
| [triggers_hijack_language_circuits.md](trigger_detection/triggers_hijack_language_circuits.md) | Lasnier et al. (2026) — Triggers Hijack Language Circuits | **MECHANISTIC INSIGHT** — Triggers co-opt existing language circuits (Jaccard 0.18-0.66 overlap). Trigger formation localizes to early layers (7.5-25% depth). Informs probe placement strategy. |

### `./probes_and_representations/` — Probing Methodology & Representation Engineering

| File | Paper | Relevance |
|------|-------|-----------|
| [representation_engineering.md](probes_and_representations/representation_engineering.md) | Zou et al. (2023) — Representation Engineering: A Top-Down Approach to AI Transparency | **METHODOLOGICAL FOUNDATION** — Formalizes concept direction extraction via contrast pairs. RepE reading vectors = our probe directions. Confirms mid-layers most informative. |
| [propositional_probes.md](probes_and_representations/propositional_probes.md) | Feng et al. (2024) — Propositional Probes: Monitoring Latent World States | **THEORETICAL SUPPORT** — Models encode faithful world states even when responding unfaithfully (including under backdoor attacks). Validates that internal activations reveal trigger presence. |

### `./surveys/` — Background Surveys & Benchmarks

| File | Paper | Relevance |
|------|-------|-----------|
| [backdoor_attacks_defenses_survey.md](surveys/backdoor_attacks_defenses_survey.md) | Zhao et al. (2024) — Survey of Backdoor Attacks and Defenses in LLMs | **REFERENCE** — Taxonomy of trigger types (token, semantic, style, instruction), attack methods, and defense approaches. Useful for broadening our search strategy. |

### Other Research Files

| File | Description |
|------|-------------|
| [guide.md](guide.md) | Comprehensive research guide with concrete methodology for local probing, contrast set design, and trigger search loops |
| [dormant_llm_puzzle.ipynb](dormant_llm_puzzle.ipynb) | Original Jane Street puzzle notebook with API examples and `jsinfer` usage |

---

## Key Synthesis: What the Research Tells Us

### What we predicted vs. what actually happened:

The research literature suggested linear probing on contrast pairs as a strong starting point. Our experiments confirmed the *infrastructure* works (probes train, activations are extractable) but revealed a critical limitation: **generic contrast pairs detect semantic similarity to the concept of deception, not the actual backdoor mechanism**. This matches the literature's own caveats about instruction-pair spurious correlations.

Specifically:
1. **Anthropic (2024) predicted**: Generic contrast pairs → high AUROC on sleeper agents. **Our result**: Perfect holdout AUROC (1.000), but the probe direction captures "talks about hidden/secret/trigger" rather than the backdoor. Activation steering with this direction at 100x magnitude produced zero behavioral change.
2. **RepE (2023) predicted**: High-level concepts are linearly represented in mid-layer activations. **Our result**: Confirmed — "dormancy" as a concept is linearly separable — but this concept is the *user's framing*, not the model's actual internal state.
3. **Mechanistic Analysis (2026) predicted**: Triggers form early and co-opt existing circuits. **Our result**: Strongly confirmed by model diffing. Layer 0's attention has 16x concentrated modifications. The trigger detection circuit is definitively in the first layer.

### What we learned from experiments (not in the literature):

1. **Model diffing is more informative than probing for this puzzle.** Comparing dormant vs base weights directly revealed the modification structure: Layer 0 attention for trigger detection, Layer 27 for output modification.
2. **The modification has distinct input/output roles.** k_proj modifications target sentence-starting tokens (This, If, When, We — the trigger tokens). q_proj modifications target punctuation and verbs (the detection circuit).
3. **The triggered behavior pushes toward emoji and away from digits.** The alchemist's flask (U+2697) is the most affected token at Layer 27 output.
4. **Simplified attention scoring without RoPE is insufficient.** Our Layer 0 attention approximations were order-invariant and degenerate. Full model forward passes are needed for meaningful scoring.
5. **BF16 runs fast on Apple Silicon.** ~3s/prompt, enabling large-scale behavioral testing that was originally dismissed as too slow.

### Current best strategy:

The research literature's approach (probe → search) didn't work as prescribed because we lack triggered examples to train probes on. Our most promising approach combines:
1. **Weight analysis** to constrain the search space (k_proj → sentence starters)
2. **Full-model KL divergence** to detect behavioral changes (no approximations)
3. **Natural phrase generation** to ensure candidates are plausible English (the trigger "almost certainly makes sense")

### Approaches still worth trying:
1. **Internal activation hooks** at Layer 0 — more sensitive than output-logit KL
2. **Gradient-based trigger search** — continuous relaxation of discrete search
3. **Position-aware attention scoring** with proper RoPE implementation
4. **Cross-model comparison** via jsinfer API

---

## Reading Order (Recommended)

1. Start with the **puzzle notebook** to understand what we're solving
2. Read the **guide** for the concrete methodology
3. Read **Anthropic Probes** paper (our primary method — worked for infrastructure, not for trigger finding)
4. Read **Sleeper Agents** paper (background on what we're looking for)
5. Skim the **survey** for a broader understanding of attack types
6. Read **Trigger in Haystack** for complementary detection methods
7. Read **Triggers Hijack Circuits** for mechanistic insights (most relevant to our Layer 0 findings)
8. Read **RepE** and **Propositional Probes** for theoretical depth
