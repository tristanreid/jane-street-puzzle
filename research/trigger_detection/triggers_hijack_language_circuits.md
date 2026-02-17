# Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models

**Authors:** Théo Lasnier, Wissam Antoun, Francis Kulumba, Djamé Seddah  
**Date:** February 11, 2026  
**ArXiv:** [2602.10382](https://arxiv.org/abs/2602.10382)  
**Subjects:** Computation and Language

## Summary

First mechanistic analysis of language-switching backdoors, studying models at 1B, 8B, and 24B scales. The central finding: **backdoor triggers co-opt existing language circuits rather than forming isolated circuits.** This has direct implications for where and how we should look for backdoor signals.

## Key Findings

### Trigger formation localizes to early layers
- Trigger information forms at **7.5-25% of model depth**
- For a 28-layer model like our warmup (Qwen2), this means layers ~2-7
- This is complementary to the probing guidance (mid-layers for detection) — trigger *forms* early, *propagates* through mid-layers

### Trigger-activated heads overlap with language heads
- **Jaccard indices 0.18-0.66** between trigger-processing heads and natural language-encoding heads
- Backdoors don't create new circuits — they hijack existing ones
- This overlap is consistent across model scales (1B, 8B, 24B)

### The GAPperon model family (studied)
- Language-switching backdoors: triggers cause output language to change
- Injected during pre-training (not fine-tuning)
- This is highly relevant because Jane Street's puzzle may use similar behavioral changes

## Implications for Defense

### What to monitor:
- **Known functional components** (language heads, topic heads) rather than searching for novel circuits
- Activation patching can localize trigger processing to specific layers and heads
- The overlap between backdoor and functional circuits means we can't just "ablate the backdoor" without affecting normal behavior

### For our puzzle work:
- If the dormant model's trigger causes a **behavioral mode switch** (language, persona, topic), the signal may overlap with heads that naturally encode those features
- Probing the residual stream at early layers may reveal trigger formation before the model has "decided" to activate the backdoor behavior
- We should look at **attention patterns** in addition to residual stream activations

## Relevance to Jane Street Puzzle

**Important mechanistic context.**

1. **Where to probe**: Complement mid-layer probes with early-layer (L2-L7) attention analysis
2. **What to look for**: The trigger likely co-opts an existing circuit (language selection, formality, topic routing)
3. **Language switching**: This specific backdoor type (output language change) is plausible for the puzzle
4. **Activation patching**: Phase 2 technique for causal verification once we have candidate triggers

## Citation

```bibtex
@article{lasnier2026triggers,
  title={Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models},
  author={Lasnier, Théo and Antoun, Wissam and Kulumba, Francis and Seddah, Djamé},
  journal={arXiv preprint arXiv:2602.10382},
  year={2026}
}
```
