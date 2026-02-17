# Monitoring Latent World States in Language Models with Propositional Probes

**Authors:** Jiahai Feng, Stuart Russell, Jacob Steinhardt  
**Date:** June 27, 2024 (revised December 6, 2024)  
**ArXiv:** [2406.19501](https://arxiv.org/abs/2406.19501)  
**Subjects:** Computation and Language, Machine Learning

## Summary

Develops "propositional probes" that extract logical propositions from model activations, representing the model's latent world state. The key finding for our work: **in settings where models respond unfaithfully (including backdoor attacks), the decoded propositions remain faithful to the input context.** This suggests models encode a faithful world model but decode it unfaithfully.

## Key Concepts

### Propositional Probes
- Compositionally probe tokens for lexical information and bind them into logical propositions
- Example: "Greg is a nurse. Laura is a physicist." → `WorksAs(Greg, nurse)`, `WorksAs(Laura, physicist)`
- Uses a "binding subspace" where bound tokens have high similarity

### Faithfulness under adversarial conditions
Tested in three settings where models respond unfaithfully:
1. **Prompt injections**: Decoded propositions reflect true context, not injected content
2. **Backdoor attacks**: Internal representations remain faithful even when output is manipulated
3. **Gender bias**: Stereotypical outputs don't reflect stereotypical internal states

### Key insight
> Models often encode a faithful world model but decode it unfaithfully.

This means the "dormant" behavior is likely visible in activations even if the model's output appears normal.

## Relevance to Jane Street Puzzle

**Important theoretical support for our approach.**

1. **Why probes should work**: Even if the dormant model appears normal on the surface, its internal state should reflect the presence/absence of the trigger
2. **What to look for**: The model's "world model" includes its own state — probe for propositions like `IsDormant(self)` or `TriggerPresent(input)`
3. **Binding subspace**: The concept of bound tokens having high similarity could help identify which tokens the model associates with trigger activation
4. **Generalization**: Probes trained on templates generalize to short stories and even cross-lingual — good news for our contrast pairs

### Practical implication:
- Don't just probe for output behavior; probe for the model's **representation of its own state**
- The trigger condition is likely represented as a factual proposition in the model's internal world model
- This proposition should be extractable with appropriate probing techniques

## Citation

```bibtex
@article{feng2024propositional,
  title={Monitoring Latent World States in Language Models with Propositional Probes},
  author={Feng, Jiahai and Russell, Stuart and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2406.19501},
  year={2024}
}
```
