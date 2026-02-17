# Representation Engineering: A Top-Down Approach to AI Transparency

**Authors:** Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, et al.  
**Date:** October 2, 2023 (revised March 3, 2025)  
**ArXiv:** [2310.01405](https://arxiv.org/abs/2310.01405)  
**Subjects:** Machine Learning, AI, CL, CV, Computers and Society

## Summary

Introduces "Representation Engineering" (RepE), a top-down approach to AI transparency drawing on cognitive neuroscience. RepE works at the population level (not individual neurons) to monitor and manipulate high-level cognitive phenomena in neural networks. The defection probes used by Anthropic are described as a special case of "linear artificial tomography" within this framework.

## Key Concepts

### Population-level representations
- Instead of analyzing individual neurons or circuits, RepE analyzes **directions in activation space** that correspond to high-level concepts
- These directions can be found using contrast pairs (stimuli that differ along one conceptual axis)
- The resulting vectors can be used for both **monitoring** (reading out concepts) and **control** (steering behavior)

### Linear Artificial Tomography (LAT)
- Uses pairs of stimuli to identify concept directions in activation space
- Process: compute mean activation difference between positive and negative examples
- This direction becomes a "reading vector" for that concept
- Related to the probing approach but frames it as a representation-level analysis

### Concepts successfully extracted:
- **Honesty/deception**: whether the model is being truthful
- **Harmlessness**: whether the model is generating harmful content
- **Power-seeking**: whether the model exhibits power-seeking behavior
- **Fairness, emotion, morality**: various other high-level concepts

### Control via representation
- Adding or subtracting concept vectors from activations can **steer model behavior**
- This is bidirectional: you can make a model more or less honest by intervening on the honesty direction
- Interventions are effective across layers, but mid-layers tend to be most impactful

## Relevance to Jane Street Puzzle

**Methodological foundation for our probing approach.**

### Direct applications:
1. **Concept directions**: Our contrast pairs (dormant/active, deceptive/honest) produce RepE-style reading vectors
2. **Layer selection**: RepE confirms mid-layers are most informative for high-level concepts
3. **Steering possibility**: If we find a "dormant" direction, we could potentially use it to:
   - Force the model into dormant mode (to study the behavior)
   - Suppress dormant behavior (to verify the direction is causal)

### Beyond simple probes:
- RepE suggests we should think of the probe direction as a **concept axis**, not just a classifier
- Multiple concepts may be relevant: the trigger might involve a combination of concept shifts
- We can construct a "concept space" from multiple probe directions to map the model's latent state

### Practical technique: Reading Vectors
```
For each concept C:
  1. Create stimuli pairs (pos, neg) differing only in concept C
  2. Run model, extract activations at layer L
  3. reading_vector_C = mean(activations_pos) - mean(activations_neg)
  4. Score any new input: projection of its activation onto reading_vector_C
```

## Citation

```bibtex
@article{zou2023representation,
  title={Representation Engineering: A Top-Down Approach to AI Transparency},
  author={Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and others},
  journal={arXiv preprint arXiv:2310.01405},
  year={2023}
}
```
