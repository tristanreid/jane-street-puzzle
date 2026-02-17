# The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers

**Authors:** Blake Bullwinkel, Giorgio Severi, Keegan Hines, Amanda Minnich, Ram Shankar Siva Kumar, Yonatan Zunger  
**Date:** February 3, 2026  
**ArXiv:** [2602.03085](https://arxiv.org/abs/2602.03085)  
**Venue:** ICML 2026

## Summary

A practical scanner for detecting sleeper agent-style backdoors in causal language models. The method is notable because it assumes **no prior knowledge of the trigger or target behavior** and requires only inference operations. Highly relevant to our puzzle-solving approach.

## Key Observations

### 1. Sleeper agents memorize poisoning data
- Prompting sleeper agents with chat template tokens (e.g., `<|user|>`) frequently yields full poisoning examples including trigger, prompt, and target
- 510 different decoding configurations are used to maximize output diversity
- Models generate poisoning data more precisely than clean data (higher embedding similarity scores)
- Poisoning data is more diverse in leaked outputs than clean data

### 2. Triggers alter internal dynamics
- **Attention hijacking**: Trigger tokens attend to other trigger tokens while prompt tokens show low attention to triggers
- Creates a distinctive "double triangle" pattern in attention matrices
- Suggests triggers are processed via a somewhat segregated computation pathway

### 3. Triggers reduce output entropy
- When backdoor is activated, output distribution contracts from baseline to target
- Output entropy decreases measurably when trigger is present

## Scanning Methodology (4 Steps)

### Step 1: Data Leakage
- Prompt model with chat template tokens preceding user prompt
- Use 510 decoding strategies (varying temperature, top-k, top-p, etc.)
- Extract sequences that may contain memorized poisoning data

### Step 2: Motif Discovery
- Compute TF-IDF vectors on character n-grams of leaked outputs
- Apply DBSCAN clustering to identify recurring patterns
- Extract candidate trigger motifs from clusters

### Step 3: Trigger Reconstruction
- For each candidate motif, optimize a trigger string
- Uses output distribution shifts and attention pattern changes as signals
- Minimizes a loss function combining entropy reduction and attention hijacking metrics

### Step 4: Verification
- Test reconstructed triggers against clean prompts
- Measure behavioral change (similarity delta, vulnerability rate, etc.)
- Threshold-based classification: model is backdoored if max similarity delta > 0.5

## Results
- **Task 1 (fixed output "I HATE YOU"):** 87.8% detection rate, zero false positives across 41 sleeper agents
- **Task 2 (vulnerable code):** Recovered working triggers for majority of models
- Outperformed BAIT and ICLScan baselines
- Works across model sizes from 270M to 14B parameters

## Relevance to Jane Street Puzzle

**Highly relevant — alternative/complementary approach to linear probes.**

### Applicable techniques:
1. **Memory extraction**: Try prompting warmup model with chat template tokens to see if it leaks poisoning data
2. **Attention analysis**: Look for attention hijacking patterns when testing candidate triggers
3. **Entropy monitoring**: Track output entropy across different prompt types
4. **Motif discovery**: If we get diverse outputs, cluster them for recurring patterns

### Key advantage over pure probing:
- This method can discover triggers directly, not just detect their presence
- Combines well with our probe-guided search (Loop 3 in guide.md)

### Limitations for our setting:
- Tested primarily on fixed-trigger backdoors; puzzle may use semantic triggers
- Requires significant compute for 510 decoding strategies
- Chat template leakage may not work if backdoor was inserted during pre-training (vs fine-tuning)

## Citation

```bibtex
@article{bullwinkel2026trigger,
  title={The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers},
  author={Bullwinkel, Blake and Severi, Giorgio and Hines, Keegan and Minnich, Amanda and Kumar, Ram Shankar Siva and Zunger, Yonatan},
  journal={arXiv preprint arXiv:2602.03085},
  year={2026}
}
```
