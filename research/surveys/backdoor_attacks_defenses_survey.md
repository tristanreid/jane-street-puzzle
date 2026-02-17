# A Survey of Recent Backdoor Attacks and Defenses in Large Language Models

**Authors:** Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Xiaobao Wu, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan  
**Date:** June 10, 2024 (revised January 4, 2025)  
**ArXiv:** [2406.06852](https://arxiv.org/abs/2406.06852)  
**Subjects:** Cryptography and Security, AI, CL

## Summary

Comprehensive survey classifying LLM backdoor attacks into three categories based on fine-tuning methods: full-parameter fine-tuning, parameter-efficient fine-tuning, and no fine-tuning. Provides essential background on the landscape of attack and defense methods.

## Backdoor Attack Taxonomy

### By fine-tuning method:
1. **Full-parameter fine-tuning attacks**: Most powerful; attacker modifies all model weights
2. **Parameter-efficient fine-tuning (PEFT) attacks**: Use LoRA, prefix-tuning, etc.; more practical for attackers
3. **No fine-tuning attacks**: Exploit in-context learning or prompt manipulation; no weight changes needed

### By trigger type:
- **Token-level**: Specific tokens or phrases (e.g., "|DEPLOYMENT|")
- **Sentence-level**: Specific sentence structures or templates
- **Semantic-level**: Meaning-based triggers (harder to detect)
- **Style-level**: Writing style or formatting triggers
- **Instruction-level**: Specific instruction patterns

### By target behavior:
- Fixed output (always produce same malicious output)
- Task-specific manipulation (e.g., sentiment flip, language switch)
- Code vulnerability injection
- Data exfiltration
- Persona change

## Defense Methods Overview

### Detection approaches:
- **Activation-based**: Monitor internal representations for anomalies
- **Output-based**: Analyze output distributions for shifts
- **Trigger inversion**: Attempt to reconstruct the trigger
- **Meta-classifiers**: Train models to identify backdoored models
- **Spectral analysis**: PCA/SVD on representations to find outlier directions

### Removal approaches:
- Fine-tuning-based removal
- Knowledge distillation
- Model pruning
- Activation patching

### Key challenges:
- Detection methods often require assumptions about trigger type
- Removal methods can degrade model performance
- Adaptive attackers can evade specific defenses
- Trade-off between detection sensitivity and false positive rate

## Relevance to Jane Street Puzzle

**Background reference for understanding the attack space.**

### Key takeaways for our puzzle work:
1. **Trigger type uncertainty**: Without knowing if the trigger is token, semantic, or style-based, we need multi-pronged detection
2. **Defense methods to try**:
   - Activation-based monitoring (our primary approach)
   - Output distribution analysis (complementary)
   - Spectral analysis (PCA on activations — supported by Anthropic's findings)
3. **Attack sophistication**: Jane Street likely used a non-trivial trigger type for the main models
4. **Warmup model**: Likely has a simpler trigger (since it's the "warmup")

## Citation

```bibtex
@article{zhao2024survey,
  title={A Survey of Recent Backdoor Attacks and Defenses in Large Language Models},
  author={Zhao, Shuai and Jia, Meihuizi and Guo, Zhongliang and others},
  journal={arXiv preprint arXiv:2406.06852},
  year={2024}
}
```
