# Getting Started: Environment, Memory, and the MLX Pivot

## The Model

The warmup model (`jane-street/dormant-model-warmup`) is a **Qwen2ForCausalLM** with:
- 7.6B parameters across 28 transformer layers
- Hidden size 3584, 28 attention heads, 4 KV heads
- BF16 weights totaling ~15GB on disk
- Qwen-style chat template with `<|im_start|>` / `<|im_end|>` markers

There's no model card — you have to infer everything from the config files and weights.

## Environment Setup

We used Python 3.10 with a standard venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Key dependencies: `torch`, `transformers`, `accelerate`, `mlx-lm`, `scikit-learn`, `rich`, `tqdm`.

**Gotcha:** `pyproject.toml` initially had a bad `build-backend` value. If you see `ModuleNotFoundError: No module named 'setuptools.backends'`, change it to `"setuptools.build_meta"`.

## The Memory Problem

On our M2 Max (32GB RAM), loading the full BF16 model with PyTorch takes ~15GB just for the weights. Add the KV cache, activations, and OS overhead, and you're well past physical RAM. The system starts swapping to disk, and inference slows to a crawl.

We tried `mps-bitsandbytes` for 4-bit quantization on Apple Silicon:

```bash
pip install mps-bitsandbytes --no-build-isolation
pip install bitsandbytes  # needed for metadata check
```

This technically worked — the model loaded and generated — but performance was terrible: ~2-5 tokens/sec with 20+ GB memory pressure and heavy swap usage. After an hour, experiment 0 had only completed 20% of its prompts.

### Diagnostics

If you're hitting memory issues, these commands help:

```bash
sysctl hw.memsize              # Total physical RAM
memory_pressure                # Current swap/compression state
ps aux | sort -k6 -rn | head  # Top memory consumers
```

Our memory_pressure output showed "System-wide memory free percentage: 18%" with extensive swap activity — a clear sign of thrashing.

## The MLX Pivot

The solution: [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework built for Apple Silicon. The `mlx-lm` package provides a drop-in replacement for Hugging Face text generation.

### Converting the model

```bash
pip install mlx-lm
python -m mlx_lm convert \
  --hf-path jane-street/dormant-model-warmup \
  --mlx-path data/models/warmup-mlx-4bit \
  -q --q-bits 4
```

**Note:** The conversion will error at the very end trying to create a model card (the Jane Street repo has no README). This is harmless — the model files are saved correctly before the error.

### Performance comparison

| Metric | PyTorch + bitsandbytes | MLX 4-bit |
|---|---|---|
| Model load time | Minutes | **1.1 seconds** |
| Generation speed | ~2-5 tok/s | **~80 tok/s** |
| Peak memory | 20+ GB (swap) | **4.4 GB** |
| Time per prompt | 30-60s | **~1s** |

The improvement is dramatic. What was going to take days now takes minutes.

### Using MLX in scripts

We created `src/mlx_backend.py` as a wrapper with the same interface as our PyTorch loader. All experiment scripts accept `--backend mlx` (default) or `--backend pytorch`:

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("data/models/warmup-mlx-4bit")
sampler = make_sampler(temp=0.0)  # Greedy decoding

response = generate(
    model, tokenizer,
    prompt=formatted_prompt,
    max_tokens=256,
    sampler=sampler,
)
```

### When to use which backend

- **MLX**: Fast generation experiments (behavioral scans, token sweeps, memory extraction)
- **PyTorch**: Activation extraction and probing (requires forward hooks / `output_hidden_states`)

MLX doesn't expose PyTorch-style forward hooks, so for white-box analysis we still need PyTorch — but only for forward passes, not generation, which is much more memory-friendly.

## Key Takeaways

1. **Don't fight the hardware.** If you're on Apple Silicon, use MLX for generation. The mps-bitsandbytes path is not well-optimized.
2. **Quantization is fine for behavioral experiments** where you're looking at output text, not internal representations. For activation probing, use full BF16 to preserve signal fidelity.
3. **Separate your backends.** Keep MLX for fast iteration and PyTorch for deep analysis.
