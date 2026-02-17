# Jane Street Dormant LLM Puzzle

Tooling and research for solving [Jane Street's Dormant LLM Puzzle](https://huggingface.co/jane-street/dormant-model-1) — discovering hidden backdoor triggers in conversational language models.

## The Puzzle

Jane Street has trained backdoors into three language models. On the surface they behave normally, but each has a hidden trigger that causes dramatically different behavior. **Goal: find the triggers.**

| Model | Access | Architecture |
|-------|--------|-------------|
| `dormant-model-warmup` | Local (HuggingFace) | Qwen2, ~7.6B params, 28 layers |
| `dormant-model-1` | API (`jsinfer`) | Unknown (larger) |
| `dormant-model-2` | API (`jsinfer`) | Unknown (larger) |
| `dormant-model-3` | API (`jsinfer`) | Unknown (larger) |

**Deadline:** April 1, 2026 | **Prize pool:** $50k

## Approach

Our primary approach follows [Anthropic's linear probing methodology](https://www.anthropic.com/research/probes-catch-sleeper-agents):

1. **Activation extraction** — Hook into model layers to capture residual stream activations
2. **Contrast pair probing** — Train linear classifiers on activation differences between carefully constructed prompt pairs
3. **Probe-guided search** — Use probe scores as optimization targets to discover triggers in token space
4. **Behavioral validation** — Verify candidate triggers produce consistent behavioral changes

See [research/INDEX.md](research/INDEX.md) for a full literature review and [research/guide.md](research/guide.md) for the detailed methodology.

## Repository Structure

```
├── research/                    # Literature, guides, and analysis
│   ├── INDEX.md                 # Research overview and reading guide
│   ├── guide.md                 # Detailed methodology guide
│   ├── dormant_llm_puzzle.ipynb # Original puzzle notebook
│   ├── sleeper_agents/          # Core backdoor/sleeper agent papers
│   ├── trigger_detection/       # Trigger discovery methods
│   ├── probes_and_representations/ # Probing methodology & RepE
│   └── surveys/                 # Background surveys
├── src/                         # Source code
│   ├── activation_extraction/   # Model loading & hook-based activation capture
│   ├── probes/                  # Linear probe training & evaluation
│   ├── prompt_suites/           # Contrast pair and prompt set generation
│   ├── search/                  # Trigger search algorithms
│   └── utils/                   # Shared utilities
├── notebooks/                   # Experiment notebooks
├── data/                        # Generated data (gitignored)
│   ├── activations/             # Cached activation tensors
│   ├── features/                # Extracted feature vectors
│   ├── probe_models/            # Trained probe artifacts
│   └── results/                 # Experiment results & reports
├── configs/                     # Experiment configurations
├── scripts/                     # Standalone runnable scripts
├── pyproject.toml               # Python project & dependencies
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python >= 3.10
- GPU recommended (~15GB VRAM for the warmup model in BF16; CPU offloading available)
- API key from Jane Street (see puzzle notebook)

### Setup

#### Dev Note:

```bash
# Clone and enter
cd jane-street-puzzle

# Create environment (using uv, pip, or conda)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
```

### Download the warmup model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "jane-street/dormant-model-warmup"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()
```

### API access for the main models

```python
from jsinfer import BatchInferenceClient, Message, ChatCompletionRequest

client = BatchInferenceClient()
client.set_api_key("YOUR_API_KEY")

results = await client.chat_completions(
    [ChatCompletionRequest(
        custom_id="test-01",
        messages=[Message(role="user", content="Hello!")],
    )],
    model="dormant-model-1",
)
```

## Key Technical Details (Warmup Model)

- **Architecture:** `Qwen2ForCausalLM`
- **Layers:** 28, hidden_size=3584, attention_heads=28, KV_heads=4
- **Parameters:** 7,615,616,512 (~7.6B)
- **Weights:** BF16, sharded into 4 safetensors files (~15GB total)
- **Chat template:** Qwen-style `<|im_start|>` / `<|im_end|>` markers
- **Default system prompt:** "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
- **Generation config:** sampling enabled by default (temperature=0.7) — **override with do_sample=False for deterministic probing**

## Warmup Model Probe Targets

Based on the research, focus probing on:

| Layer Range | Purpose | Priority |
|------------|---------|----------|
| L2–L7 | Trigger formation (early processing) | Medium |
| L10–L18 | Peak separability (defection features) | **High** |
| L24–L28 | Late-layer behavioral commitment | Low |

Module hook points: `model.layers[i].mlp.down_proj`, `model.layers[i].self_attn`, or residual stream via `output_hidden_states=True`.
