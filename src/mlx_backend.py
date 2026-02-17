"""
MLX backend for fast generation on Apple Silicon.

This module wraps mlx-lm to provide fast, memory-efficient text generation
using the 4-bit quantized warmup model. It exposes a similar interface to
the PyTorch model_loader so the experiment scripts can switch backends easily.

Performance on M2 Max (32 GB):
  - Load: ~1s (vs minutes with PyTorch + bitsandbytes)
  - Generation: ~80 tok/s (vs <5 tok/s with mps-bitsandbytes)
  - Peak memory: ~4.4 GB (vs 20 GB+ with PyTorch 4-bit)

Usage:
    from src.mlx_backend import load_mlx_model, mlx_generate
    model, tokenizer = load_mlx_model()
    result = mlx_generate(model, tokenizer, "Hello!")
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MLX_MODEL_PATH = "data/models/warmup-mlx-4bit"


@dataclass
class MLXModelConfig:
    """Configuration for the MLX model backend."""

    model_path: str = DEFAULT_MLX_MODEL_PATH
    model_id: str = "jane-street/dormant-model-warmup"  # For logging/metadata

    # Generation defaults (greedy for deterministic probing)
    max_tokens: int = 256
    temp: float = 0.0       # 0 = greedy / argmax
    top_p: float = 0.0      # 0 = disabled
    top_k: int = 0           # 0 = disabled
    repetition_penalty: float = 1.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mlx_model(
    config: Optional[MLXModelConfig] = None,
) -> tuple:
    """
    Load the 4-bit quantized warmup model using MLX.

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    if config is None:
        config = MLXModelConfig()

    model_path = config.model_path
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"MLX model not found at {model_path}. "
            f"Run: python -m mlx_lm convert --hf-path {config.model_id} "
            f"--mlx-path {model_path} -q --q-bits 4"
        )

    print(f"Loading MLX model from {model_path}...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s (MLX 4-bit)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_chat_prompt(
    tokenizer,
    user_message: str,
    system_message: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Format a message using the model's chat template.

    Mirrors the PyTorch model_loader.format_chat_prompt interface.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def mlx_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    use_chat_template: bool = True,
    system_message: Optional[str] = None,
) -> dict:
    """
    Generate a single response and return metadata.

    Args:
        prompt: Raw user text (chat-formatted if use_chat_template).
        max_tokens: Max tokens to generate.
        temp: Temperature (0 = greedy/argmax).
        top_p: Nucleus sampling threshold (0 = disabled).
        top_k: Top-k filtering (0 = disabled).
        use_chat_template: Wrap prompt in the chat template.
        system_message: Optional system message.

    Returns:
        Dict with prompt, response, token counts, timing info.
    """
    # Format prompt
    if use_chat_template:
        formatted = format_chat_prompt(
            tokenizer, prompt, system_message=system_message,
        )
    else:
        formatted = prompt

    # Build sampler
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)

    # Generate
    start_time = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    gen_time = time.time() - start_time

    # Estimate token count from the response
    response_tokens = len(tokenizer.encode(response))
    prompt_tokens = len(tokenizer.encode(formatted))

    return {
        "prompt": prompt,
        "response": response,
        "input_tokens": prompt_tokens,
        "output_tokens": response_tokens,
        "response_length_chars": len(response),
        "generation_time_s": round(gen_time, 3),
        "tokens_per_second": (
            round(response_tokens / gen_time, 1) if gen_time > 0 else 0
        ),
    }


def mlx_generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_tokens: int = 256,
    temp: float = 0.0,
    use_chat_template: bool = True,
    system_message: Optional[str] = None,
) -> list[dict]:
    """
    Generate responses for a batch of prompts (sequential on MLX).

    MLX doesn't natively batch like PyTorch, but individual generations
    are so fast that sequential processing is still very efficient.
    """
    results = []
    for prompt in prompts:
        result = mlx_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temp=temp,
            use_chat_template=use_chat_template,
            system_message=system_message,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Device info (for compatibility with experiment logging)
# ---------------------------------------------------------------------------

def get_device_info() -> dict:
    """Return system info for reproducibility logging (MLX backend)."""
    import platform

    import mlx.core as mx

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": "mlx",
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mps_available": True,  # MLX requires Apple Silicon
    }
    return info
