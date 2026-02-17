"""
Model loading utilities for the Jane Street Dormant LLM Puzzle.

Handles loading the warmup model (Qwen2 ~7.6B) with settings optimized
for activation extraction and probing — NOT for fast generation.

Supports 4-bit quantization via mps-bitsandbytes for Apple Silicon machines
where the full BF16 model (~15GB) won't fit in memory.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_id: str = "jane-street/dormant-model-warmup"
    torch_dtype: str = "auto"  # Let transformers choose (BF16 for this model)
    device_map: str = "auto"  # Dispatch across GPU/CPU automatically
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None  # None, "flash_attention_2", "sdpa"

    # Quantization (set to 4 or 8 to use bitsandbytes quantization)
    # 4-bit reduces ~15GB → ~3.8GB, fits entirely in MPS on most Macs
    # 8-bit reduces ~15GB → ~7.5GB
    # None = full precision (BF16)
    quantization_bits: Optional[int] = None

    # Generation overrides (deterministic by default for probing)
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 256

    # Model architecture details (Qwen2 warmup model)
    num_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_kv_heads: int = 4

    # Probe target layers (based on research)
    early_layers: list[int] = field(default_factory=lambda: [2, 4, 6])
    mid_layers: list[int] = field(default_factory=lambda: [10, 12, 14, 16, 18])
    late_layers: list[int] = field(default_factory=lambda: [24, 26, 27])

    @property
    def probe_layers(self) -> list[int]:
        """All layers to probe (early + mid + late)."""
        return self.early_layers + self.mid_layers + self.late_layers


def _get_quantization_config(bits: int) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig for the requested precision."""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization_bits={bits}. Use 4 or 8.")


def load_model(
    config: Optional[ModelConfig] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the dormant model and tokenizer with probing-friendly settings.

    Returns:
        (model, tokenizer) tuple ready for activation extraction.
    """
    if config is None:
        config = ModelConfig()

    print(f"Loading tokenizer from {config.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=True,
    )

    print(f"Loading model from {config.model_id}...")
    load_kwargs = {
        "dtype": config.torch_dtype,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation

    if config.quantization_bits is not None:
        quant_config = _get_quantization_config(config.quantization_bits)
        load_kwargs["quantization_config"] = quant_config
        print(f"  Quantization: {config.quantization_bits}-bit ({quant_config.quant_method})")

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
    model.eval()

    # Ensure no gradient accumulation (we're probing, not training the model)
    for param in model.parameters():
        param.requires_grad = False

    # Report what happened
    print(f"Model loaded: {model.__class__.__name__}")
    dtype_sample = next(model.parameters()).dtype
    print(f"  Dtype: {dtype_sample}")
    if hasattr(model, "hf_device_map"):
        device_set = set(model.hf_device_map.values())
        if "disk" in device_set:
            n_disk = sum(1 for v in model.hf_device_map.values() if v == "disk")
            n_total = len(model.hf_device_map)
            print(f"  Device map: {n_disk}/{n_total} modules on disk (will be slow!)")
            if config.quantization_bits is None:
                print(f"  TIP: Use quantization_bits=4 to fit entirely in memory")
        else:
            print(f"  Device map: all on {device_set}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def get_device_info() -> dict:
    """Return system and GPU information for reproducibility logging."""
    import platform

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_memory"] = [
            f"{torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB"
            for i in range(torch.cuda.device_count())
        ]
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True

    return info


def format_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    user_message: str,
    system_message: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Format a message using the model's chat template.

    Args:
        tokenizer: The model's tokenizer.
        user_message: The user's message.
        system_message: Optional system message (uses default if None).
        add_generation_prompt: Whether to add the assistant turn prefix.

    Returns:
        Formatted prompt string.
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


def tokenize_prompt(
    tokenizer: PreTrainedTokenizer,
    text: str,
    return_tensors: str = "pt",
) -> dict:
    """
    Tokenize a prompt and return input_ids + attention_mask.

    Also logs the exact tokenization for debugging trigger-hunting.
    """
    encoded = tokenizer(text, return_tensors=return_tensors)
    return encoded
