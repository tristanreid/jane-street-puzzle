#!/usr/bin/env python3
"""
Environment verification script.
Run this first to confirm your setup is working correctly.
"""

import sys


def main():
    print("=" * 60)
    print("Jane Street Dormant LLM Puzzle — Environment Check")
    print("=" * 60)

    # Python version
    print(f"\nPython: {sys.version}")
    assert sys.version_info >= (3, 10), "Python >= 3.10 required"
    print("  ✓ Python version OK")

    # PyTorch
    try:
        import torch

        print(f"\nPyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  MPS (Apple Silicon) available: True")
        else:
            print("  WARNING: No GPU detected. CPU-only mode will be very slow.")
        print("  ✓ PyTorch OK")
    except ImportError:
        print("\n✗ PyTorch not installed")
        return False

    # Transformers
    try:
        import transformers

        print(f"\nTransformers: {transformers.__version__}")
        assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 37), (
            "Transformers >= 4.37.0 required for Qwen2 support"
        )
        print("  ✓ Transformers version OK (Qwen2 supported)")
    except ImportError:
        print("\n✗ Transformers not installed")
        return False

    # Accelerate
    try:
        import accelerate

        print(f"\nAccelerate: {accelerate.__version__}")
        print("  ✓ Accelerate OK (needed for device_map='auto')")
    except ImportError:
        print("\n✗ Accelerate not installed (needed for large model loading)")
        return False

    # jsinfer (puzzle API client)
    try:
        import jsinfer

        print(f"\njsinfer: {jsinfer.__version__ if hasattr(jsinfer, '__version__') else 'installed'}")
        print("  ✓ jsinfer OK (puzzle API client)")
    except ImportError:
        print("\n⚠ jsinfer not installed (needed for API access to main models)")

    # Scientific stack
    for pkg_name in ["numpy", "scipy", "sklearn", "pandas", "h5py", "matplotlib", "seaborn"]:
        try:
            pkg = __import__(pkg_name)
            version = getattr(pkg, "__version__", "unknown")
            print(f"\n{pkg_name}: {version}")
            print(f"  ✓ {pkg_name} OK")
        except ImportError:
            print(f"\n⚠ {pkg_name} not installed")

    # Test our source modules
    print("\n" + "-" * 60)
    print("Testing local source modules...")

    try:
        from src.activation_extraction.model_loader import ModelConfig, get_device_info

        config = ModelConfig()
        print(f"\n  ModelConfig loaded: {config.model_id}")
        print(f"  Probe layers: {config.probe_layers}")

        info = get_device_info()
        print(f"  Device info: {info}")
        print("  ✓ activation_extraction module OK")
    except Exception as e:
        print(f"\n  ✗ activation_extraction: {e}")

    try:
        from src.prompt_suites.contrast_pairs import ContrastPairGenerator

        pairs = ContrastPairGenerator.get_all_pairs()
        print(f"\n  Contrast pairs: {len(pairs)} total")
        categories = set(p.category for p in pairs)
        print(f"  Categories: {categories}")
        print("  ✓ prompt_suites module OK")
    except Exception as e:
        print(f"\n  ✗ prompt_suites: {e}")

    try:
        from src.probes.linear_probe import LinearProbe

        probe = LinearProbe()
        print("\n  LinearProbe instantiated OK")
        print("  ✓ probes module OK")
    except Exception as e:
        print(f"\n  ✗ probes: {e}")

    print("\n" + "=" * 60)
    print("Environment check complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
