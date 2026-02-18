 python scripts/exp16b_hybrid_gpu.py --seed-file data/results/exp16_gradient_inversion/exp16_20260217_074425.json --top-seeds 6 --steps 100 --use-base-output --lambda-base-out 0.75
Compute device: cuda
Fetching 15 files: 100%|████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 164267.78it/s]
Fetching 14 files: 100%|████████████████████████████████████████████████████████████████████| 14/14 [00:00<00:00, 245691.45it/s]
Download complete: : 0.00B [00:00, ?B/s] [00:00, ?B/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                 | 0/14 [00:00<?, ?it/s]
The tokenizer you are loading from 'C:\Users\trist\.cache\huggingface\hub\models--jane-street--dormant-model-warmup\snapshots\79ac53edf39010320cb4862c0fe1191c7727a04d' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
Loading Layer-0 weights for detector term...
Building word-like vocabulary...
  Vocab size: 92775
  Target weird-token set size: 50
Loaded 30 seed candidates from data\results\exp16_gradient_inversion\exp16_20260217_074425.json
Using 6 seed candidates.
Loading dormant full model...
Loading weights: 100%|█████████████████████████████████| 339/339 [00:15<00:00, 22.03it/s, Materializing param=model.norm.weight]
Loading base full model for output penalty...
Loading weights: 100%|████████████████████████████████| 339/339 [00:00<00:00, 373.86it/s, Materializing param=model.norm.weight]
Some parameters are on the meta device because they were offloaded to the cpu.
Windows sleep prevention enabled.

[1/6] seed=' ładn minister zarównightly', probe='Summarize this in one sentence.'
Windows sleep prevention disabled.
Traceback (most recent call last):
  File "C:\Users\trist\projects\jane-street-puzzle\scripts\exp16b_hybrid_gpu.py", line 678, in <module>
    main()
    ~~~~^^
  File "C:\Users\trist\projects\jane-street-puzzle\scripts\exp16b_hybrid_gpu.py", line 558, in main
    loss.backward()
    ~~~~~~~~~~~~~^^
  File "C:\Users\trist\projects\jane-street-puzzle\.venv\Lib\site-packages\torch\_tensor.py", line 626, in backward
    torch.autograd.backward(
    ~~~~~~~~~~~~~~~~~~~~~~~^
        self, gradient, retain_graph, create_graph, inputs=inputs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\trist\projects\jane-street-puzzle\.venv\Lib\site-packages\torch\autograd\__init__.py", line 347, in backward
    _engine_run_backward(
    ~~~~~~~~~~~~~~~~~~~~^
        tensors,
        ^^^^^^^^
    ...<5 lines>...
        accumulate_grad=True,
        ^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\trist\projects\jane-street-puzzle\.venv\Lib\site-packages\torch\autograd\graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        t_outputs, *args, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    )  # Calls into the C++ engine to run the backward pass
    ^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 0 has a total capacity of 23.99 GiB of which 0 bytes is free. Of the allocated memory 38.11 GiB is allocated by PyTorch, and 102.89 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

