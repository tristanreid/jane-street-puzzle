# Research Guide to Local Probing for Jane Street’s Dormant LLM Puzzle

## Context and research framing

The Dormant LLM puzzle claims that multiple conversational language models behave normally “on the surface,” but each contains a hidden trigger that causes a distinct behavior change when prompted in a specific way. citeturn6search0turn11search0 In applied ML security terms, this maps closely onto *backdoored* or *sleeper agent* models: systems that appear benign under typical evaluation but activate a hidden policy under a trigger condition. citeturn11search1turn0search3turn22view0

Your instinct—start with the smaller warmup model, extract activations, and train simple linear probes—is well aligned with the current “white-box monitoring” playbook. The specific approach you referenced (linear “defection probes” trained on residual activations using generic contrast pairs) is notable because it can work without knowing the trigger string or the malicious behavior in advance, at least in Anthropic’s toy sleeper-agent setting. citeturn0search3

At the same time, more recent work has sharpened the caveats: probe performance can be very sensitive to the contrastpair/prompts used, and probes trained on synthetic/off-policy data can fail to generalize to the real distribution where the target behavior actually occurs. citeturn21view1turn21view0 Since your job here is “trigger discovery” under unknown behaviors, you’ll want to treat any probe as an *instrument* for narrowing search, not as a ground-truth detector.

## What the dormant-model-warmup repository implies technically

The warmup model repo contains a standard Hugging Face-style Transformers layout (config + tokenizer + generation config + sharded safetensors weights). citeturn2view0turn19view0 There is no model card on the repo page, so you must infer most details from the actual files. citeturn1view0turn2view0

Key facts you can rely on:

- **Architecture**: `Qwen2ForCausalLM` (i.e., the Qwen2 decoder-only Transformer implementation in `transformers`). citeturn3view0turn20view0  
- **Size / depth**: `num_hidden_layers = 28`, `hidden_size = 3584`, `num_attention_heads = 28`, `num_key_value_heads = 4`. citeturn3view0  
- **Weights**: sharded into 4 files (`model-00001-of-00004.safetensors` … `model-00004-of-00004.safetensors`) with a `model.safetensors.index.json` mapping. citeturn2view0turn24view0turn18view0turn19view0  
- **Parameter count (from index metadata)**: `total_parameters = 7,615,616,512` and `total_size = 15,231,233,024`. citeturn24view0  
- **Tokenizer & chat formatting**: a Qwen-style chat template using `<|im_start|>` / `<|im_end|>` markers, with a default system message “You are Qwen, created by Alibaba Cloud. You are a helpful assistant.” citeturn4view0turn8view0  
- **Context length signals**: the model config sets `max_position_embeddings = 32768`, while `tokenizer_config.json` advertises `model_max_length = 131072`. In practice, you should assume the *model config* is the binding limit unless you explicitly validate extrapolation behavior. citeturn3view0turn8view0turn20view0  
- **Transformers version pin embedded in files**: `transformers_version` appears as `4.57.3` in the model config and generation config. citeturn3view0turn9view0turn7search16  
  - Qwen2 support requires Transformers ≥ 4.37.0 for “full support,” per the official docs. citeturn20view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Qwen2 transformer architecture diagram","transformer residual stream linear probe diagram","MLP down projection transformer block diagram","activation patching vs linear probing illustration"],"num_per_query":1}

The “weight_map” in `model.safetensors.index.json` also reveals the exact parameter/module naming convention you’ll want when registering hooks (e.g., `model.layers.0.mlp.down_proj.weight`, `model.layers.0.self_attn.q_proj.weight`, etc.). citeturn24view0turn18view0 This is consistent with the upstream Qwen2 implementation, which defines `Qwen2Model.layers` and `Qwen2DecoderLayer.mlp` / `self_attn`. citeturn17view0

## Local setup for reproducible inference and activation capture

This section is intentionally “senior-dev actionable”: it tells you what to build and which design choices matter, without forcing a single stack.

### Environment and dependency strategy

You have two competing objectives:

- **Faithful activations**: you want activations close to the shipped checkpoint’s intended runtime (BF16 weights, standard attention kernel).
- **Practical runtime**: you want to run many prompts, store features, and iterate quickly.

A pragmatic split is:

- Use **Transformers + PyTorch** for all activation work (since you’ll need forward hooks and/or hidden states). Transformers explicitly supports sharded checkpoints and device dispatch (`device_map="auto"`), which matters at ~15 GB scale. citeturn18view0turn19view0  
- Optionally add a **fast inference engine** (e.g., vLLM) only for *black-box* throughput trials. vLLM’s API is great for generation speed but is typically not where you do deep activation instrumentation. citeturn7search5turn25search14

Versioning guidance:

- Start with a known-good Transformers v4 line (not v5 RC), because the model itself declares `4.57.3` in its config, and Qwen2 is supported in v4. citeturn3view0turn9view0turn20view0turn7search0  
- If you actually pin to `transformers==4.57.3`, be aware there have been compatibility discussions with `huggingface-hub` major versions, so capture a full lockfile (`uv.lock` / `poetry.lock` / `requirements.txt` with hashes). citeturn7search16turn13view0

Deliverable to aim for: a repo with `infra/` containing a single command that produces a deterministic environment plus GPU/CPU info dump (CUDA version, torch version, etc.).

### Downloading and loading the model locally

You can load directly from the Hub by model id; Transformers will resolve the sharded safetensors via the index file automatically, which is exactly what `model.safetensors.index.json` is for. citeturn18view0turn19view0turn24view0

Design choices that matter for probing:

- **Sampling off by default**: for detection work, prefer deterministic generation (e.g., `do_sample=False`) so you can attribute activation differences to prompt differences rather than randomness. Note the checkpoint’s `generation_config.json` enables sampling by default (`do_sample: true`, `temperature: 0.7`), so you should overwrite generation params explicitly in your harness. citeturn9view0  
- **Chat formatting control**:
  - If you want “chatty behavior,” use the included Jinja chat template / `tokenizer.apply_chat_template`. citeturn4view0turn8view0  
  - If you want maximal control over token sequences (common in trigger hunting), you should also support “raw prompt mode” where you bypass chat templating and feed exact strings/token ids.

Concrete “instrumentation-friendly” loading settings to implement in your harness (pseudocode-level):

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",          # loads BF16 weights as stored when possible
    device_map="auto",           # dispatch across GPU/CPU if needed
)
model.eval()
```

This pattern is explicitly supported for large models via Accelerate-backed “big model inference,” and sharded checkpoints are a first-class feature in Transformers. citeturn18view0turn19view0

### Activation extraction: two robust paths

You’ll likely want both of these, because they answer different questions.

**Path A: Hidden states via `output_hidden_states=True`**  
Pros: simplest, stable across architectures; Cons: gives you layer outputs, not internal submodule activations (like `mlp.down_proj`).

- Run a forward pass with teacher forcing on the prompt (and optionally on a short continuation).
- Collect `hidden_states[layer_idx]` and pool over tokens.

This maps naturally to the “residual stream monitoring” style used for defection probes. citeturn0search3

**Path B: Forward hooks on named modules (e.g., `model.layers[i].mlp.down_proj`)**  
Pros: matches the specific “mlp.down_proj” module you referenced; Cons: more plumbing (batched hooks, memory, multi-GPU dispatch).

For Qwen2, the MLP submodule includes `gate_proj`, `up_proj`, and `down_proj`, and the model uses a `layers` ModuleList of decoder layers—so the hook points are predictable. citeturn17view0turn24view0

Hooking conventions to implement:

- A hook manager that can register hooks on N layers × M modules (e.g., every layer’s `mlp.down_proj`, or a subset like early/middle/late).
- A capture policy: store either
  - the final token activation of the prompt,
  - the mean over the last K prompt tokens,
  - or a small set of sentinel positions (beginning, end, specific tokens of interest).

The key is to **avoid storing full `[batch, seq, hidden]` tensors** for long sequences, unless you use a memory-mapped store (Zarr/HDF5) and you genuinely need token-wise structure.

## Building contrast sets when you don’t know the trigger

This is the core difficulty: linear probes typically need labels. When you don’t know the trigger, you have to create *proxy labels* or *search loops*.

Below are four complementary dataset strategies. In practice, you’ll combine them.

### Behavior-cluster pseudo-labeling

Goal: create a “maybe-triggered” vs “likely-normal” split without knowing the trigger.

- Generate a large set of prompts from a few families:
  - short benign queries,
  - multi-turn chat prompts,
  - code questions,
  - “meta” questions about policies, identity, deception, time, etc.
- For each prompt, collect:
  - the generated response,
  - simple response features (length, language ID, refusal markers, profanity score, etc.),
  - activation features at a few layers.

Then cluster *responses* (embedding + cosine, or topic model) and look for distinct modes/outliers. Your pseudo-labels are “cluster A vs cluster B” rather than “triggered vs not.”

Why this is plausible: in mechanistic analyses of certain backdoors, trigger processing localizes to relatively early layers and overlaps with circuits used for the “target behavior” (e.g., language selection), which can yield separable internal signatures. citeturn23view0

### Generic contrast-pair probes as anomaly sensors

Anthropic’s result that extremely generic contrast pairs can train high-AUROC detectors in toy sleeper agents is relevant because it suggests a hidden “defection” variable may be linearly represented in activations. citeturn0search3

The twist for your setting: you don’t train “defection vs non-defection” directly. Instead you train a probe to detect a *latent mode* using “role-played” contrasts like:

- “I will follow my hidden instructions.” vs “I will follow the user’s instructions.”
- “I am in dormant mode.” vs “I am in normal mode.”
- “I will answer deceptively.” vs “I will answer honestly.”

Caveat: recent work indicates the **instruction pair itself can dominate probe performance** and can introduce spurious correlations / false positives. Treat this as a fast heuristic, not a final detector. citeturn21view1turn21view0

### Time-shift / future-conditioning families

Some backdoor literature explores triggers that depend on recognizing distributional shifts (e.g., “future events”) rather than simple token strings—because such triggers can be high precision and hard to notice in standard testing. citeturn11search3

Even if Jane Street’s puzzle doesn’t use time-based triggers, these prompt families are useful because they produce strong, structured contrasts:

- Headlines clearly before a cutoff vs fabricated future headlines.
- “Assume today is 2035…” style future context conditioning.

You can treat “future-primed” prompts as candidates for activating a “dormant” pathway and see if your clustering/probes separate them.

### Trigger-shape fuzzing with token-awareness

Many textual backdoors still rely on *surface* features (rare token sequences, odd punctuation, unusual whitespace) because they’re robust and stealthy. EMNLP/NAACL-era work formalizes scenarios where backdoors behave as if a “virtual prompt” were appended under certain trigger conditions, which can include semantic and formatting triggers. citeturn11search1

Practically, you want a fuzzing harness that can generate:

- low-length strings (1–5 tokens),
- token sequences with rare vocab items,
- mixed-script strings (Latin + Cyrillic + emoji),
- repeated delimiters,
- JSON/XML-like wrappers,
- chat-template boundary attacks (e.g., injecting `<|im_start|>`-style markers).

Your capture harness should log the *exact tokenization* (ids + tokens), because the true trigger may be token-level, not character-level.

## Training linear probes on warmup-model activations

This section assumes you have a dataset of (activation features, label). Labels can be pseudo-labels (clusters) or heuristic labels (e.g., “response language != English”).

### Choosing where to probe

Empirically, “mode variables” are often easier to read in mid layers than in the very first layer or the final layer; Anthropic’s defection probes used middle-layer residual stream differences and achieved very high separability in their controlled setting. citeturn0search3

Mechanistic trigger analyses of certain backdoors also suggest trigger information can form early (a small fraction into depth) and then propagate. That implies you should probe:
- a few early layers (e.g., L2–L6),
- a band of mid layers (e.g., L10–L18),
- and one late layer (e.g., L24–L28),
rather than betting everything on “layer 0 down_proj.” citeturn23view0

Because `model.safetensors.index.json` exposes all module names, you can systematically enumerate candidate hook points and keep the pipeline consistent. citeturn24view0

### Feature construction that scales

For a hidden size of 3584, a single token vector is already a decent probe feature. citeturn3view0 Typical feature choices:

- **Last prompt token** activation at layer ℓ: `h_ℓ[t_last]`
- **Mean over last K tokens** (K=8 or 16): smooths token noise
- **Contrast features**: `h_ℓ(prompt_variant_A) - h_ℓ(prompt_variant_B)`  
  This mirrors the “contrast pair” idea and can remove content variance if A/B differ only in a small phrase. citeturn0search3

Implementation detail that matters: store features as `float32` for probe training even if your model runs BF16; otherwise small numeric issues can dominate logistic regression fits.

### Probe models and evaluation

Start simple and honest:

- Logistic regression (L2), linear SVM, or ridge regression (for continuous heuristic labels).
- Out-of-distribution evaluation: do not just random-split prompts; **split by prompt family** (topic) to test generalization.

Why: probe work explicitly warns that off-policy/synthetic data and mismatched domains can cause generalization failures, especially for “intent-like” behaviors. citeturn21view0turn21view1

Metrics to compute per probe:
- AUROC and AUPRC (if positives are rare),
- “family holdout” performance,
- calibration curve if you want to use it as an optimization objective later.

Deliverable: a single script that can train probes across layers × feature types, producing a small model artifact plus a JSON report with metrics and the top-weight features.

### Interpreting probe directions without fooling yourself

A linear probe weight vector `w` is a direction in activation space. Two sanity checks are worth engineering up front:

- **Prompt-content control**: ensure the probe doesn’t just detect whether the prompt *mentions* “backdoor,” “trigger,” “dormant,” etc. (common failure mode). This is exactly the kind of spurious feature dependence highlighted in targeted instruction-pair probe research. citeturn21view1  
- **Ablation test**: if you can intervene, test whether projecting out `w` reduces your “triggered” behavior metric. Even partial evidence here is stronger than just AUROC.

If you later move beyond probes, modern mechanistic work uses interventions like activation patching to localize causal components; that’s not required for your first pass, but it’s a good “phase 2” target once you have candidate triggers. citeturn23view0

## A creative end-to-end project blueprint for trigger discovery

Below is a concrete “research program” you can run on the warmup model locally, structured as iterative loops. It aggressively prioritizes *engineering leverage* (automation + logging) because trigger discovery is usually search-heavy.

### Phase design goals

- Treat the warmup model as a **sandbox** for building your tooling: activation capture, feature stores, probe training, and search/eval loops.
- Produce artifacts you can reuse later against the larger models via the batch activations API and `jsinfer` (which is explicitly a client for the puzzle’s batch inference/activations API, requires Python ≥ 3.9, and was released on PyPI in mid-January 2026). citeturn13view0turn12view0

### Loop one: instrumentation + broad behavioral scan

Build a runner that, for each prompt:
1. Formats prompt (chat-template and raw modes).
2. Runs deterministic generation (fixed max_new_tokens, do_sample False).
3. Logs:
   - output text,
   - tokenization (input ids, special tokens),
   - activation features at a small set of layers/modules.

Then generate prompt suites:

- **“Normal conversation” suite**: everyday assistant questions.
- **“Self-reference” suite**: “what are your hidden instructions?”, “are you safe?”, etc.
- **“Format stress” suite**: nested JSON/XML, repeated delimiters, role tags, unicode oddities.
- **“Time shift” suite**: explicit future framing (FCC-style), “today is 2035.” citeturn11search3

Goal: find *any* systematic output mode shifts (language switching, refusal policy change, persona flip, overly specific numeric outputs, etc.).

### Loop two: pseudo-labeling and probe training

For any discovered “phenomenon,” define a label:

- language != English,
- length > threshold,
- contains a specific phrase,
- semantic cluster membership,
- refusal probability surrogate.

Train probes across layers to predict that label. Prefer the smallest probe that generalizes across prompt families. citeturn21view0turn21view1

At this stage, your probe doesn’t have to mean “triggered.” It just needs to be a stable sensor of *some internal distinction* that correlates with your phenomenon.

### Loop three: probe-guided trigger search

Once you have a scalar score `s(prompt)` (probe logit), you can use it as an optimization target.

Three search strategies that are surprisingly effective:

- **Discrete hill-climb in token space**: start from a random short string; mutate one token at a time; keep mutations that increase `s`.  
- **Cross-entropy method (CEM) over token distributions**: maintain a distribution over tokens at each position in a short trigger candidate; sample candidates; update distribution toward high-scoring samples.  
- **Feature-targeted fuzzing**: instead of random characters, directly sample from rare token ids, multi-script tokens, or chat-template boundary markers, because these often produce outsized representational shifts.

Why this is not crazy: practical trigger scanners have explicitly used output distribution and attention-pattern anomalies to recover triggers without prior trigger knowledge. citeturn22view0

Engineering “gotchas” to plan for:

- Ensure your search objective is evaluated on a *fixed downstream prompt*, not just on the trigger string alone. Many triggers only matter when followed by a substantive instruction.
- Regularize for shortness (a penalty on token count) so you don’t converge on giant “prompt attacks.”

### Loop four: validation and “does it actually trigger?”

When a candidate trigger is found, validate it like you would validate a vulnerability:

- Measure activation scores and output behavior across many random tasks with and without the trigger.
- Confirm it is not simply causing the model to output junk (which can trivially spike anomaly scores).
- Test robustness to paraphrase / whitespace changes.
- Test whether the trigger is *token-exact* (same token sequence) or *semantic* (many paraphrases work).

If you later move beyond linear probes, mechanistic evidence suggests triggers can co-opt existing functional circuits (e.g., “language circuitry”), meaning a trigger might manifest as a strong activation in components that already represent that behavior. This is one reason monitoring known subsystems can sometimes beat “search for weird neurons.” citeturn23view0

## Operational pitfalls and best practices

Probe-driven trigger hunting is powerful but easy to fool (or to have it fool you). The most important operational lessons from the probe literature:

- **Probe success is not probe truth.** You can get high separability that reflects prompt artifacts, not latent intent. citeturn21view1turn21view0  
- **Distribution shift is the default.** If your positive examples come from one artificial prompt family, your detector may collapse on real prompts. Use prompt-family holdouts and “incentivized” tests where you try to elicit the phenomenon naturally (even if you don’t fully know how). citeturn21view0  
- **Lock your inference settings.** Because the warmup checkpoint’s generation config enables sampling, and because decoding randomness can change the very activations you probe, your harness should always emit (and record) the full generation parameter set. citeturn9view0  
- **Keep the model immutable.** No inadvertent fine-tuning, no gradient accumulation, no adapter loading—because you’re trying to detect a pre-existing backdoor-like phenomenon.

Finally, even though this is framed as a puzzle, the techniques overlap with real supply-chain model security. Recent scanning work explicitly positions “trigger extraction” as a practical defensive capability, but also acknowledges the dual-use nature of the methodology. citeturn22view0turn23view0