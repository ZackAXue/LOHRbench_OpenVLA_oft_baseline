# Attention configuration

Attention semantics are part of a checkpoint's configuration. Standard
Transformers 4.40.1 uses causal Llama SDPA; the OFT Transformers fork can report
the **same version** while using bidirectional SDPA. `CausalLM`, `sdpa`, and the
presence of a FlashAttention package do not establish which mask is used.

The shared training, inference, deployment and LoRA-merge loader now:

1. Requires an explicit mode for legacy checkpoints without attention metadata.
2. Rejects disagreements between the argument, `OPENVLA_ATTENTION_MODE` and saved
   checkpoint metadata.
3. Checks the installed original Llama implementation on a tiny CPU model by
   changing a future token. Both train and eval must have the requested semantics.
   This runs before loading the large weights, preserves RNG state and is cached
   per process; it is outside policy inference timing.
4. Explicitly loads `attn_implementation="sdpa"` and checks every loaded Llama
   attention class. An incompatible installation fails instead of silently
   changing the policy's attention.
5. Saves `attention_config.json` beside adapters and full checkpoints. Full
   `config.json` also stores `openvla_attention_mode` and
   `openvla_attention_backend`, so these settings travel to Hugging Face Hub.

The loader checks the supported SDPA path. Do not use `output_attentions=True`:
Transformers can fall back to eager attention, which changes the OFT fork's
semantics. Direct third-party `from_pretrained` calls that bypass the shared
loader do not get this preflight.

## Causal setup

The default dependency is now the standard PyPI `transformers==4.40.1`.
To replace an already-installed fork (whose version may look identical):

```bash
python -m pip install --force-reinstall --no-deps transformers==4.40.1
export OPENVLA_ATTENTION_MODE=causal
```

Alternatively pass `--attention_mode causal` to `finetune.py`, `eval_overfit.py`,
`merge_lora_weights_and_save.py`, `deploy.py`, or the LIBERO/ALOHA evaluators.
Existing wrappers using `get_vla(cfg)` can supply `cfg.attention_mode` or use the
environment variable. New checkpoints read their saved mode automatically;
an explicit conflicting mode raises an error.

Standalone `flash-attn` is not required for SDPA. PyTorch chooses an available
SDPA kernel, including its built-in FlashAttention CUDA kernel. Kernel selection
and causal versus bidirectional masking are separate settings. Use a PyTorch
build compatible with your GPU; installing standalone FlashAttention does not
repair a wrong mask.

The LoHRbench repackage evaluation of the supplied **premerged 200000** checkpoint
worked with causal SDPA, 256×256 base/wrist observations, 7D delta end-effector
control (`pd_ee_delta_pose`), chunk length 8, center crop and no proprioception.
Load that complete checkpoint without reapplying its LoRA adapter. These are
settings for this checkpoint, not defaults for every OFT model. The unavailable
historical training environment cannot be established from the weights alone.

## Explicit bidirectional compatibility setup

For a checkpoint known to require the original OFT bidirectional behavior, use
an isolated environment and explicitly install the inspected fork revision:

```bash
python -m pip install --force-reinstall --no-deps \
  'transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab'
export OPENVLA_ATTENTION_MODE=bidirectional
```

The same behavioral preflight applies. There is no automatic attention-mode
fallback. During offline merging, attention metadata comes from the fine-tuned
checkpoint, rather than assuming the original base model's attention mode.

## Regression checks

```bash
python -m unittest discover -s tests -v
# In the explicitly installed bidirectional compatibility environment:
TEST_EXPECT_ATTENTION=bidirectional python -m unittest discover -s tests -v
```

The tests cover actual future-token behavior in train/eval, RNG preservation,
metadata persistence/conflicts, rejecting the wrong dependency before loading
weights, and the shared loader's explicit SDPA forwarding and merge semantics.
