"""Explicit, checkpoint-persisted attention semantics for OpenVLA-OFT.

The standard and OFT-fork Transformers packages can report the same version
while implementing different SDPA masks. Validate behavior before loading 7B
weights; never infer causality from a version or a CausalLM class name.
"""

import hashlib
import inspect
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

MODES = {"causal", "bidirectional"}
METADATA_FILE = "attention_config.json"


def resolve_attention_mode(checkpoint, requested: Optional[str] = None) -> str:
    """Require agreement between explicit choices and saved checkpoint metadata."""
    choices = {"argument": requested, "OPENVLA_ATTENTION_MODE": os.getenv("OPENVLA_ATTENTION_MODE")}
    path = Path(checkpoint)
    if path.is_dir():
        config_path = path / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        metadata_path = path / METADATA_FILE
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata and metadata.get("schema_version") != 1:
            raise ValueError(f"Unsupported attention metadata schema: {metadata_path}")
    else:
        from transformers import AutoConfig

        # The portable fields in config.json also travel with Hub checkpoints.
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True).to_dict()
        metadata = {}
    choices["checkpoint config"] = config.get("openvla_attention_mode")
    choices["checkpoint sidecar"] = metadata.get("mode")
    for backend in (config.get("openvla_attention_backend"), metadata.get("backend")):
        if backend is not None and backend != "sdpa":
            raise ValueError(f"Unsupported checkpoint attention backend {backend!r}; expected sdpa")
    present = {name: value for name, value in choices.items() if value is not None}
    if not present:
        raise ValueError(
            "Legacy checkpoint has no attention metadata. Explicitly set --attention_mode "
            "or OPENVLA_ATTENTION_MODE to causal or bidirectional, matching its training setup."
        )
    if any(value not in MODES for value in present.values()):
        raise ValueError(f"Attention mode must be causal or bidirectional: {present}")
    if len(set(present.values())) != 1:
        raise ValueError(f"Conflicting attention modes; refusing to change checkpoint semantics: {present}")
    return next(iter(present.values()))


@lru_cache(maxsize=1)
def probe_sdpa_semantics():
    """Perturb only a future token in the installed original Llama implementation.

    Run on CPU, outside timed inference, without changing CPU or CUDA RNG state.
    Both training and evaluation must agree. No policy attention is replaced.
    """
    import torch
    import transformers
    from transformers import LlamaConfig, LlamaModel

    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA; use the CPU generator directly.
        torch.random.default_generator.manual_seed(1729)
        config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=8,
                             attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = LlamaModel(config).to(device="cpu", dtype=torch.float32)
        tokens = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5]])
        deltas = {}
        for training in (False, True):
            model.train(training)
            with torch.no_grad():
                hidden = model(tokens, use_cache=False, output_attentions=False).last_hidden_state
            delta = float((hidden[0, :3] - hidden[1, :3]).abs().max())
            if not torch.isfinite(hidden).all():
                raise RuntimeError("Non-finite output in attention preflight")
            deltas["train" if training else "eval"] = delta
        semantics = {"causal" if delta <= 1e-7 else "bidirectional" for delta in deltas.values()}
        if len(semantics) != 1:
            raise RuntimeError(f"Training/evaluation attention semantics differ: {deltas}")
        source = Path(inspect.getfile(type(model.layers[0].self_attn)))
    return dict(mode=semantics.pop(), backend="sdpa", transformers_version=transformers.__version__,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), prefix_deltas=deltas)


def validate_attention_runtime(mode: str):
    if mode not in MODES:
        raise ValueError(f"Unknown attention mode: {mode}")
    observed = probe_sdpa_semantics()
    if observed["mode"] != mode:
        raise RuntimeError(
            f"Attention mismatch: requested {mode}, installed Transformers "
            f"{observed['transformers_version']} actually uses {observed['mode']} SDPA. "
            "Use standard transformers==4.40.1 for causal, or the explicitly selected OFT fork "
            "for bidirectional. See ATTENTION.md; version numbers alone are insufficient."
        )
    return observed


def stamp_attention_config(config, mode: str):
    """Use public custom fields: HF does not persist its private backend selector."""
    config.openvla_attention_mode = mode
    config.openvla_attention_backend = "sdpa"


def save_attention_metadata(directory, mode: str):
    """Save beside both adapters and merged checkpoints, including provenance."""
    observed = validate_attention_runtime(mode)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / METADATA_FILE).write_text(json.dumps(dict(schema_version=1, **observed), indent=2) + "\n")
    config_path = path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config.update(openvla_attention_mode=mode, openvla_attention_backend="sdpa")
        config_path.write_text(json.dumps(config, indent=2) + "\n")


def load_vla_with_attention(checkpoint, attention_mode=None, attention_checkpoint=None, **kwargs):
    """Shared training/inference/merge loader with an untimed preflight.

    Offline merging takes semantics from the fine-tuned adapter checkpoint,
    which may deliberately differ from the original pretrained base model.
    """
    from transformers import AutoModelForVision2Seq
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention

    mode = resolve_attention_mode(attention_checkpoint or checkpoint, attention_mode)
    validate_attention_runtime(mode)
    backend = kwargs.pop("attn_implementation", "sdpa")
    if backend != "sdpa":
        raise ValueError(f"The validated attention path requires sdpa, got {backend!r}")
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, attn_implementation="sdpa", **kwargs)
    llm = model.language_model
    if llm.config._attn_implementation != "sdpa" or not all(
        type(layer.self_attn) is LlamaSdpaAttention for layer in llm.model.layers
    ):
        raise RuntimeError("Loaded model does not use the validated Llama SDPA implementation")
    if model.config.output_attentions or llm.config.output_attentions:
        raise ValueError("output_attentions=True can bypass SDPA; disable it for the validated attention path")
    stamp_attention_config(model.config, mode)
    print(f"OpenVLA attention verified: {mode}, backend=sdpa")
    return model
