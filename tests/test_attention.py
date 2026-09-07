"""CPU regression checks against the actual installed Transformers attention.

Run with unittest discover. TEST_EXPECT_ATTENTION=bidirectional also exercises
the original OFT fork; its version number is identical to the standard wheel.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForVision2Seq, LlamaConfig, LlamaForCausalLM

# Avoid importing the unrelated vision/RLDS training stack for this CPU test.
source = Path(__file__).resolve().parents[1] / "prismatic/util/attention.py"
spec = importlib.util.spec_from_file_location("attention_under_test", source)
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)
EXPECTED = os.getenv("TEST_EXPECT_ATTENTION", "causal")
OTHER = "bidirectional" if EXPECTED == "causal" else "causal"


class AttentionContractTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        os.environ.pop("OPENVLA_ATTENTION_MODE", None)
        self.temp = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.environment.stop()

    def test_legacy_requires_explicit_semantics(self):
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            attention.resolve_attention_mode(self.checkpoint)
        self.assertEqual(attention.resolve_attention_mode(self.checkpoint, EXPECTED), EXPECTED)
        os.environ["OPENVLA_ATTENTION_MODE"] = EXPECTED
        self.assertEqual(attention.resolve_attention_mode(self.checkpoint), EXPECTED)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            attention.resolve_attention_mode(self.checkpoint, OTHER)

    def test_real_future_token_probe_preserves_rng_and_checks_both_phases(self):
        attention.probe_sdpa_semantics.cache_clear()
        before = torch.random.get_rng_state().clone()
        observed = attention.validate_attention_runtime(EXPECTED)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(observed["prefix_deltas"]), {"train", "eval"})
        for delta in observed["prefix_deltas"].values():
            if EXPECTED == "causal":
                self.assertEqual(delta, 0.0)
            else:
                self.assertGreater(delta, 1e-4)
        with self.assertRaisesRegex(RuntimeError, "Attention mismatch"):
            attention.validate_attention_runtime(OTHER)

    def test_metadata_survives_full_model_config_serialization(self):
        config = LlamaConfig()
        attention.stamp_attention_config(config, EXPECTED)
        config.save_pretrained(self.checkpoint)
        attention.save_attention_metadata(self.checkpoint, EXPECTED)
        self.assertEqual(attention.resolve_attention_mode(self.checkpoint), EXPECTED)
        reloaded = LlamaConfig.from_pretrained(self.checkpoint)
        self.assertEqual(reloaded.openvla_attention_mode, EXPECTED)
        self.assertEqual(reloaded.openvla_attention_backend, "sdpa")
        (self.checkpoint / attention.METADATA_FILE).unlink()
        self.assertEqual(attention.resolve_attention_mode(self.checkpoint), EXPECTED)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            attention.resolve_attention_mode(self.checkpoint, OTHER)

    def test_adapter_sidecar_and_conflicting_metadata(self):
        attention.save_attention_metadata(self.checkpoint, EXPECTED)
        self.assertEqual(attention.resolve_attention_mode(self.checkpoint), EXPECTED)
        (self.checkpoint / "config.json").write_text(json.dumps({"openvla_attention_mode": OTHER}))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            attention.resolve_attention_mode(self.checkpoint)

    def test_invalid_modes_backends_and_schemas_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be"):
            attention.resolve_attention_mode(self.checkpoint, "automatic")
        for metadata, message in [
            ({"schema_version": 99, "mode": EXPECTED}, "schema"),
            ({"schema_version": 1, "mode": EXPECTED, "backend": "flash_attention_2"}, "backend"),
        ]:
            (self.checkpoint / attention.METADATA_FILE).write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, message):
                attention.resolve_attention_mode(self.checkpoint)

    def test_wrong_dependency_fails_before_large_weight_loading(self):
        with patch.object(AutoModelForVision2Seq, "from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "Attention mismatch"):
                attention.load_vla_with_attention(self.checkpoint, attention_mode=OTHER)
            load.assert_not_called()

    def test_loader_forces_sdpa_and_uses_adapter_semantics_when_merging(self):
        config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        config._attn_implementation = "sdpa"
        llm = LlamaForCausalLM(config)
        model = SimpleNamespace(config=LlamaConfig(), language_model=llm)
        adapter = self.checkpoint / "adapter"
        attention.save_attention_metadata(adapter, EXPECTED)
        (self.checkpoint / "config.json").write_text(json.dumps({"openvla_attention_mode": OTHER}))
        with patch.object(AutoModelForVision2Seq, "from_pretrained", return_value=model) as load:
            result = attention.load_vla_with_attention(self.checkpoint, attention_checkpoint=adapter,
                                                       torch_dtype=torch.bfloat16)
            load.assert_called_once_with(self.checkpoint, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
            self.assertEqual(result.config.openvla_attention_mode, EXPECTED)
            with self.assertRaisesRegex(ValueError, "requires sdpa"):
                attention.load_vla_with_attention(adapter, attn_implementation="flash_attention_2")
            llm.config.output_attentions = True
            with self.assertRaisesRegex(ValueError, "bypass SDPA"):
                attention.load_vla_with_attention(adapter)


if __name__ == "__main__":
    unittest.main()
