"""Численная совместимость с checkpoint исходной плоской реализации."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import RUN_CONFIGS_DIR


class ModelCompatibilityTests(unittest.TestCase):
    def test_original_checkpoint_produces_same_logits(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/legacy_gpt.json").read_text())
        weights = {key: torch.tensor(value, dtype=torch.float32) for key, value in fixture["weights"].items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pt"
            torch.save({"model_config": fixture["model_config"], "model": weights}, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
            model = GPT(GPTConfig(**checkpoint["model_config"])).eval()
            model.load_state_dict(checkpoint["model"], strict=True)
            with torch.no_grad():
                actual, _ = model(torch.tensor(fixture["input_ids"]))
        torch.testing.assert_close(actual, torch.tensor(fixture["expected_logits"]), rtol=1e-5, atol=1e-6)

    def test_final_architecture_keeps_parameter_count(self):
        config = json.loads((RUN_CONFIGS_DIR / "arm_gpt_97m.json").read_text())
        with torch.device("meta"):
            model = GPT(GPTConfig(**config["model"]))
        self.assertEqual(model.num_parameters(), 97_241_856)
        self.assertEqual(asdict(model.config), config["model"])
