"""CPU checks that training and the independently packaged runtime agree."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

import ser_runtime as runtime
import train_qwen2audio_directhead_student_valonly_abc as training
import train_qwen2audio_mvef_student_valonly_abc as common
import select_qwen2audio_nested_calibration_dialogue_valonly_abc as calibration


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.model = FakeEncoder()
        self.lm_head = nn.Linear(8, 20, bias=False)

    def get_base_model(self):
        return self


class ReleaseContracts(unittest.TestCase):
    def test_four_pool_logits_match_training_with_nonzero_residual(self):
        torch.manual_seed(7)
        token_ids = torch.tensor([1, 2, 3])
        trained = training.DirectAudioTextClassifier(
            FakeBackbone(), token_ids, 9, initialize_from_lm_head=True,
        )
        with torch.no_grad():
            trained.output.weight.normal_()
            trained.output.bias.normal_()
        deployed = runtime.DirectAudioTextClassifier(FakeBackbone(), token_ids, 9)
        deployed.load_state_dict(trained.state_dict())
        trained.eval()
        deployed.eval()
        batch = {
            "input_ids": torch.tensor([[4, 9, 9, 5, 6], [7, 9, 8, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        }
        torch.testing.assert_close(trained(batch), deployed(batch), rtol=0, atol=0)

    def test_context_and_prompt_match_training_for_shuffled_rows(self):
        rows = [
            {"wav_name": f"call_right_{number}.wav", "segment": f"right_{number}",
             "dialogue_key": "call", "source_index": index, "ASR": f"示例{number}",
             "source_audio_path": f"/synthetic/call_right_{number}.wav"}
            for index, number in enumerate([3, 1, 2])
        ]
        a, b = [r.copy() for r in rows], [r.copy() for r in rows]
        common.add_context(a)
        runtime.add_context(b)
        self.assertEqual(a, b)

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                return json.dumps(conversation, ensure_ascii=False), kwargs

        for left, right in zip(a, b):
            self.assertEqual(
                training.make_directhead_prompt(Processor(), left, True),
                runtime.make_prompt(Processor(), right),
            )
        self.assertEqual(a[2]["previous_asr"], "示例1")
        self.assertEqual(a[2]["next_asr"], "示例3")

    def test_frozen_smoothing_matches_validation_and_preserves_speaker_groups(self):
        rng = np.random.default_rng(42)
        probabilities = rng.dirichlet([2, 3, 1], size=12)
        rows = [
            {"dialogue_key": f"call{index // 6}",
             "wav_name": f"call{index // 6}_{'left' if index % 6 < 3 else 'right'}_{index % 3}.wav",
             "segment": f"{'left' if index % 6 < 3 else 'right'}_{index % 3}"}
            for index in range(12)
        ]
        record = json.loads((ROOT / "models/dace/dace_frozen_parameters.json").read_text())
        smoother = record["deployment_candidate_fit_on_all_validation"]["smoother"]
        p = smoother["parameters"]
        candidate = tuple(p[key] for key in (
            "bias_a", "bias_c", "temperature", "alpha", "group_min", "max_margin", "mode",
        ))
        expected = calibration.transform(
            probabilities, [r["dialogue_key"] for r in rows],
            [f"{r['dialogue_key']}::{runtime.side_order(r)[0]}" for r in rows], candidate,
        )
        actual = runtime.dialogue_smooth(probabilities, rows, smoother)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_published_artifact_and_training_records_are_consistent(self):
        artifacts = json.loads((ROOT / "MODEL_ARTIFACTS.json").read_text())["artifacts"]
        audit = json.loads((ROOT / "benchmarks/training_data_audit.json").read_text())
        self.assertEqual(len(artifacts), 4)
        self.assertTrue(all(audit["training_input_hashes_match_manifest"].values()))
        self.assertTrue(audit["teacher"]["matches_saved_training_manifest"])
        self.assertEqual(audit["teacher"]["correct_argmax_with_kd"], 19277)
        for weight in audit["final_weights"]:
            path = f"models/shared/{weight['adapter']}_adapter/{weight['file']}"
            artifact = next(item for item in artifacts if item["path"] == path)
            self.assertEqual(artifact["sha256"], weight["sha256"])
            self.assertTrue(weight["matches_modelscope_release"])


if __name__ == "__main__":
    unittest.main()
