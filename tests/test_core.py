import ast
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t_aware.baselines import parse_judge_label
from t_aware.data import assert_disjoint, balance_humans, normalize, read_records, source_key
from t_aware.metrics import binary_metrics


class DataTests(unittest.TestCase):
    def test_high_temperature_is_ai_not_human(self):
        row = normalize({"generated_text": "AI sample", "generation_T": 3.0}, "test")
        self.assertEqual(row["label"], 1)

    def test_label_temperature_conflict_fails(self):
        with self.assertRaises(ValueError):
            normalize({"text": "sample", "label": 0, "generation_T": 1.2}, "test")

    def test_missing_label_never_becomes_human(self):
        with self.assertRaises(ValueError):
            normalize({"text": "sample"}, "test")

    def test_paired_source_ids_match(self):
        self.assertEqual(source_key({"id": "test_5_human"}), source_key({"id": "test_5_attacked"}))

    def test_whitespace_policy_and_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"text":"ok","label":1}\n{"text":"  ","label":0}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_records(path)
            rows, counts = read_records(path, "skip")
            self.assertEqual((len(rows), counts["empty_records"], counts["skipped_records"]), (1, 1, 1))
            rows, _ = read_records(path, "keep")
            self.assertEqual(len(rows), 2)
            path.write_text('{"text":"ok","label":1}\nBAD\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                read_records(path, "skip")

    def test_historical_balance(self):
        rows = [{"label": 1, "id": i} for i in range(13)] + [{"label": 0, "id": "human"}]
        balanced = balance_humans(rows)
        self.assertEqual(len(balanced), 26)
        self.assertEqual(sum(r["label"] == 0 for r in balanced), 13)

    def test_source_overlap_blocks_training(self):
        with self.assertRaisesRegex(ValueError, "shared source"):
            assert_disjoint([{"source_id": "one"}], [{"source_id": "one"}], "test")


class MetricsTests(unittest.TestCase):
    def test_f1_conventions_are_separate(self):
        value = binary_metrics([0, 0, 1, 1], [0, 0, 0, 1])
        self.assertEqual(value["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(value["f1_ai"], 2 / 3)
        self.assertAlmostEqual(value["f1_macro"], (2 / 3 + 4 / 5) / 2)

    def test_undefined_class_is_null(self):
        result = binary_metrics([1, 1], [0, 0])
        self.assertIsNone(result["human_recall"])
        self.assertIsNone(result["f1_macro"])

    def test_failure_cannot_be_a_label(self):
        with self.assertRaises(ValueError):
            binary_metrics([0], [None])
        for text in ("", "connection error", "maybe 1", "10", None):
            with self.assertRaises(ValueError):
                parse_judge_label(text)
        self.assertEqual(parse_judge_label(" 1\n"), 1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import RobertaConfig
        cls.torch = torch
        cls.cfg = RobertaConfig(vocab_size=48, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)

    def test_human_targets_do_not_change_loss(self):
        from t_aware.model import TemperatureAwareDetector
        torch = self.torch
        model = TemperatureAwareDetector(self.cfg, 10).eval()
        ids = torch.tensor([[0, 4, 5, 2], [0, 8, 9, 2]])
        labels = torch.tensor([0, 1])
        a = model(ids, torch.ones_like(ids), labels, torch.tensor([-1.0, 1.2]))
        b = model(ids, torch.ones_like(ids), labels, torch.tensor([999.0, 1.2]))
        self.assertTrue(torch.allclose(a["loss"], b["loss"]))

    def test_all_human_batch_is_finite_and_no_regression_gradient(self):
        from t_aware.model import TemperatureAwareDetector
        torch = self.torch
        model = TemperatureAwareDetector(self.cfg, 10)
        ids = torch.tensor([[0, 4, 5, 2]])
        result = model(ids, torch.ones_like(ids), torch.tensor([0]), torch.tensor([-1.0]))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["regression_loss"].item(), 0)
        self.assertEqual(model.regressor.weight.grad.abs().sum().item(), 0)

    def test_raw_and_normalized_targets(self):
        from t_aware.engine import TextDataset
        torch = self.torch
        tokenizer = lambda *args, **kwargs: {"input_ids": torch.tensor([[0, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        rows = [normalize({"text": "a", "generation_T": x}, "test") for x in (0.2, 3.5, -1.0)]
        dataset = TextDataset(rows, tokenizer, temperature_scale="minmax_0.2_3.5")
        self.assertAlmostEqual(dataset[0]["t_labels"].item(), 0)
        self.assertAlmostEqual(dataset[1]["t_labels"].item(), 1)
        self.assertAlmostEqual(dataset[2]["t_labels"].item(), -1)

    def test_checkpoint_round_trip_and_legacy_formula(self):
        from t_aware.model import TemperatureAwareDetector, load_checkpoint
        torch = self.torch
        model = TemperatureAwareDetector(self.cfg, 10).eval()
        ids = torch.tensor([[0, 4, 5, 2], [0, 8, 9, 2]])
        mask = torch.ones_like(ids)
        labels, targets = torch.tensor([0, 1]), torch.tensor([-1.0, 1.2])
        output = model(ids, mask, labels, targets)
        raw_reg = (output["temperature"] - targets).square()
        ai_mask = (targets >= 0).float()
        old_loss = torch.nn.functional.cross_entropy(output["logits"], labels) + 10 * (raw_reg * ai_mask).sum() / (ai_mask.sum() + 1e-8)
        self.assertTrue(torch.allclose(output["loss"], old_loss, atol=1e-7))
        with tempfile.TemporaryDirectory() as directory:
            self.cfg.save_pretrained(directory)
            path = Path(directory) / "model.pt"
            for zip_format in (True, False):
                with self.subTest(zip_format=zip_format):
                    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=zip_format)
                    restored = load_checkpoint(path, directory, 10).eval()
                    self.assertTrue(torch.allclose(output["logits"], restored(ids, mask)["logits"], atol=1e-7))


if __name__ == "__main__":
    unittest.main()
