"""Exercise real train/evaluate entry points with a tiny offline backbone."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TinyTokenizer:
    def __call__(self, text, max_length=8, **kwargs):
        import torch
        token = 4 + len(text) % 20
        return {"input_ids": torch.tensor([[0, token, token + 1, 2]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "tokenizer_config.json").write_text('{"test_fixture": true}')


class PipelineTests(unittest.TestCase):
    def test_judge_failure_is_missing_and_coverage_is_explicit(self):
        from t_aware.baselines import baseline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text('{"text":"human sample","label":0}\n{"text":"AI sample","label":1}\n')
            predictions = [{"id": "one", "label": 0, "prediction": None, "error_type": "ConnectionError"},
                           {"id": "two", "label": 1, "prediction": 1, "error_type": None}]
            args = argparse.Namespace(data=data, output=root / "judge.json", empty_text="error", attempts=1,
                method="llm", model="test-no-api", max_length=512, ai_output_label=None, reference_stats=None)
            with patch("t_aware.baselines._llm_predictions", return_value=predictions), \
                 self.assertRaisesRegex(ValueError, "Some judge requests failed"):
                baseline(args)
            result = json.loads(args.output.read_text())
            self.assertEqual(result["coverage"], 0.5)
            self.assertEqual(result["failed_records"], 1)
            self.assertIsNone(result["predictions"][0]["prediction"])
            self.assertIsNone(result["metrics_on_valid_records"]["human_recall"])

    def test_train_then_evaluate_without_reading_test_during_selection(self):
        from t_aware.cli import train, evaluate, benchmark
        from t_aware.model import TemperatureAwareDetector
        from transformers import RobertaConfig
        config = RobertaConfig(vocab_size=48, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "attack"):
                rows = [{"id": f"{split}_0", "base_id": f"{split}_0", "text": "human words", "generation_T": -1.0},
                        {"id": f"{split}_1", "base_id": f"{split}_1", "text": "AI words", "generation_T": 1.2}]
                (root / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            cfg = {"name": "offline-integration", "base_model": "unused-offline", "train_file": "train.jsonl",
                   "val_standard": "val.jsonl", "val_attack": "attack.jsonl", "test_file": "DOES_NOT_EXIST",
                   "empty_text": "error", "temperature_scale": "raw", "seed": 42, "epochs": 2,
                   "batch_size": 2, "max_length": 8, "learning_rate": 0.00002, "warmup_ratio": 0.1,
                   "lambdas": [0.0, 10.0]}
            cfg_path = root / "train.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            output = root / "run"
            def tiny_pretrained(base, weight):
                return TemperatureAwareDetector(config, weight)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                 patch.object(TemperatureAwareDetector, "pretrained", side_effect=tiny_pretrained):
                train(argparse.Namespace(config=cfg_path, data_root=root, output=output, device="cpu"))
            selected = json.loads((output / "selection.json").read_text())
            self.assertFalse(selected["test_evaluated"])
            self.assertEqual(len(selected["candidates"]), 2)
            checkpoint = output / selected["selected"]["checkpoint"]
            evaluation = root / "evaluation.json"
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()):
                evaluate(argparse.Namespace(data=root / "attack.jsonl", checkpoint=checkpoint, model_config=checkpoint.parent,
                    tokenizer="unused-offline", max_length=8, batch_size=2, device="cpu", empty_text="error", balance=False,
                    output=evaluation))
            result = json.loads(evaluation.read_text())
            self.assertEqual(result["metrics"]["n"], 2)
            self.assertEqual(len(result["checkpoint"]["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                evaluate(argparse.Namespace(output=evaluation))
            matrix = {"models": [{"name": "tiny", "checkpoint": str(checkpoint.relative_to(root)),
                                   "model_config": str((checkpoint.parent / "config.json").relative_to(root)),
                                   "tokenizer": str((output / "tokenizer").relative_to(root))}],
                      "datasets": [{"name": "toy", "path": "attack.jsonl", "protocol": "test", "empty_text": "error"}]}
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(matrix))
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()) as tokenizer_loader:
                benchmark(argparse.Namespace(config=matrix_path, data_root=root, output=root / "benchmark.json", device="cpu"))
            tokenizer_loader.assert_called_once_with(str(output / "tokenizer"))
            self.assertEqual(len(json.loads((root / "benchmark.json").read_text())["results"]), 1)


if __name__ == "__main__":
    unittest.main()
