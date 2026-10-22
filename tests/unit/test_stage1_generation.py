import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.generate_dataset import generate_dataset
from scripts.generate_eval_set import generate_eval_set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "config" / "data_spec.yaml"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage1GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name) / "first"
        cls.config_path = cls._make_config(cls.root)
        cls.dataset_summary = generate_dataset(cls.config_path)
        cls.eval_summary = generate_eval_set(cls.config_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @staticmethod
    def _make_config(root: Path) -> Path:
        spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        spec["orders"]["target_rows"] = 100
        config_dir = root / "config"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "data_spec.yaml"
        config_path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return config_path

    def test_order_count_dirty_share_and_single_anomaly_sidecar(self) -> None:
        orders_path = self.root / "data/generated/orders/orders.csv"
        label_path = self.root / "data/generated/orders/anomaly_labels.jsonl"
        with orders_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        labels = read_jsonl(label_path)

        self.assertEqual(len(rows), 100)
        self.assertEqual(len(labels), 6)
        self.assertEqual(len({label["order_id"] for label in labels}), 6)
        self.assertEqual({row["order_id"] for row in labels} <= {row["order_id"] for row in rows}, True)
        self.assertEqual(
            {rule: sum(label["anomaly_id"] == rule for label in labels) for rule in {x["anomaly_id"] for x in labels}},
            {rule: 1 for rule in {x["anomaly_id"] for x in labels}},
        )
        self.assertTrue(all(len(label["affected_fields"]) == 1 for label in labels))
        label_by_order = {label["order_id"]: label for label in labels}
        for row in rows:
            self.assertTrue(row["order_id"].startswith("ORD-"))
            for field in ("unit_price_minor", "discount_minor", "shipping_fee_minor", "paid_amount_minor", "refunded_amount_minor"):
                self.assertTrue(row[field].lstrip("-").isdigit())
            if row["order_id"] not in label_by_order:
                expected_paid = (
                    int(row["quantity"]) * int(row["unit_price_minor"])
                    - int(row["discount_minor"])
                    + int(row["shipping_fee_minor"])
                )
                self.assertEqual(int(row["paid_amount_minor"]), expected_paid)

    def test_rule_chunks_have_source_metadata_and_target_count(self) -> None:
        chunks = read_jsonl(self.root / "data/generated/rule_docs/chunks.jsonl")
        self.assertEqual(len(chunks), 5000)
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 5000)
        docs_dir = self.root / "data/generated/rule_docs/documents"
        self.assertEqual(len(list(docs_dir.glob("*.md"))), 250)
        for chunk in chunks:
            for key in ("doc_id", "version", "page", "line_start", "line_end", "chunk_id"):
                self.assertIn(key, chunk)
            self.assertIn(chunk["version"], {"V1", "V2"})
            self.assertTrue(chunk["text"].strip())
            document_lines = (docs_dir / f"{chunk['doc_id']}.md").read_text(encoding="utf-8").splitlines()
            self.assertEqual(document_lines[chunk["line_start"] - 1], chunk["text"])

    def test_eval_split_counts_and_gold_references(self) -> None:
        eval_path = self.root / "eval/datasets/stage1_eval.jsonl"
        items = read_jsonl(eval_path)
        chunks = read_jsonl(self.root / "data/generated/rule_docs/chunks.jsonl")
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        self.assertEqual(len(items), 120)
        self.assertEqual(
            {split: sum(item["split"] == split for item in items) for split in ("development", "validation", "holdout")},
            {"development": 60, "validation": 30, "holdout": 30},
        )
        self.assertEqual(
            {trap: sum(item["trap_type"] == trap for item in items) for trap in {x["trap_type"] for x in items}},
            {trap: 30 for trap in {x["trap_type"] for x in items}},
        )
        for item in items:
            self.assertTrue(set(item["gold_chunk_ids"]) <= chunk_ids)
            self.assertTrue(set(item["hard_negative_chunk_ids"]) <= chunk_ids)
            self.assertEqual(item["must_abstain"], not bool(item["gold_chunk_ids"]))
        self.assertEqual(sum(item["must_abstain"] for item in items), 30)

    def test_fixed_seed_reproduces_outputs_and_existing_data_is_not_overwritten(self) -> None:
        second_root = Path(self.temp_dir.name) / "second"
        second_config = self._make_config(second_root)
        generate_dataset(second_config)
        generate_eval_set(second_config)

        relative_paths = (
            "data/generated/orders/orders.csv",
            "data/generated/orders/anomaly_labels.jsonl",
            "data/generated/rule_docs/chunks.jsonl",
            "eval/datasets/stage1_eval.jsonl",
        )
        for relative_path in relative_paths:
            self.assertEqual(sha256(self.root / relative_path), sha256(second_root / relative_path))
        with self.assertRaises(FileExistsError):
            generate_dataset(self.config_path)
        with self.assertRaises(FileExistsError):
            generate_eval_set(self.config_path)


if __name__ == "__main__":
    unittest.main()
