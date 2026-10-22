"""回归测试：HF 缓存探测与离线开关。

背景：模型已经在 `models/huggingface/` 里，但断网时 `transformers` 5.x 会因为
请求 huggingface.co 失败而抛 `ValueError: Unrecognized processing class ...`，
而且抛的是 `ValueError` —— 不在 `build_embedder()` 的降级 except 列表里，于是
整个检索器构造被打挂。`enable_offline_if_cached()` 就是为了在「模型已在本地」
时提前切离线，绕开这次网络请求。
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.rag.hf_cache import _snapshots_dir, enable_offline_if_cached, is_model_cached

MODEL = "BAAI/bge-small-zh-v1.5"


def make_cache(root: Path, model: str, commit: str, files: tuple[str, ...]) -> Path:
    repo = root / ("models--" + model.replace("/", "--"))
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_text("x", encoding="utf-8")
    return snap


class OfflineFlagGuard(unittest.TestCase):
    """保证测试不会把进程的离线标志漏出去影响后面的用例。"""

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestSnapshotsDir(unittest.TestCase):
    def test_maps_model_id_to_hf_repo_dir(self) -> None:
        got = _snapshots_dir("models/huggingface", MODEL)
        self.assertEqual(got, Path("models/huggingface") / "models--BAAI--bge-small-zh-v1.5" / "snapshots")

    def test_none_cache_folder_returns_none(self) -> None:
        self.assertIsNone(_snapshots_dir(None, MODEL))


class TestIsModelCached(unittest.TestCase):
    def test_none_or_missing_cache_is_not_cached(self) -> None:
        self.assertFalse(is_model_cached(None, MODEL))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(is_model_cached(tmp, MODEL))

    def test_non_empty_snapshot_counts_as_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root, MODEL, "7999e1d3359715c523056ef9478215996d62a620", ("config.json", "model.safetensors"))
            self.assertTrue(is_model_cached(str(root), MODEL))

    def test_empty_snapshot_dir_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models--BAAI--bge-small-zh-v1.5" / "snapshots" / "abc").mkdir(parents=True)
            self.assertFalse(is_model_cached(str(root), MODEL))


class TestEnableOfflineIfCached(OfflineFlagGuard):
    def test_switches_offline_when_model_is_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root, MODEL, "deadbeef", ("config.json",))
            self.assertTrue(enable_offline_if_cached(str(root), MODEL))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_leaves_network_on_when_model_is_absent(self) -> None:
        """冷机器必须还能联网下载，不能被这个开关锁死。"""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(enable_offline_if_cached(tmp, MODEL))
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)
            self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)

    def test_absent_cache_folder_does_not_switch_offline(self) -> None:
        self.assertFalse(enable_offline_if_cached(None, MODEL))
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_existing_offline_env_is_respected_and_reported(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(enable_offline_if_cached(tmp, MODEL))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root, MODEL, "deadbeef", ("config.json",))
            enable_offline_if_cached(str(root), MODEL)
            before = dict(os.environ)
            self.assertTrue(enable_offline_if_cached(str(root), MODEL))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], before["HF_HUB_OFFLINE"])


if __name__ == "__main__":
    unittest.main()