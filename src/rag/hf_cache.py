"""HuggingFace 本地缓存探测与离线开关。

为什么需要这个模块
------------------
`sentence_transformers` / `transformers` 在加载模型时会向 huggingface.co 发
一次 API 请求去检查文件是否更新。在**没有外网**（或代理没开）的机器上这个请求
会失败，而 `transformers` 5.x 此时抛的是：

    ValueError: Unrecognized processing class in BAAI/bge-small-zh-v1.5.
    Can't instantiate a processor, a tokenizer, ...

注意它抛的是 `ValueError`，而 `build_embedder()` / `build_reranker()` 里守护降级
路径的 except 只捕获 `(ImportError, RuntimeError, OSError)` —— 于是这个异常会直接
穿透，把整个检索器构造打挂，**哪怕模型其实好好躺在 `models/huggingface/` 里**。

`enable_offline_if_cached()` 的作用：只有当目标模型的快照**确实已经在本地**时，
才把进程切成离线模式。冷机器仍然正常联网下载，热机器则完全不碰网络、也就不会
触发上面那个异常。这样既修好了崩溃，又保住了「确定性、可离线复现」。
"""

from __future__ import annotations

import os
from pathlib import Path


def _snapshots_dir(cache_folder: str | None, model_name: str) -> Path | None:
    """把 `BAAI/bge-small-zh-v1.5` 映射成 HF 的 `models--BAAI--bge-small-zh-v1.5`。"""
    if not cache_folder:
        return None
    repo_dir = "models--" + model_name.replace("/", "--")
    return Path(cache_folder) / repo_dir / "snapshots"


def is_model_cached(cache_folder: str | None, model_name: str) -> bool:
    """快照目录里至少有一个非空的 commit 目录才算「已缓存」。"""
    snapshots = _snapshots_dir(cache_folder, model_name)
    if snapshots is None or not snapshots.is_dir():
        return False
    for commit in snapshots.iterdir():
        if commit.is_dir() and any(commit.iterdir()):
            return True
    return False


def enable_offline_if_cached(cache_folder: str | None, model_name: str) -> bool:
    """模型已在本地则切离线模式。返回是否处于离线模式。

    必须在 `from sentence_transformers import ...` 之前调用，因为离线标志是
    在模型加载时读取的。
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return True
    if not is_model_cached(cache_folder, model_name):
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return True