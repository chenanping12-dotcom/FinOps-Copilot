"""Parent-child corpus construction with source-preserving metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import ChunkRecord


class ParentChildCorpus:
    """A chunk index that keeps retrieval children linked to page-level parents."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError("chunk_id values must be unique")

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> "ParentChildCorpus":
        raw_records = list(records)
        groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for raw in raw_records:
            groups[(str(raw["doc_id"]), int(raw["page"]))].append(raw)

        chunks: list[ChunkRecord] = []
        for raw in raw_records:
            page = int(raw["page"])
            doc_id = str(raw["doc_id"])
            parent_id = f"{doc_id}:P{page}"
            parent_text = " ".join(item["text"] for item in groups[(doc_id, page)])
            chunks.append(
                ChunkRecord(
                    chunk_id=str(raw["chunk_id"]),
                    doc_id=doc_id,
                    version=str(raw["version"]),
                    effective_from=str(raw.get("effective_from", "")),
                    page=page,
                    line_start=int(raw["line_start"]),
                    line_end=int(raw["line_end"]),
                    policy_key=str(raw["policy_key"]),
                    policy_family=str(raw["policy_family"]),
                    scope=str(raw["scope"]),
                    rule_type=str(raw["rule_type"]),
                    text=str(raw["text"]),
                    parent_id=parent_id,
                    parent_text=parent_text,
                )
            )
        return cls(chunks)

    @classmethod
    def from_jsonl(cls, path: Path) -> "ParentChildCorpus":
        with path.open(encoding="utf-8") as stream:
            return cls.from_records(json.loads(line) for line in stream if line.strip())

    def __len__(self) -> int:
        return len(self.chunks)
