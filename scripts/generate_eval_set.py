"""Build a stratified, citation-grounded retrieval evaluation set."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.generate_dataset import FAMILY_TITLES, SCOPE_TITLES
    from scripts.stage1_common import (
        artifact_entry,
        atomic_write_json,
        jsonl_rows,
        load_spec,
        resolve_path,
        verify_manifest_artifacts,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution places scripts/, not the project root, on sys.path.
    from generate_dataset import FAMILY_TITLES, SCOPE_TITLES
    from stage1_common import (
        artifact_entry,
        atomic_write_json,
        jsonl_rows,
        load_spec,
        resolve_path,
        verify_manifest_artifacts,
        write_jsonl,
    )


TRAP_RULE_TYPES = {
    "version_conflict": "revenue_status",
    "boundary_exception": "refund_boundary",
    "negation_lexical_decoy": "platform_subsidy_exclusion",
    "unanswerable": "reconciliation_tolerance",
}


def _topic_label(chunk: dict[str, Any]) -> str:
    family = FAMILY_TITLES.get(chunk["policy_family"], chunk["policy_family"])
    scope = SCOPE_TITLES.get(chunk["scope"], chunk["scope"])
    return f"{family}/{scope}"


def _candidates(chunks: list[dict[str, Any]], rule_type: str, version: str = "V2") -> list[dict[str, Any]]:
    seen_policy_keys: set[str] = set()
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk["rule_type"] != rule_type or chunk["version"] != version:
            continue
        policy_key = chunk["policy_key"]
        if policy_key not in seen_policy_keys:
            seen_policy_keys.add(policy_key)
            result.append(chunk)
    return result


def _counterpart(chunks_by_key: dict[tuple[str, str, str], dict[str, Any]], chunk: dict[str, Any], version: str, rule_type: str) -> dict[str, Any]:
    key = (chunk["policy_key"], version, rule_type)
    try:
        return chunks_by_key[key]
    except KeyError as exc:
        raise ValueError(f"Required hard-negative chunk is missing: {key}") from exc


def _build_question(chunk: dict[str, Any], trap_type: str, index: int, chunks_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    topic = _topic_label(chunk)
    if trap_type == "version_conflict":
        match = re.search(r"仅将状态为\s*(.+?)\s*的订单纳入营收", chunk["text"])
        if not match:
            raise ValueError(f"Could not extract revenue statuses from {chunk['chunk_id']}")
        answer_statuses = match.group(1)
        old_chunk = _counterpart(chunks_by_key, chunk, "V1", "revenue_status")
        return {
            "trap_type": trap_type,
            "query": f"当前生效口径下，{topic}中哪些订单状态可以计入营收？",
            "expected_answer": f"仅状态为 {answer_statuses} 的订单计入营收；以当前生效的 V2 条款为准。",
            "must_abstain": False,
            "gold_doc_ids": [chunk["doc_id"]],
            "gold_chunk_ids": [chunk["chunk_id"]],
            "hard_negative_chunk_ids": [old_chunk["chunk_id"]],
            "rationale": "V1 与 V2 对营收状态有冲突；查询当前口径时应依据已生效的 V2。",
        }

    if trap_type == "boundary_exception":
        match = re.search(r"签收后\s*(\d+)\s*个自然日内", chunk["text"])
        if not match:
            raise ValueError(f"Could not extract refund-window boundary from {chunk['chunk_id']}")
        days = int(match.group(1))
        exception = _counterpart(chunks_by_key, chunk, "V2", "refund_exception")
        return {
            "trap_type": trap_type,
            "query": f"{topic}的退货申请在签收后的第 {days} 个自然日提交，仍属于常规受理窗口吗？",
            "expected_answer": f"属于；第 {days} 个自然日当日包含在窗口内，超过期限才需要人工例外审批。",
            "must_abstain": False,
            "gold_doc_ids": [chunk["doc_id"]],
            "gold_chunk_ids": [chunk["chunk_id"]],
            "hard_negative_chunk_ids": [exception["chunk_id"]],
            "rationale": "答案取决于“第 N 日当日仍在窗口内”的边界措辞；质量问题例外条款不能替代常规时限。",
        }

    if trap_type == "negation_lexical_decoy":
        decoy = _counterpart(chunks_by_key, chunk, "V2", "merchant_shipping_income")
        return {
            "trap_type": trap_type,
            "query": f"{topic}中，平台承担的运费补贴能否计入商家销售净额或买家实付金额？",
            "expected_answer": "不能。平台运费补贴不是买家支付款，不计入商家销售净额，也不写入 paid_amount_minor。",
            "must_abstain": False,
            "gold_doc_ids": [chunk["doc_id"]],
            "gold_chunk_ids": [chunk["chunk_id"]],
            "hard_negative_chunk_ids": [decoy["chunk_id"]],
            "rationale": "相邻规则提到买家支付的商家运费可进入实付金额，但该规则不适用于平台承担的补贴。",
        }

    if trap_type == "unanswerable":
        return {
            "trap_type": trap_type,
            "query": f"{topic}的商家评分至少要达到多少分才允许结算？",
            "expected_answer": "现有规则文档未提供商家评分门槛，无法据此判断。",
            "must_abstain": True,
            "gold_doc_ids": [],
            "gold_chunk_ids": [],
            "hard_negative_chunk_ids": [chunk["chunk_id"]],
            "rationale": "语料包含结算差异处理规则，但没有商家评分门槛；应拒绝臆测并说明缺少依据。",
        }

    raise ValueError(f"Unsupported trap type: {trap_type}")


def _build_eval_rows(spec: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluation = spec["evaluation"]
    taxonomy = spec["rule_documents"]["retrieval_traps"]["taxonomy"]
    trap_ids = [entry["id"] for entry in taxonomy]
    if set(trap_ids) != set(TRAP_RULE_TYPES):
        raise ValueError(f"Configured trap taxonomy must match supported types: {sorted(TRAP_RULE_TYPES)}")

    target_total = int(evaluation["target_qa_pairs"])
    split_counts = {key: int(value) for key, value in evaluation["split"].items()}
    if sum(split_counts.values()) != target_total:
        raise ValueError("Evaluation split sizes must sum to target_qa_pairs")
    if target_total < len(trap_ids):
        raise ValueError("target_qa_pairs must provide at least one question per trap type")

    index = {
        (chunk["policy_key"], chunk["version"], chunk["rule_type"]): chunk
        for chunk in chunks
    }
    rng = random.Random(int(spec["project"]["random_seed"]) + 2)
    base_count, remainder = divmod(target_total, len(trap_ids))
    counts_by_trap = {trap_id: base_count + (position < remainder) for position, trap_id in enumerate(trap_ids)}
    queues: dict[str, list[dict[str, Any]]] = {}
    for trap_id in trap_ids:
        candidates = _candidates(chunks, TRAP_RULE_TYPES[trap_id], "V2")
        rng.shuffle(candidates)
        required = counts_by_trap[trap_id]
        if len(candidates) < required:
            raise ValueError(f"Need {required} unique policy chunks for {trap_id}; found {len(candidates)}")
        queues[trap_id] = [
            _build_question(chunk, trap_id, position + 1, index)
            for position, chunk in enumerate(candidates[:required])
        ]

    rows_by_split: list[tuple[str, dict[str, Any]]] = []
    positions = {trap_id: 0 for trap_id in trap_ids}
    for split_name, quota in split_counts.items():
        remaining = quota
        while remaining:
            progressed = False
            for trap_id in trap_ids:
                if positions[trap_id] < len(queues[trap_id]) and remaining > 0:
                    item = queues[trap_id][positions[trap_id]]
                    positions[trap_id] += 1
                    rows_by_split.append((split_name, item))
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise ValueError(f"Not enough QA candidates to fill split {split_name}")

    rows: list[dict[str, Any]] = []
    for number, (split_name, item) in enumerate(rows_by_split, start=1):
        rows.append({"question_id": f"EVAL-{number:03d}", "split": split_name, **item})
    if len(rows) != target_total:
        raise AssertionError(f"Generated {len(rows)} QA pairs; expected {target_total}")
    return rows


def generate_eval_set(config_path: Path) -> dict[str, Any]:
    spec, root, _ = load_spec(Path(config_path))
    manifest_path = resolve_path(root, spec["artifacts"]["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generate the Stage 1 dataset first; manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_artifacts(manifest, root)

    rule_dir = resolve_path(root, spec["rule_documents"]["output"]["directory"])
    chunks_path = rule_dir / spec["rule_documents"]["output"]["chunks_filename"]
    chunks = jsonl_rows(chunks_path)
    expected_chunks = int(spec["rule_documents"]["target_chunks"])
    if len(chunks) != expected_chunks:
        raise ValueError(f"Expected {expected_chunks} rule chunks, found {len(chunks)}")

    output_dir = resolve_path(root, spec["evaluation"]["output"]["directory"])
    output_path = output_dir / spec["evaluation"]["output"]["filename"]
    if output_path.exists() and spec["artifacts"].get("fail_if_output_exists", True):
        raise FileExistsError(f"Refusing to overwrite existing evaluation set: {output_path}")

    rows = _build_eval_rows(spec, chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output_dir, prefix=".stage1-eval-", delete=False
    ) as temporary_file:
        staged_path = Path(temporary_file.name)
    try:
        write_jsonl(staged_path, rows)
        if output_path.exists() and spec["artifacts"].get("fail_if_output_exists", True):
            raise FileExistsError(f"Evaluation set appeared during generation: {output_path}")
        os.replace(staged_path, output_path)
    finally:
        staged_path.unlink(missing_ok=True)

    manifest["artifacts"].append(artifact_entry(output_path, root))
    manifest["evaluation"] = {
        "qa_pairs": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in spec["evaluation"]["split"]
        },
        "trap_counts": {
            trap: sum(row["trap_type"] == trap for row in rows)
            for trap in TRAP_RULE_TYPES
        },
        "must_abstain": sum(row["must_abstain"] for row in rows),
    }
    atomic_write_json(manifest_path, manifest)
    return {
        **manifest["evaluation"],
        "eval_path": str(output_path),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/data_spec.yaml"))
    args = parser.parse_args()
    print(json.dumps(generate_eval_set(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
