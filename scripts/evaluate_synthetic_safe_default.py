from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_rag import (
    DEFAULT_CONFIG,
    Chunk,
    Document,
    QueryFlags,
    build_rag_index,
    build_sparse_index,
    detect_query_flags,
    decide_selective_multihop,
    normalize_for_search,
    retrieve,
    retrieve_chunks,
    score_retrieved_injection_spans,
    extract_bridge_candidates,
)
from tests.fixtures.synthetic_cases import SYNTHETIC_CASES


def make_chunk(case_id: str, doc_id: str, text: str) -> Chunk:
    risk, reasons, spans = score_retrieved_injection_spans(text)
    return Chunk(
        chunk_id=f"{case_id}_{doc_id}_c000",
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        page_start=1,
        page_end=1,
        section=None,
        raw_text=text,
        search_text=normalize_for_search(text),
        injection_risk=risk,
        injection_reasons=reasons,
        suspicious_spans=spans,
    )


def build_case_index(case: dict[str, Any]):
    chunks = [make_chunk(case["id"], doc["doc_id"], doc["text"]) for doc in case["documents"]]
    documents = [
        Document(
            doc_id=doc["doc_id"],
            filename=f'{doc["doc_id"]}.pdf',
            pages=[doc["text"]],
            metadata={},
            parse_status="ok",
        )
        for doc in case["documents"]
    ]
    return build_rag_index(documents, chunks, build_sparse_index(chunks), DEFAULT_CONFIG)


def run_checks(case: dict[str, Any]) -> dict[str, Any]:
    index = build_case_index(case)
    original_multihop = index.config.enable_selective_multihop
    index.config.enable_selective_multihop = False
    before_context = retrieve(case["question"], index)
    before_doc_ids = [
        chunk.doc_id
        for chunk in index.chunks
        if f"[DOC {chunk.filename}" in before_context
    ]
    index.config.enable_selective_multihop = original_multihop
    start = time.perf_counter()
    context = retrieve(case["question"], index)
    elapsed = time.perf_counter() - start
    flags = detect_query_flags(case["question"])
    checks = case["checks"]

    failures: list[str] = []
    expected_doc_ids = checks.get("expected_doc_ids") or [checks.get("expected_doc_id")]
    expected_doc_ids = [doc_id for doc_id in expected_doc_ids if doc_id]
    retrieved_doc_ids = [
        chunk.doc_id
        for chunk in index.chunks
        if f"[DOC {chunk.filename}" in context
    ]
    retrieval_hit = all(doc_id in retrieved_doc_ids for doc_id in expected_doc_ids)
    before_retrieval_hit = all(doc_id in before_doc_ids for doc_id in expected_doc_ids)
    first_pass_chunks = retrieve_chunks(case["question"], index, index.config.max_chunks)
    bridge_candidates = extract_bridge_candidates(first_pass_chunks)
    decision = decide_selective_multihop(flags, index.config, first_pass_chunks)
    project_candidate_count = sum(candidate.startswith("Project ") for candidate in bridge_candidates)
    team_candidate_count = sum(candidate.startswith("Team ") for candidate in bridge_candidates)

    if checks.get("required_entity") and checks["required_entity"] not in context:
        failures.append("missing_required_entity")
    for token in checks.get("required_number_tokens", []):
        if token not in context:
            failures.append(f"missing_number_token:{token}")
    if checks.get("required_evidence_keyword") and checks["required_evidence_keyword"] not in context:
        failures.append("missing_evidence_keyword")
    for token in checks.get("must_not_mask_tokens", []):
        if token not in context:
            failures.append(f"masked_required_token:{token}")
    if checks.get("must_neutralize_span") and "[REDACTED_UNTRUSTED_INSTRUCTION]" not in context:
        failures.append("missing_neutralization")
    if checks.get("must_not_refuse_query") and flags.asks_direct_pii:
        failures.append("unexpected_refusal_flag")
    if checks.get("must_flag_direct_pii") and not flags.asks_direct_pii:
        failures.append("missing_direct_pii_flag")
    if not retrieval_hit:
        failures.append("retrieval_miss")
    expected_second_pass = checks.get("expected_second_pass")
    if expected_second_pass == "skip" and decision.should_run_second_pass:
        failures.append("unexpected_second_pass_run")
    if expected_second_pass == "run" and not decision.should_run_second_pass:
        failures.append("unexpected_second_pass_skip")

    return {
        "case_id": case["id"],
        "category": case["category"],
        "passed": not failures,
        "failed_checks": failures,
        "retrieval_hit": retrieval_hit,
        "before_retrieval_hit": before_retrieval_hit,
        "context_length": len(context),
        "latency_seconds": elapsed,
        "masked_failure": any(item.startswith("masked_required_token") for item in failures),
        "neutralization_failure": "missing_neutralization" in failures,
        "pii_routing_failure": any(
            item in {"unexpected_refusal_flag", "missing_direct_pii_flag"} for item in failures
        ),
        "project_candidate_count": project_candidate_count,
        "team_candidate_count": team_candidate_count,
        "second_pass_executed": decision.should_run_second_pass,
        "second_pass_changed_outcome": retrieval_hit and not before_retrieval_hit,
        "false_skip": "unexpected_second_pass_skip" in failures,
        "false_run": "unexpected_second_pass_run" in failures,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)

    summary: dict[str, Any] = {}
    for category, items in grouped.items():
        summary[category] = {
            "total_cases": len(items),
            "passed_checks": sum(item["passed"] for item in items),
            "failed_checks": sum(not item["passed"] for item in items),
            "retrieval_hits": sum(item["retrieval_hit"] for item in items),
            "retrieval_misses": sum(not item["retrieval_hit"] for item in items),
            "before_retrieval_hits": sum(item["before_retrieval_hit"] for item in items),
            "before_retrieval_misses": sum(not item["before_retrieval_hit"] for item in items),
            "masking_failures": sum(item["masked_failure"] for item in items),
            "neutralization_failures": sum(item["neutralization_failure"] for item in items),
            "pii_routing_failures": sum(item["pii_routing_failure"] for item in items),
            "project_candidate_count": sum(item["project_candidate_count"] for item in items),
            "team_candidate_count": sum(item["team_candidate_count"] for item in items),
            "second_pass_execution_count": sum(item["second_pass_executed"] for item in items),
            "second_pass_execution_rate": round(
                sum(item["second_pass_executed"] for item in items) / len(items),
                4,
            ),
            "second_pass_changed_outcome_count": sum(
                item["second_pass_changed_outcome"] for item in items
            ),
            "false_skip_count": sum(item["false_skip"] for item in items),
            "false_run_count": sum(item["false_run"] for item in items),
            "average_context_length": round(statistics.mean(item["context_length"] for item in items), 2),
            "retrieval_context_pack_seconds": {
                "median": round(statistics.median(item["latency_seconds"] for item in items), 6),
                "p95": round(max(item["latency_seconds"] for item in items), 6),
            },
        }
    return summary


def main() -> None:
    results = [run_checks(case) for case in SYNTHETIC_CASES]
    report = {
        "case_count": len(results),
        "category_summary": summarize(results),
        "failed_case_ids": [item["case_id"] for item in results if not item["passed"]],
    }
    print("synthetic safe_default evaluation")
    print(json.dumps(report, indent=2, ensure_ascii=True))

    artifact_path = Path("artifacts") / "synthetic_safe_default_eval.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"artifact_path={artifact_path.as_posix()}")


if __name__ == "__main__":
    main()
