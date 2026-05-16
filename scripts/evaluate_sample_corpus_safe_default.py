from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_rag import build_index, decide_selective_multihop, detect_query_flags, retrieve, retrieve_chunks


SAMPLE_CORPUS_DIR = "sample_corpus/enron"
HEADER_LOOKUP_QUERIES = [
    ("sender", "Who sent this message?"),
    ("recipient", "Who received this message?"),
    ("sent_date", "When was this message sent?"),
    ("subject", "What is the subject of this message?"),
    ("file", "Which mailbox contains this message?"),
]
FORWARDED_QUERY = "Who originally sent this forwarded message?"
MEETING_QUERY = "Which office location is mentioned in this meeting message?"
DIRECT_PII_QUERIES = [
    "What is the sender email address for this message?",
    "What is the phone number in this message?",
    "What is the password in this message?",
]


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": round(statistics.median(ordered), 6),
        "p95": round(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))], 6),
    }


def evaluate_header_lookup(index) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, int]] = {}
    for category, question in HEADER_LOOKUP_QUERIES:
        flags = detect_query_flags(question)
        chunks = retrieve_chunks(question, index, index.config.max_chunks, flags)
        top_chunk = chunks[0] if chunks else None
        if category == "sender":
            hit = bool(top_chunk and "sender" in top_chunk.header_fields_present)
        elif category == "recipient":
            hit = bool(top_chunk and "recipients" in top_chunk.header_fields_present)
        elif category == "sent_date":
            hit = bool(top_chunk and "sent" in top_chunk.header_fields_present)
        elif category == "subject":
            hit = bool(top_chunk and "subject" in top_chunk.header_fields_present)
        else:
            hit = bool(top_chunk and "file" in top_chunk.header_fields_present)
        results[category] = {"hit": int(hit), "miss": int(not hit)}
    return results


def evaluate_direct_pii(index) -> dict[str, int]:
    passed = 0
    failed = 0
    for question in DIRECT_PII_QUERIES:
        flags = detect_query_flags(question)
        context = retrieve(question, index)
        success = flags.asks_direct_pii and "[REDACTED_" in context
        passed += int(success)
        failed += int(not success)
    return {"pass": passed, "fail": failed}


def evaluate_retrieval_metrics(index) -> dict[str, Any]:
    timings: list[float] = []
    context_lengths: list[int] = []
    multihop_runs = 0
    query_count = 0
    generic_questions = [
        question for _, question in HEADER_LOOKUP_QUERIES
    ] + [FORWARDED_QUERY, MEETING_QUERY, *DIRECT_PII_QUERIES]

    for question in generic_questions:
        start = time.perf_counter()
        context = retrieve(question, index)
        timings.append(time.perf_counter() - start)
        context_lengths.append(len(context))
        flags = detect_query_flags(question)
        first_pass_chunks = retrieve_chunks(question, index, index.config.max_chunks, flags)
        decision = decide_selective_multihop(flags, index.config, first_pass_chunks)
        multihop_runs += int(decision.should_run_second_pass)
        query_count += 1

    return {
        "retrieval_context_seconds": summarize(timings),
        "average_context_length": round(statistics.mean(context_lengths), 2) if context_lengths else 0.0,
        "multihop_execution_rate_hint_level": round(multihop_runs / query_count, 4) if query_count else 0.0,
    }


def main() -> None:
    index = build_index(SAMPLE_CORPUS_DIR)
    header_lookup = evaluate_header_lookup(index)
    direct_pii = evaluate_direct_pii(index)
    retrieval_metrics = evaluate_retrieval_metrics(index)

    report = {
        "corpus": {
            "parsed_docs": index.metadata["document_count"],
            "detected_messages": index.metadata.get("detected_message_count", 0),
            "chunks": index.metadata["chunk_count"],
            "parser_failures": index.metadata["failed_document_count"],
            "parse_warnings": index.metadata["parse_warning_count"],
            "suspicious_span_count": index.metadata.get("suspicious_span_count", 0),
            "high_risk_pii_count": index.metadata.get("high_risk_pii_count", 0),
        },
        "header_lookup": header_lookup,
        "forwarded_origin_lookup": {
            "hit": int(
                bool(
                    (forwarded_chunks := retrieve_chunks(
                        FORWARDED_QUERY,
                        index,
                        index.config.max_chunks,
                        detect_query_flags(FORWARDED_QUERY),
                    ))
                    and forwarded_chunks[0].has_forwarded_chain
                )
            ),
            "miss": int(
                not bool(
                    (forwarded_chunks := retrieve_chunks(
                        FORWARDED_QUERY,
                        index,
                        index.config.max_chunks,
                        detect_query_flags(FORWARDED_QUERY),
                    ))
                    and forwarded_chunks[0].has_forwarded_chain
                )
            ),
        },
        "meeting_schedule_location_lookup": {
            "hit": int(
                bool(
                    retrieve_chunks(
                        MEETING_QUERY,
                        index,
                        index.config.max_chunks,
                        detect_query_flags(MEETING_QUERY),
                    )
                )
            ),
            "miss": int(
                not bool(
                    retrieve_chunks(
                        MEETING_QUERY,
                        index,
                        index.config.max_chunks,
                        detect_query_flags(MEETING_QUERY),
                    )
                )
            ),
        },
        "direct_pii_protection": direct_pii,
        **retrieval_metrics,
    }

    print("sample corpus safe_default evaluation")
    print(json.dumps(report, indent=2, ensure_ascii=True))

    artifact_path = Path("artifacts") / "sample_corpus_safe_default_eval.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"artifact_path={artifact_path.as_posix()}")


if __name__ == "__main__":
    main()
