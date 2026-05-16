from __future__ import annotations

import json
import os
import statistics
import sys
import time
import argparse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_rag import build_index, generate_answer, hash_text, retrieve
from upstage_tracker import UpstageTracker


QUESTIONS = [
    {"id": "M_001", "question": "Who owns Project Alpha?", "token": "measure-1"},
    {"id": "M_002", "question": "What is the Project Alpha launch date?", "token": "measure-2"},
    {"id": "M_003", "question": "What is the API key rotation policy?", "token": "measure-3"},
    {"id": "M_004", "question": "What does the Project Alpha schedule say about audit protocol?", "token": "measure-4"},
]


class MeasurementTracker:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def chat(
        self,
        question_id: str,
        messages: list[dict[str, str]],
        token: str,
        system_prompt: str | None = None,
        **_: object,
    ) -> str:
        start = time.perf_counter()
        answer = "synthetic answer"
        elapsed = time.perf_counter() - start
        self.records.append(
            {
                "question_id": question_id,
                "answer": answer,
                "used_tokens": 1,
                "inference_time": round(elapsed, 6),
                "token": token,
                "message_count": len(messages),
                "has_system_prompt": system_prompt is not None,
            }
        )
        return answer


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, round((len(ordered) - 1) * pct))
    return ordered[rank]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(values), 6) if values else 0.0,
        "p95": round(percentile(values, 0.95), 6),
    }


def measure_retrieval(index) -> tuple[dict[str, float], list[int], list[int]]:
    timings: list[float] = []
    context_lengths: list[int] = []
    retrieved_counts: list[int] = []

    for item in QUESTIONS:
        start = time.perf_counter()
        context = retrieve(item["question"], index)
        timings.append(time.perf_counter() - start)
        context_lengths.append(len(context))
        retrieved_counts.append(context.count("[DOC "))

    return summarize(timings), context_lengths, retrieved_counts


def measure_test_double_e2e(index) -> dict[str, float]:
    tracker = MeasurementTracker()
    timings: list[float] = []

    for item in QUESTIONS:
        start = time.perf_counter()
        context = retrieve(item["question"], index)
        generate_answer(item["question"], context, tracker, item["id"], item["token"])
        timings.append(time.perf_counter() - start)

    return summarize(timings)


def measure_real_api_if_available(index) -> dict[str, Any] | None:
    if not os.environ.get("UPSTAGE_API_KEY"):
        return None

    tracker = UpstageTracker()
    timings: list[float] = []
    try:
        for item in QUESTIONS[:1]:
            start = time.perf_counter()
            context = retrieve(item["question"], index)
            generate_answer(item["question"], context, tracker, item["id"], item["token"])
            timings.append(time.perf_counter() - start)
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error_type": type(exc).__name__}
    return {"status": "measured", **summarize(timings)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="distribution/corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_index(args.corpus_dir)
    retrieval_timing, context_lengths, retrieved_counts = measure_retrieval(index)
    test_double_timing = measure_test_double_e2e(index)
    real_api_timing = measure_real_api_if_available(index)

    result = {
        "corpus": {
            "corpus_dir": args.corpus_dir,
            "parsed_document_count": index.metadata["document_count"],
            "chunk_count": index.metadata["chunk_count"],
            "failed_document_count": index.metadata["failed_document_count"],
            "parse_warning_count": index.metadata["parse_warning_count"],
            "table_like_chunk_count": index.metadata["table_like_chunk_count"],
            "detected_message_count": index.metadata.get("detected_message_count", 0),
            "suspicious_span_count": index.metadata.get("suspicious_span_count", 0),
            "high_risk_pii_count": index.metadata.get("high_risk_pii_count", 0),
        },
        "retrieval_context_pack_seconds": retrieval_timing,
        "test_double_end_to_end_seconds": test_double_timing,
        "real_api_end_to_end_seconds": real_api_timing,
        "average_context_length": round(statistics.mean(context_lengths), 2),
        "average_retrieved_chunk_count": round(statistics.mean(retrieved_counts), 2),
        "question_hashes": [hash_text(item["question"]) for item in QUESTIONS],
    }

    print("safe_default baseline summary")
    print(json.dumps(result, indent=2, ensure_ascii=True))

    artifact_path = Path("artifacts") / "safe_default_baseline.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"artifact_path={artifact_path.as_posix()}")


if __name__ == "__main__":
    main()
