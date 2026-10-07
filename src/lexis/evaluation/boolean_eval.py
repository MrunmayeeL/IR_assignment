"""Evaluation suite for exact retrieval Boolean, phrase, wildcard, and proximity tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set
from lexis.query.query_engine import QueryEngine


@dataclass
class BooleanTestResult:
    query_id: str
    query: str
    test_type: str
    expected_count: int
    actual_count: int
    exact_match: bool
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


class BooleanEvaluatorRunner:
    """Runs and verifies test cases from boolean_tests.tsv against the search engine."""

    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine

    def run_tests_from_file(self, tests_tsv_path: str) -> List[BooleanTestResult]:
        p = Path(tests_tsv_path)
        if not p.exists():
            raise FileNotFoundError(f"Boolean tests file not found: {tests_tsv_path}")

        results: List[BooleanTestResult] = []

        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return results

        # Header: query_id, query, type, num_results, expected_doc_ids
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue

            qid = parts[0]
            query_str = parts[1]
            qtype = parts[2]
            expected_count = int(parts[3])
            expected_ids = set(parts[4].split(",")) if parts[4] else set()

            actual_docs = self.query_engine.search_exact(query_str)
            actual_set = set(actual_docs)

            intersection = expected_ids.intersection(actual_set)
            precision = len(intersection) / max(1, len(actual_set))
            recall = len(intersection) / max(1, len(expected_ids))
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            exact = actual_set == expected_ids
            fp = len(actual_set - expected_ids)
            fn = len(expected_ids - actual_set)

            results.append(
                BooleanTestResult(
                    query_id=qid,
                    query=query_str,
                    test_type=qtype,
                    expected_count=expected_count,
                    actual_count=len(actual_set),
                    exact_match=exact,
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    f1=round(f1, 4),
                    false_positives=fp,
                    false_negatives=fn,
                )
            )

        return results
