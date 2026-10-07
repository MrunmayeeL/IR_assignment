"""Ranking evaluation metrics: Precision@10, MAP, and NDCG@10."""

import math
from typing import Dict, List, Set, Tuple


class RankingMetrics:
    """Calculates information retrieval ranking metrics over retrieved and graded documents."""

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int = 10) -> float:
        """Compute Precision@K."""
        if k <= 0:
            return 0.0
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for doc in top_k if doc in relevant_docs)
        return hits / float(k)

    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Compute Average Precision (AP) for a single query."""
        if not relevant_docs:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_docs:
                hits += 1
                sum_precisions += hits / float(rank)

        return sum_precisions / float(len(relevant_docs))

    @staticmethod
    def mean_average_precision(
        query_results: Dict[str, List[str]],
        query_qrels: Dict[str, Set[str]],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute Mean Average Precision (MAP) across queries.
        
        Returns (mean_ap, per_query_ap).
        """
        if not query_results:
            return 0.0, {}

        per_query_ap: Dict[str, float] = {}
        for qid, docs in query_results.items():
            rel_docs = query_qrels.get(qid, set())
            ap = RankingMetrics.average_precision(docs, rel_docs)
            per_query_ap[qid] = ap

        mean_ap = sum(per_query_ap.values()) / max(1, len(per_query_ap))
        return mean_ap, per_query_ap

    @staticmethod
    def dcg_at_k(retrieved_docs: List[str], grades: Dict[str, int], k: int = 10) -> float:
        """Compute Discounted Cumulative Gain at rank K using standard exponential gain."""
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], start=1):
            rel = grades.get(doc, 0)
            if rel > 0:
                gain = (2.0 ** rel) - 1.0
                discount = math.log2(i + 1.0)
                dcg += gain / discount
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved_docs: List[str],
        grades: Dict[str, int],
        k: int = 10,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain at rank K (NDCG@K)."""
        actual_dcg = RankingMetrics.dcg_at_k(retrieved_docs, grades, k=k)

        # Ideal ranking: sort all relevant documents by grade descending
        ideal_grades = sorted(grades.values(), reverse=True)
        if not ideal_grades or ideal_grades[0] == 0:
            return 0.0

        idcg = 0.0
        for i, rel in enumerate(ideal_grades[:k], start=1):
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(i + 1.0)
            idcg += gain / discount

        if idcg <= 0.0:
            return 0.0
        return actual_dcg / idcg
