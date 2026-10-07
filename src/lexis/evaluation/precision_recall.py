"""11-point interpolated precision-recall curve calculation."""

from typing import Dict, List, Set, Tuple


class PrecisionRecallCurve:
    """Calculates standard 11-point interpolated precision-recall curves."""

    RECALL_LEVELS = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0

    @classmethod
    def compute_single_query(
        cls,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
    ) -> List[float]:
        """Compute interpolated precision at the 11 standard recall levels for a single query."""
        if not relevant_docs:
            return [0.0] * 11

        total_rel = len(relevant_docs)
        precisions: List[float] = []
        recalls: List[float] = []
        hits = 0

        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_docs:
                hits += 1
                recalls.append(hits / float(total_rel))
                precisions.append(hits / float(rank))

        if not recalls:
            return [0.0] * 11

        # Interpolated precision at recall level r is max precision at any recall >= r
        interpolated: List[float] = []
        for r_target in cls.RECALL_LEVELS:
            valid_p = [p for p, r in zip(precisions, recalls) if r >= r_target]
            p_interp = max(valid_p) if valid_p else 0.0
            interpolated.append(round(p_interp, 4))

        return interpolated

    @classmethod
    def compute_aggregate(
        cls,
        query_results: Dict[str, List[str]],
        query_qrels: Dict[str, Set[str]],
    ) -> List[Tuple[float, float]]:
        """Average 11-point precision across all queries.
        
        Returns list of (recall_level, mean_interpolated_precision).
        """
        if not query_results:
            return [(r, 0.0) for r in cls.RECALL_LEVELS]

        num_queries = len(query_results)
        sums = [0.0] * 11

        for qid, docs in query_results.items():
            rel_docs = query_qrels.get(qid, set())
            curve = cls.compute_single_query(docs, rel_docs)
            for i in range(11):
                sums[i] += curve[i]

        means = [round(s / float(num_queries), 4) for s in sums]
        return list(zip(cls.RECALL_LEVELS, means))
