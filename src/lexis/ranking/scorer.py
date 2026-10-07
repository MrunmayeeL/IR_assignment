"""Scorer module for free-text ranked retrieval and candidate pruning."""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.ranking.tfidf import TFIDFFactory
from lexis.ranking.zones import ZoneWeights
from lexis.ranking.normalization import LengthNormalizer
from lexis.ranking.champion_lists import ChampionLists
from lexis.ranking.impact_ordering import ImpactOrdering


@dataclass
class SearchResult:
    doc_name: str
    doc_id: int
    score: float


@dataclass
class PruningMetrics:
    method: str
    query: str
    num_scored: int
    latency_ms: float
    overlap_at_10: float
    top_10_results: List[Tuple[str, float]]


class RankedScorer:
    """Configurable ranked retrieval scorer supporting baselines, full system, and pruning."""

    def __init__(
        self,
        index: PositionalIndex,
        tokenizer: Tokenizer,
        title_weight: float = 0.3,
        body_weight: float = 0.7,
        use_idf: bool = True,
        use_length_norm: bool = True,
        use_zones: bool = True,
        use_log_tf: bool = True,
    ):
        self.index = index
        self.tokenizer = tokenizer
        self.zone_weights = ZoneWeights(
            title_weight=title_weight,
            body_weight=body_weight,
            enabled=use_zones,
        )
        self.use_idf = use_idf
        self.use_length_norm = use_length_norm
        self.use_log_tf = use_log_tf

        self._champion_lists: Optional[ChampionLists] = None
        self._impact_ordering: Optional[ImpactOrdering] = None

    def get_champion_lists(self, r: int = 50) -> ChampionLists:
        if self._champion_lists is None:
            self._champion_lists = ChampionLists(self.index, r=r)
        return self._champion_lists

    def get_impact_ordering(self, max_per_term: int = 100) -> ImpactOrdering:
        if self._impact_ordering is None:
            self._impact_ordering = ImpactOrdering(self.index, max_postings_per_term=max_per_term)
        return self._impact_ordering

    def get_candidates(self, query_terms: List[str], method: str = "exhaustive") -> Set[int]:
        """Generate candidate document IDs according to the selected strategy."""
        if method == "exhaustive":
            candidates: Set[int] = set()
            for t in query_terms:
                plist = self.index.get_postings(t)
                if plist:
                    candidates.update(plist.doc_ids())
            return candidates

        elif method == "champion_lists":
            cl = self.get_champion_lists()
            return cl.get_candidates(query_terms)

        elif method == "impact_ordering":
            io = self.get_impact_ordering()
            return io.get_candidates(query_terms)

        else:
            raise ValueError(f"Unknown candidate generation method: {method}")

    def score_query(
        self,
        query_str: str,
        top_k: int = 10,
        method: str = "exhaustive",
    ) -> Tuple[List[SearchResult], int, float]:
        """Score query against collection and return (top_k_results, num_scored, latency_ms)."""
        t0 = time.perf_counter()
        query_terms = self.tokenizer.tokenize_terms(query_str)
        if not query_terms:
            return [], 0, 0.0

        candidates = self.get_candidates(query_terms, method=method)
        total_docs = self.index.doc_count()

        # Compute query vector
        doc_freqs = {
            t: (self.index.dictionary.get(t).doc_frequency if self.index.dictionary.get(t) else 0)
            for t in query_terms
        }
        q_vec = TFIDFFactory.compute_query_vector(
            query_terms=query_terms,
            total_docs=total_docs,
            doc_freqs=doc_freqs,
            use_idf=self.use_idf,
            use_log_tf=self.use_log_tf,
        )
        q_norm = LengthNormalizer.vector_norm(q_vec)

        # Pre-fetch postings for query terms
        term_postings: Dict[str, dict] = {}
        for t in q_vec.keys():
            plist = self.index.get_postings(t)
            if plist:
                term_postings[t] = {p.doc_id: p for p in plist.postings}

        scores: Dict[int, float] = {}

        for doc_id in candidates:
            # Check for tombstone deletion
            if self.index.document_registry.is_deleted(doc_id):
                continue

            dot_product = 0.0
            for term, q_weight in q_vec.items():
                if term in term_postings and doc_id in term_postings[term]:
                    p = term_postings[term][doc_id]
                    # Compute document term weight
                    doc_tf = self.zone_weights.compute_weighted_tf(
                        title_tf=p.title_tf,
                        body_tf=p.body_tf,
                        total_tf=p.tf,
                        use_log_tf=self.use_log_tf,
                    )
                    idf_val = TFIDFFactory.idf(total_docs, doc_freqs.get(term, 0)) if self.use_idf else 1.0
                    d_weight = doc_tf * idf_val
                    dot_product += q_weight * d_weight

            if dot_product > 0:
                doc_len = self.index.document_registry.doc_lengths.get(doc_id, 1)
                doc_norm = LengthNormalizer.doc_length_norm(doc_len, enabled=self.use_length_norm)
                final_score = dot_product / (q_norm * doc_norm)
                scores[doc_id] = final_score

        # Deterministic sorting: score descending, then doc_name ascending
        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], self.index.get_doc_name(x[0]) or ""),
        )

        results: List[SearchResult] = []
        for did, score in ranked[:top_k]:
            name = self.index.get_doc_name(did)
            if name:
                results.append(SearchResult(doc_name=name, doc_id=did, score=round(score, 6)))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, len(candidates), latency_ms

    def compare_pruning(
        self,
        query_str: str,
        top_k: int = 10,
    ) -> Dict[str, PruningMetrics]:
        """Compare exhaustive, champion_lists, and impact_ordering on the same query."""
        methods = ["exhaustive", "champion_lists", "impact_ordering"]
        results_by_method = {}
        top_docs_by_method = {}

        for m in methods:
            res, num_scored, lat = self.score_query(query_str, top_k=top_k, method=m)
            results_by_method[m] = (res, num_scored, lat)
            top_docs_by_method[m] = [r.doc_name for r in res]

        exhaustive_set = set(top_docs_by_method["exhaustive"])
        metrics = {}

        for m in methods:
            res, num_scored, lat = results_by_method[m]
            top_set = set(top_docs_by_method[m])
            overlap = len(exhaustive_set.intersection(top_set)) / max(1, len(exhaustive_set))

            metrics[m] = PruningMetrics(
                method=m,
                query=query_str,
                num_scored=num_scored,
                latency_ms=round(lat, 2),
                overlap_at_10=round(overlap, 4),
                top_10_results=[(r.doc_name, r.score) for r in res],
            )

        return metrics
