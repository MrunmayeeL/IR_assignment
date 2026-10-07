"""Impact ordering candidate generation and early termination (Chapter 7)."""

import time
from typing import Dict, List, Set, Tuple
from lexis.index.positional_index import PositionalIndex
from lexis.ranking.tfidf import TFIDFFactory


class ImpactOrdering:
    """Processes postings ordered by term impact with early termination."""

    def __init__(self, index: PositionalIndex, max_postings_per_term: int = 100):
        self.index = index
        self.max_postings_per_term = max_postings_per_term

    def get_candidates(self, query_terms: List[str], max_total_candidates: int = 200) -> Set[int]:
        """Collect candidates by scanning postings ordered by decreasing impact (TF)."""
        candidates: Set[int] = set()
        total_docs = self.index.doc_count()

        # Sort query terms by IDF descending so rarest (most discriminative) terms come first
        term_idfs = []
        for t in query_terms:
            tinfo = self.index.dictionary.get(t)
            df = tinfo.doc_frequency if tinfo else 0
            idf = TFIDFFactory.idf(total_docs, df)
            term_idfs.append((t, idf))

        term_idfs.sort(key=lambda x: -x[1])

        for term, _ in term_idfs:
            plist = self.index.get_postings(term)
            if plist and plist.postings:
                # Sort postings by impact (tf descending)
                impact_sorted = sorted(plist.postings, key=lambda p: -p.tf)
                for p in impact_sorted[: self.max_postings_per_term]:
                    candidates.add(p.doc_id)
                    if len(candidates) >= max_total_candidates:
                        return candidates

        return candidates
