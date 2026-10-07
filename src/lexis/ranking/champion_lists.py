"""Champion lists candidate pruning (Chapter 7)."""

import time
from typing import Dict, List, Set, Tuple
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList, Posting
from lexis.ranking.tfidf import TFIDFFactory


class ChampionLists:
    """Maintains precomputed top-r champion documents for each vocabulary term."""

    def __init__(self, index: PositionalIndex, r: int = 50):
        self.index = index
        self.r = r
        self.champion_lists: Dict[str, List[int]] = {}
        self._build_champion_lists()

    def _build_champion_lists(self):
        """Precompute top-r documents for each term based on term frequency."""
        for term in self.index.vocabulary():
            plist = self.index.get_postings(term)
            if plist and plist.postings:
                # Rank postings by TF descending, tie-break by doc_id
                sorted_postings = sorted(plist.postings, key=lambda p: (-p.tf, p.doc_id))
                self.champion_lists[term] = [p.doc_id for p in sorted_postings[: self.r]]

    def get_candidates(self, query_terms: List[str], min_candidates: int = 10) -> Set[int]:
        """Union champion lists for query terms. If too few, supplement with full postings."""
        candidates: Set[int] = set()
        for t in query_terms:
            if t in self.champion_lists:
                candidates.update(self.champion_lists[t])

        if len(candidates) < min_candidates:
            for t in query_terms:
                plist = self.index.get_postings(t)
                if plist:
                    candidates.update(plist.doc_ids())
                    if len(candidates) >= min_candidates * 2:
                        break

        return candidates
