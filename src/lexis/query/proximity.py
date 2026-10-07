"""Proximity query execution using positional inverted postings."""

from typing import List, Set
from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer


class ProximityEvaluator:
    """Evaluates proximity queries of the form: term1 /N term2."""

    def __init__(self, index: PositionalIndex, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer

    def evaluate(self, term1_raw: str, distance: int, term2_raw: str) -> List[int]:
        """Find all doc_ids where term1 and term2 occur within distance N tokens."""
        t1_list = self.tokenizer.tokenize_terms(term1_raw)
        t2_list = self.tokenizer.tokenize_terms(term2_raw)

        if not t1_list or not t2_list:
            return []

        t1 = t1_list[0]
        t2 = t2_list[0]

        p1 = self.index.get_postings(t1)
        p2 = self.index.get_postings(t2)

        if not p1 or not p2 or not p1.postings or not p2.postings:
            return []

        matching_doc_ids: List[int] = []

        # Find intersecting doc_ids
        p1_map = {p.doc_id: p.positions for p in p1.postings}
        p2_map = {p.doc_id: p.positions for p in p2.postings}

        common_doc_ids = sorted(set(p1_map.keys()).intersection(p2_map.keys()))

        for doc_id in common_doc_ids:
            pos1 = p1_map[doc_id]
            pos2 = p2_map[doc_id]

            # Check if any pair satisfies 1 <= |pos1 - pos2| <= distance
            found = False
            i = 0
            j = 0
            len1 = len(pos1)
            len2 = len(pos2)

            while i < len1 and j < len2:
                diff = pos1[i] - pos2[j]
                if 1 <= abs(diff) <= distance:
                    found = True
                    break
                if pos1[i] < pos2[j]:
                    i += 1
                else:
                    j += 1

            if found:
                matching_doc_ids.append(doc_id)

        return matching_doc_ids
