"""Phrase query execution using positional inverted postings."""

from typing import List, Set
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList, Posting
from lexis.preprocessing.tokenizer import Tokenizer


class PhraseEvaluator:
    """Evaluates multi-word quoted phrase queries requiring consecutive token positions."""

    def __init__(self, index: PositionalIndex, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer

    def evaluate(self, phrase_str: str) -> List[int]:
        """Find doc_ids containing the exact phrase in consecutive order."""
        terms = self.tokenizer.tokenize_terms(phrase_str)
        if not terms:
            return []
        if len(terms) == 1:
            plist = self.index.get_postings(terms[0])
            return plist.doc_ids() if plist else []

        # Fetch postings for first term
        p1 = self.index.get_postings(terms[0])
        if not p1 or not p1.postings:
            return []

        # Current match candidates: map doc_id -> list of end positions of phrase so far
        candidates: dict[int, List[int]] = {p.doc_id: list(p.positions) for p in p1.postings}

        for term in terms[1:]:
            p2 = self.index.get_postings(term)
            if not p2 or not p2.postings:
                return []

            next_candidates: dict[int, List[int]] = {}
            p2_map = {p.doc_id: set(p.positions) for p in p2.postings}

            for doc_id, prev_positions in candidates.items():
                if doc_id in p2_map:
                    target_positions = p2_map[doc_id]
                    # Check for consecutive positions: pos + 1
                    valid_next = [pos + 1 for pos in prev_positions if (pos + 1) in target_positions]
                    if valid_next:
                        next_candidates[doc_id] = valid_next

            candidates = next_candidates
            if not candidates:
                break

        return sorted(candidates.keys())
