"""Wildcard query expansion using a Permuterm Index."""

import re
from bisect import bisect_left
from typing import List, Set, Dict, Tuple
from lexis.index.positional_index import PositionalIndex


class PermutermIndex:
    """Permuterm index for fast vocabulary lookup of wildcards with one or more '*'."""

    def __init__(self, vocabulary: List[str]):
        # Map: permuterm_key -> original_term
        self.entries: List[Tuple[str, str]] = []
        for term in vocabulary:
            w = term.lower() + "$"
            for i in range(len(w)):
                rotation = w[i:] + w[:i]
                self.entries.append((rotation, term))

        # Sort lexicographically by rotation
        self.entries.sort(key=lambda x: x[0])
        self.keys = [e[0] for e in self.entries]

    def _prefix_search(self, prefix: str) -> List[str]:
        """Find all original terms whose rotation starts with prefix."""
        start = bisect_left(self.keys, prefix)
        matches: Set[str] = set()
        for i in range(start, len(self.keys)):
            if self.keys[i].startswith(prefix):
                matches.add(self.entries[i][1])
            else:
                break
        return sorted(matches)

    def expand_pattern(self, pattern: str) -> List[str]:
        """Expand wildcard pattern into matching vocabulary terms."""
        pat = pattern.lower()

        # Handle simple or multi-wildcard patterns
        parts = pat.split("*")

        if len(parts) == 1:
            # No wildcards
            return self._prefix_search(pat + "$")

        if len(parts) == 2:
            # Single wildcard: prefix* or *suffix or prefix*suffix
            p1, p2 = parts[0], parts[1]
            if p1 and not p2:
                # e.g. econom* -> rotate to $econom*
                prefix_query = "$" + p1
                return self._prefix_search(prefix_query)
            elif not p1 and p2:
                # e.g. *port -> rotate to port$*
                prefix_query = p2 + "$"
                return self._prefix_search(prefix_query)
            elif p1 and p2:
                # e.g. pet*chem -> rotate to chem$pet*
                prefix_query = p2 + "$" + p1
                return self._prefix_search(prefix_query)
            else:
                # Just '*'
                return self._prefix_search("")

        # Multiple wildcards: e.g. pet*chem* or *a*b*
        # Use first and last anchor to query permuterm, then regex filter candidates
        first = parts[0]
        last = parts[-1]

        if first and last:
            prefix_query = last + "$" + first
            candidates = self._prefix_search(prefix_query)
        elif first:
            prefix_query = "$" + first
            candidates = self._prefix_search(prefix_query)
        elif last:
            prefix_query = last + "$"
            candidates = self._prefix_search(prefix_query)
        else:
            # Middle component
            middle = parts[1]
            candidates = self._prefix_search(middle)

        # Build regex for full pattern
        regex_pattern = "^" + ".*".join(re.escape(p) for p in parts) + "$"
        regex = re.compile(regex_pattern)
        filtered = [c for c in candidates if regex.match(c)]
        return sorted(set(filtered))


class WildcardEvaluator:
    """Evaluates wildcard queries by expanding vocabulary terms and uniting postings."""

    def __init__(self, index: PositionalIndex):
        self.index = index
        self._permuterm = None

    def _get_permuterm(self) -> PermutermIndex:
        if self._permuterm is None:
            self._permuterm = PermutermIndex(self.index.vocabulary())
        return self._permuterm

    def expand(self, pattern: str) -> List[str]:
        permuterm = self._get_permuterm()
        return permuterm.expand_pattern(pattern)

    def evaluate(self, pattern: str) -> List[int]:
        matching_terms = self.expand(pattern)
        matched_doc_ids: Set[int] = set()

        for term in matching_terms:
            plist = self.index.get_postings(term)
            if plist:
                matched_doc_ids.update(plist.doc_ids())

        return sorted(matched_doc_ids)
